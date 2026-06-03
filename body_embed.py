#!/usr/bin/env python3
"""
Generates body-shape embeddings from image URLs using MediaPipe Pose (Tasks API).

Where face_embed.py captures the *face*, this captures *build/silhouette*: it
detects 33 body landmarks and derives scale- and position-invariant proportion
ratios (shoulder/hip breadth, torso vs leg length, etc.). Two people with the
same proportions get similar vectors regardless of image size, crop, or distance.

Usage:  python body_embed.py <url1> [url2 ...]
Output: a JSON array, one entry per URL, in order:
    [{"body": [...ratios...]}, {"error": "No pose detected"}, ...]

Downloads the ~5 MB pose model on first run (cached in ~/.luminary).
"""
import sys
import os
import json
import math
import tempfile
import urllib.request

import numpy as np

os.environ.setdefault("GLOG_minloglevel", "3")

# Landmarks come from the lite pose model. The silhouette mask (seg mode) comes
# from a *separate* ImageSegmenter — the pose model's own segmentation output
# hard-crashes the MediaPipe runtime in this environment, regardless of variant.
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
SEG_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)


def _cached(url):
    cache = os.path.join(os.path.expanduser("~"), ".luminary")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def download(url):
    ext = url.split(".")[-1].split("?")[0].lower()
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Luminary)"})
    with urllib.request.urlopen(req, timeout=30) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midp(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# Landmarks the proportion vector depends on. If any of these isn't actually
# visible, MediaPipe *extrapolates* it off-frame and the ratios become garbage —
# so a headshot/crop must be rejected, not silently turned into a fake build.
KEY_LANDMARKS = [11, 12, 23, 24, 25, 26, 27, 28]  # shoulders, hips, knees, ankles
MIN_VISIBILITY = 0.5


def is_full_body(lm):
    """True only when the joints the vector needs are genuinely in frame.
    Uses MediaPipe's per-landmark visibility — this is our headshot/crop filter."""
    vis = [getattr(lm[i], "visibility", None) for i in KEY_LANDMARKS]
    known = [v for v in vis if v is not None]
    if not known:
        return True  # visibility unsupported — don't over-reject
    # Treat a missing score as not-visible so partial crops can't sneak through.
    return min(0.0 if v is None else v for v in vis) >= MIN_VISIBILITY


def is_upright(lm):
    """True only for a roughly standing pose. The ratios assume a vertical body;
    a reclining/sitting/action frame distorts torso & leg lengths even when every
    joint is visible. Image y grows downward, so standing ⇒ shoulders above hips
    above knees above ankles."""
    sh_y = (lm[11].y + lm[12].y) / 2
    hip_y = (lm[23].y + lm[24].y) / 2
    knee_y = (lm[25].y + lm[26].y) / 2
    ank_y = (lm[27].y + lm[28].y) / 2
    return sh_y < hip_y < knee_y < ank_y


def classify_view(lm):
    """Pose-determinable view: 'side' (profile — the shoulders collapse in x) vs
    'frontal' (facing the camera-axis, front *or* rear).

    Front-vs-rear is deliberately NOT decided here: MediaPipe reports face-landmark
    visibility as high even when the face is turned away, so pose can't tell them
    apart. The ingest splits 'frontal' into front/rear using its face pass — a
    detected face means front, its absence (on a full body) means rear.
    """
    sw = abs(lm[11].x - lm[12].x)  # shoulder breadth (normalised)
    return "side" if sw < 0.08 else "frontal"


def build_vector(lm):
    """Scale-invariant body-proportion ratios from 33 pose landmarks.
    Returns None for cropped or non-standing poses so they don't skew a centroid."""
    if not is_full_body(lm) or not is_upright(lm):
        return None
    # A profile/side view collapses shoulder breadth, making every breadth ratio
    # meaningless — reject it here (build_proj_vector handles side frames instead).
    if abs(lm[11].x - lm[12].x) < 0.08:
        return None

    p = lambda i: (lm[i].x, lm[i].y)
    sh_w = d(p(11), p(12))           # shoulder breadth
    hip_w = d(p(23), p(24))          # hip breadth
    sh_mid, hip_mid = midp(p(11), p(12)), midp(p(23), p(24))
    torso = d(sh_mid, hip_mid)
    knee_mid = midp(p(25), p(26))
    ank_mid = midp(p(27), p(28))
    thigh = d(hip_mid, knee_mid)
    shin = d(knee_mid, ank_mid)
    leg = thigh + shin

    if torso < 1e-4 or sh_w < 1e-4 or hip_w < 1e-4:
        return None
    return [
        sh_w / torso,                          # shoulder breadth
        hip_w / torso,                         # hip breadth
        sh_w / hip_w,                          # >1 inverted-triangle, <1 pear/hourglass
        leg / torso,                           # leg length vs torso
        thigh / shin if shin > 1e-4 else 1.0,  # thigh:shin
    ]


def width_at(mask, y_norm, center_col):
    """Silhouette width (normalised 0..1) of the *central* body blob at a given
    vertical position — the contiguous run of person-pixels containing the body
    centerline. Using the central run (rather than the whole row's min..max)
    excludes detached arm/hand blobs that would otherwise inflate the width.
    Reads the outline, so it includes soft tissue — unlike skeletal landmarks."""
    h, w = mask.shape
    row = min(max(int(y_norm * h), 0), h - 1)
    line = mask[row] > 0.5
    c = min(max(center_col, 0), w - 1)
    if not line[c]:
        # Centerline isn't on the body (e.g. the gap between the legs) — snap to
        # the nearest person-pixel so we still measure a limb, not nothing.
        on = np.where(line)[0]
        if on.size == 0:
            return 0.0
        c = int(on[np.argmin(np.abs(on - c))])
    left = c
    while left > 0 and line[left - 1]:
        left -= 1
    right = c
    while right < w - 1 and line[right + 1]:
        right += 1
    return float(right - left) / w


def _min_visibility(lm, idxs):
    """Lowest MediaPipe visibility across the given landmarks, treating a missing
    score as not-visible. Returns 1.0 when visibility is unsupported (don't
    over-reject), mirroring is_full_body's handling."""
    vis = [getattr(lm[i], "visibility", None) for i in idxs]
    known = [v for v in vis if v is not None]
    if not known:
        return 1.0
    return min(0.0 if v is None else v for v in vis)


def build_seg_vector(lm, mask):
    """Lower-body *volume* ratios from the body silhouette. Where build_vector
    reads the skeleton, this reads the outline width at waist / hip / thigh — so
    it captures glute and thigh fullness (the thing pose & measurements miss).
    Scale-invariant: every width is divided by shoulder width.

    Works on a full-body standing frame (precise, knee-relative thigh sample) OR
    a legs-cropped rear/glute shot — the kind framed from the thighs up, where the
    legs run off the bottom — as long as the shoulders and hips are solid, falling
    back to a torso-relative thigh sample. So a tight butt shot (which fails the
    full-body gate) still yields a volume vector instead of nothing. Full-body
    frames are computed exactly as before, so the cached index needs no rebuild.
    Side/profile and non-upright frames are still rejected."""
    if mask is None:
        return None
    # Shoulders + hips must be reliable; the legs are optional (cropped shots).
    if _min_visibility(lm, (11, 12, 23, 24)) < MIN_VISIBILITY:
        return None

    mask = np.asarray(mask)
    if mask.ndim == 3:  # ImageSegmenter returns (H, W, 1)
        mask = mask[:, :, 0]
    _, w = mask.shape

    sh_y = (lm[11].y + lm[12].y) / 2
    hip_y = (lm[23].y + lm[24].y) / 2
    knee_y = (lm[25].y + lm[26].y) / 2
    sw = abs(lm[11].x - lm[12].x)  # shoulder width (normalised) — the scale ref
    # Require a roughly frontal/rear pose: a side/profile view collapses shoulder
    # width and makes every silhouette-width ratio explode.
    if sw < 0.08:
        return None
    torso = hip_y - sh_y
    if torso < 1e-4:  # shoulders must sit above the hips (upright torso)
        return None
    mid_col = int(((lm[23].x + lm[24].x) / 2) * w)  # body centerline (hip mid)

    waist_y = sh_y + 0.65 * torso  # just above the hips
    if _min_visibility(lm, (25, 26, 27, 28)) >= MIN_VISIBILITY:
        # Legs in frame: keep the strict gate + precise knee-relative thigh
        # sample, so full-body frames are scored exactly as before.
        if not is_upright(lm):
            return None
        thigh_y = hip_y + 0.35 * (knee_y - hip_y)
    else:
        # Legs cropped (framed-from-the-thighs butt shot): knee_y is extrapolated
        # junk, so anchor the thigh sample torso-relative instead.
        thigh_y = hip_y + 0.40 * torso

    waist_w = width_at(mask, waist_y, mid_col)
    hip_w = width_at(mask, hip_y, mid_col)
    thigh_w = width_at(mask, thigh_y, mid_col)
    if hip_w < 1e-4 or waist_w < 1e-4 or thigh_w < 1e-4:
        return None
    # Sanity bound: no real waist/hip/thigh is wider than ~2.5x the shoulders.
    # Anything bigger is arms *connected* to the torso (hands-on-hips/akimbo)
    # that the central-run trick can't separate — reject the frame.
    if max(waist_w, hip_w, thigh_w) > 2.5 * sw:
        return None

    return [
        waist_w / sw,        # waist breadth
        hip_w / sw,          # hip + glute breadth  (butt volume proxy)
        thigh_w / sw,        # thigh breadth        (thigh thickness proxy)
        hip_w / waist_w,     # visual waist-to-hip  (curviness from the outline)
        thigh_w / hip_w,     # thigh-to-hip balance
    ]


def build_proj_vector(lm, mask):
    """Posterior-*projection* ratios from a SIDE/profile silhouette — how far the
    lower body (buttocks) pushes back front-to-back, the thing a frontal width
    vector can't see (two builds with identical frontal hips can differ a lot in
    profile). Reads silhouette *depth* (the same central-run width_at, but in a
    profile that run spans front->back) at waist / glute / thigh height,
    normalised by torso *height* (shoulder->hip) — height stays meaningful in
    profile, where shoulder *width* collapses. Gated to profile, standing,
    full-body frames.

    First cut: the depth ratios capture relative lower-body projection, but the
    exact landmark heights and divisors want empirical tuning against real
    profile images. Returns None when the silhouette is unusable.
    """
    if mask is None or not is_full_body(lm) or not is_upright(lm):
        return None
    # Require a profile view; otherwise depth and width aren't separable here.
    if abs(lm[11].x - lm[12].x) >= 0.08:
        return None

    mask = np.asarray(mask)
    if mask.ndim == 3:  # ImageSegmenter returns (H, W, 1)
        mask = mask[:, :, 0]
    _, w = mask.shape

    sh_y = (lm[11].y + lm[12].y) / 2
    hip_y = (lm[23].y + lm[24].y) / 2
    knee_y = (lm[25].y + lm[26].y) / 2
    torso = abs(hip_y - sh_y)
    if torso < 1e-4:
        return None
    mid_col = int(((lm[23].x + lm[24].x) / 2) * w)  # hip centerline column

    waist_y = sh_y + 0.65 * (hip_y - sh_y)     # just above the hips
    glute_y = hip_y + 0.15 * (knee_y - hip_y)  # just below the hip joint
    thigh_y = hip_y + 0.45 * (knee_y - hip_y)  # upper thigh

    waist_d = width_at(mask, waist_y, mid_col)  # in profile, this run = depth
    glute_d = width_at(mask, glute_y, mid_col)
    thigh_d = width_at(mask, thigh_y, mid_col)
    if min(waist_d, glute_d, thigh_d) < 1e-4:
        return None

    return [
        glute_d / torso,    # glute-band depth, scale-free
        glute_d / waist_d,  # projection past the waist line — the "bubble" signal
        glute_d / thigh_d,  # glute vs upper-thigh depth
        waist_d / torso,    # waist depth (build context)
    ]


def debug_entry(url, lm, mask):
    """Per-URL diagnostic for --debug: every gate decision and the raw values
    behind it, so a rejected frame can be diagnosed (and the gates/measures
    tuned) from numbers rather than the image. Reports the landmark
    visibilities, the full-body / upright / profile gates, the silhouette
    samples, and what each of the three vectors came out to (null = its gate
    rejected the frame)."""
    names = {
        11: "L_shoulder", 12: "R_shoulder", 23: "L_hip", 24: "R_hip",
        25: "L_knee", 26: "R_knee", 27: "L_ankle", 28: "R_ankle",
    }
    vis = {
        f"{lbl}({i})": round(float(getattr(lm[i], "visibility", -1.0)), 3)
        for i, lbl in names.items()
    }
    sw = abs(lm[11].x - lm[12].x)
    sh_y = (lm[11].y + lm[12].y) / 2
    hip_y = (lm[23].y + lm[24].y) / 2
    knee_y = (lm[25].y + lm[26].y) / 2
    ank_y = (lm[27].y + lm[28].y) / 2

    m = None
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[:, :, 0]

    samples = {}
    if m is not None:
        _, w = m.shape
        mid_col = int(((lm[23].x + lm[24].x) / 2) * w)
        waist_y = sh_y + 0.65 * (hip_y - sh_y)
        glute_y = hip_y + 0.15 * (knee_y - hip_y)
        thigh_y = hip_y + 0.45 * (knee_y - hip_y)
        samples = {
            "waist": round(width_at(m, waist_y, mid_col), 4),
            "hip": round(width_at(m, hip_y, mid_col), 4),
            "glute": round(width_at(m, glute_y, mid_col), 4),
            "thigh": round(width_at(m, thigh_y, mid_col), 4),
        }

    return {
        "url": url.rsplit("/", 1)[-1],
        "pose_detected": True,
        "view": classify_view(lm),
        "shoulder_width_norm": round(sw, 4),
        "is_full_body": is_full_body(lm),
        "is_upright": is_upright(lm),
        "visibility": vis,
        "y_norm": {
            "shoulders": round(sh_y, 3),
            "hips": round(hip_y, 3),
            "knees": round(knee_y, 3),
            "ankles": round(ank_y, 3),
        },
        "mask_shape": list(m.shape) if m is not None else None,
        # In a frontal frame these central-run samples read as widths; in a
        # profile they read as front-to-back depths.
        "silhouette_central_run": samples,
        "pose_vector": build_vector(lm),
        "seg_vector": build_seg_vector(lm, mask),
        "proj_vector": build_proj_vector(lm, mask),
    }


def main():
    args = sys.argv[1:]
    # `--seg` switches from the pose/skeleton vector to the silhouette/volume
    # vector (butt & thigh fullness). Default stays pose for back-compat.
    seg_mode = False
    debug_mode = False
    # Flags are positional-first: `--seg` (silhouette/volume mode) and `--debug`
    # (per-URL diagnostic dump for tuning the gates/measures from numbers). Debug
    # implies seg because it needs the silhouette mask.
    while args and args[0] in ("--seg", "--debug"):
        if args[0] == "--seg":
            seg_mode = True
        else:
            debug_mode = True
            seg_mode = True
        args = args[1:]
    urls = args

    field = "seg" if seg_mode else "body"
    reject = (
        "Not a full-body standing pose (cropped/reclining)"
    )

    if not urls:
        print(json.dumps([{"error": "Usage: body_embed.py [--seg] [--debug] <url> [url ...]"}]))
        sys.exit(1)

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as e:
        print(json.dumps([{"error": f"Missing dependency: {e}. Run: pip install mediapipe"}] * len(urls)))
        sys.exit(1)

    try:
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_cached(POSE_MODEL_URL)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        landmarker = vision.PoseLandmarker.create_from_options(opts)
        # The silhouette mask comes from a dedicated segmenter (seg mode only).
        segmenter = None
        if seg_mode:
            seg_opts = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_cached(SEG_MODEL_URL)),
                output_confidence_masks=True,
            )
            segmenter = vision.ImageSegmenter.create_from_options(seg_opts)
    except Exception as e:
        print(json.dumps([{"error": f"Model init failed: {e}"}] * len(urls)))
        sys.exit(1)

    results = []
    for url in urls:
        tmp = None
        try:
            tmp = download(url)
            image = mp.Image.create_from_file(tmp)
            res = landmarker.detect(image)
            if debug_mode:
                if not res.pose_landmarks:
                    results.append(
                        {"url": url.rsplit("/", 1)[-1], "pose_detected": False}
                    )
                    continue
                seg_res = segmenter.segment(image)
                mask = (
                    seg_res.confidence_masks[0].numpy_view()
                    if seg_res.confidence_masks
                    else None
                )
                results.append(debug_entry(url, res.pose_landmarks[0], mask))
                continue
            if not res.pose_landmarks:
                results.append({"error": "No pose detected"})
                continue
            lm = res.pose_landmarks[0]
            if seg_mode:
                # Emit BOTH vectors: the pose/frame vector is free here (the
                # landmarks are already computed), so one --seg pass feeds both
                # frame and shape matching. Each may be None independently.
                seg_res = segmenter.segment(image)
                mask = (
                    seg_res.confidence_masks[0].numpy_view()
                    if seg_res.confidence_masks
                    else None
                )
                entry = {"view": classify_view(lm)}
                pose_vec = build_vector(lm)
                if pose_vec:
                    entry["body"] = pose_vec
                seg_vec = build_seg_vector(lm, mask)
                if seg_vec:
                    entry["seg"] = seg_vec
                proj_vec = build_proj_vector(lm, mask)
                if proj_vec:
                    entry["proj"] = proj_vec
                results.append(entry)
            else:
                vec = build_vector(lm)
                results.append({field: vec} if vec else {"error": reject})
        except Exception as e:
            results.append({"error": str(e)})
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    print(json.dumps(results, indent=2) if debug_mode else json.dumps(results))


if __name__ == "__main__":
    main()
