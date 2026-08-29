"""
image_to_signal.py — Digitizes a plotted ECG waveform image into a 1D signal.

Turns a screenshot/plot image of an ECG trace (a line plotted on a plain
background — e.g. a matplotlib export) into a 1D numpy array with one
amplitude value per pixel column, ordered left-to-right in time.

IMPORTANT LIMITATION — read before relying on this:
This is a heuristic digitizer, not a calibrated one. It cannot know the
real voltage scale or the real time scale of the original recording — it
only recovers the *shape* of the line. That's fine for this pipeline
because the downstream model only needs correct R-peak *timing* (via
neurokit2's peak detector), not real millivolt values. The one thing that
DOES need to be accurate is the total duration the image represents, in
seconds — you must supply that, because sampling_rate is derived from it
(sampling_rate = image_width_in_pixels / duration_seconds).

Best results when the image:
  - Is a clean digital plot (solid line), not a photo of paper.
  - Contains mostly just the waveform — heavy axis labels/legends/gridlines
    can get picked up as "trace" and distort the result.
  - Uses a trace color that contrasts with the background.
  - Spans enough real time to contain many heartbeats (the model needs at
    least 150 beats — roughly 2+ minutes of recording at a resting heart
    rate — to compute reliable HRV features).
"""

import io

import numpy as np
from PIL import Image


def image_to_ecg_signal(image_bytes, max_width=20000):
    """
    image_bytes : raw bytes of an uploaded image file (PNG/JPG/etc).
    max_width   : downscale absurdly wide images to this many pixel columns,
                  purely to bound processing time/memory. Only kicks in for
                  unusually large exports — a wide, high-resolution image is
                  actually desirable here, since sampling_rate is derived
                  from pixel-columns-per-second: too few columns for the
                  duration you specify means too low an implied sampling
                  rate to detect heartbeats reliably.

    Returns: 1D numpy array, one amplitude value per pixel column.
    Raises ValueError with a human-readable reason if no plausible trace
    could be found.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not read this file as an image: {e}")

    if img.width > max_width:
        new_h = int(img.height * (max_width / img.width))
        img = img.resize((max_width, new_h))

    arr = np.asarray(img).astype(np.float32)  # H x W x 3
    h, w, _ = arr.shape
    if w < 20 or h < 20:
        raise ValueError("Image is too small to contain a usable waveform.")

    # 1. Estimate the background color from the four corners (assumes a
    #    plain background, typical of exported plots/screenshots).
    corner_pixels = np.concatenate([
        arr[0:5, 0:5].reshape(-1, 3),
        arr[0:5, -5:].reshape(-1, 3),
        arr[-5:, 0:5].reshape(-1, 3),
        arr[-5:, -5:].reshape(-1, 3),
    ])
    bg_color = np.median(corner_pixels, axis=0)

    # 2. Score every pixel by how different it is from the background.
    #    This picks up the trace line (and, unfortunately, any axis
    #    lines/text/gridlines too — cropping the image to just the plot
    #    area before uploading gives noticeably better results).
    diff = np.linalg.norm(arr - bg_color, axis=2)  # H x W
    threshold = max(30.0, float(diff.max()) * 0.25)
    mask = diff > threshold

    if not mask.any():
        raise ValueError(
            "Could not detect any waveform trace in this image — it may be "
            "blank, too low-contrast, or the background couldn't be "
            "identified. Try a cleaner screenshot with a plain background."
        )

    # 3. For each column, take the average row of "trace-like" pixels as
    #    that column's signal value. Using the row range's midpoint (rather
    #    than a straight mean) is more robust when the line has thickness
    #    or anti-aliasing.
    signal = np.full(w, np.nan)
    for x in range(w):
        rows = np.where(mask[:, x])[0]
        if rows.size > 0:
            signal[x] = (rows.min() + rows.max()) / 2.0

    valid_cols = int(np.sum(~np.isnan(signal)))
    if valid_cols < w * 0.5:
        raise ValueError(
            f"Only {valid_cols}/{w} columns of the image had a detectable "
            f"trace — it's too sparse, noisy, or low-contrast to digitize "
            f"reliably. Try a cleaner, more zoomed-in screenshot of just "
            f"the waveform."
        )

    # 4. Fill small gaps (e.g. where the line was briefly near-vertical)
    #    with linear interpolation over the missing columns.
    idx = np.arange(w)
    good = ~np.isnan(signal)
    signal = np.interp(idx, idx[good], signal[good])

    # 5. Flip vertically (image row 0 is the TOP of the image, but a higher
    #    voltage should map to a higher signal value) and center on zero.
    #    Absolute scale is arbitrary and doesn't matter for R-peak timing.
    signal = -(signal - signal.mean())

    return signal