"""
fetch_assets.py — populate assets/ so the demo always runs.

Strategy (in order):
1. If assets/butterflies/ already holds >= MIN_SPRITES PNGs → done.
2. Download public-domain butterfly specimen photos from Wikimedia Commons
   (Special:FilePath, resized server-side). The "MHNT" (Muséum de Toulouse)
   series are dorsal specimens on a clean white background, so the script
   auto-cutouts the background with a border flood-fill + largest-component
   cleanup, trims to the alpha bounding box and saves 512-px transparent
   PNGs.
3. If downloads fail (offline etc.), procedurally generate stylized
   (non-emoji) butterflies with PIL so the demo still runs.
4. Also synthesize a soft "kiss pop" sound → assets/kiss.wav (used only if
   pygame is installed).

Run standalone:  python fetch_assets.py
"""

from __future__ import annotations

import io
import math
import os
import random
import sys
import time
import wave

import cv2
import numpy as np
import requests

MIN_SPRITES = 5
TARGET = 512                       # longest sprite edge (HD)
HEADERS = {"User-Agent": "FlyingKissButterflies/1.0 (educational demo; python-requests)"}

# Public-domain dorsal specimen photos (white background) — species per spec:
# Monarch, Blue Morpho, Swallowtails + a few extras in case some 404.
WIKIMEDIA_TITLES = [
    ("Danaus plexippus MHNT.jpg", "monarch"),
    ("Morpho peleides MHNT.jpg", "morpho"),
    ("Papilio machaon MHNT.jpg", "swallowtail"),
    ("Iphiclides podalirius MHNT.jpg", "swallowtail2"),
    ("Aglais io MHNT.jpg", "peacock"),
    ("Vanessa atalanta MHNT.jpg", "admiral"),
    ("Junonia coenia MHNT.jpg", "buckeye"),
    ("Polygonia c-album MHNT.jpg", "comma"),
    ("Danaus gilippus MHNT.jpg", "queen"),
    ("Morpho didius MHNT.jpg", "morpho2"),
    ("Papilio cresphontes MHNT.jpg", "giant-swallowtail"),
    ("Battus philenor MHNT.jpg", "pipevine"),
]


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def http_get(url, timeout=30, retries=3):
    """GET with polite retries on 429/5xx (exponential backoff)."""
    delay = 2.0
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2.5
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"    ! {e}")
                return None
            time.sleep(delay)
            delay *= 2.5
    return None


def wikimedia_url(title, width=1400):
    return ("https://commons.wikimedia.org/wiki/Special:FilePath/"
            + requests.utils.quote(title) + f"?width={width}")


# --------------------------------------------------------------------------- #
# Image processing
# --------------------------------------------------------------------------- #
def load_bgra(data):
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
    if arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    return arr


def has_real_alpha(arr):
    """True when the image already carries meaningful transparency."""
    a = arr[:, :, 3]
    return float((a < 200).mean()) > 0.10


def _remove_soft_shadow(bgr, bg, iters=64, v_min=178, s_max=70):
    """Grow the background mask into desaturated light pixels (drop shadows).

    Shadow on a white studio backdrop is nearly achromatic (s ≈ 0-25) even
    where it gets darker (v ≈ 150-230). Butterflies are either saturated
    (s ≳ 90) or very dark (v < 120), so growing the background into
    achromatic/bright pixels eats the whole shadow and spares the wings.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    cur = (bg.astype(np.uint8)) * 255
    kern = np.ones((3, 3), np.uint8)
    for _ in range(iters):
        grown = cv2.dilate(cur, kern)
        cand = (grown > 0) & (((v > v_min) & (s < s_max))   # bright haze
                              | ((s < 25) & (v > 140))      # achromatic shadow
                              | (v > 235))                  # blown white
        new = cur.copy()
        new[cand] = 255
        if int(new.sum()) == int(cur.sum()):
            break
        cur = new
    return cur > 0


def cutout_white(bgra, tol=30, kill_shadow=True):
    """Remove a clean white/light background via border flood-fill.

    Returns None when the background isn't a clean white field (so caller
    can skip the image instead of shipping a bad cutout).
    """
    h, w = bgra.shape[:2]
    bgr = bgra[:, :, :3].copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
             (w // 2, 0), (0, h // 2), (w - 1, h // 2), (w // 2, h - 1)]
    for sx, sy in seeds:
        px = bgr[sy, sx]
        if min(int(px[0]), int(px[1]), int(px[2])) > 190:      # near-white seed
            cv2.floodFill(bgr, mask, (sx, sy), 0,
                          (tol, tol, tol), (tol, tol, tol), flags)
    bg = mask[1:-1, 1:-1] > 0
    cov = float(bg.mean())
    if cov < 0.10 or cov > 0.94:            # no clean white background found
        return None
    if kill_shadow:
        bg = _remove_soft_shadow(bgr, bg)   # eat soft studio drop-shadows
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    # Erode 2px to kill white background fringing around wing edges,
    # then despeckle + feather for a clean composite.
    alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=2)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    # Keep only the largest opaque component (drops scale bars, labels, dust).
    solid = (alpha > 128).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = (labels == biggest)
        alpha = np.where(keep, alpha, 0).astype(np.uint8)

    out = bgra.copy()
    out[:, :, 3] = np.minimum(out[:, :, 3], alpha)
    return out


def trim(arr, alpha_thresh=8):
    """Crop to the alpha bounding box."""
    ys, xs = np.where(arr[:, :, 3] > alpha_thresh)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return arr[y0:y1, x0:x1]


def finalize(arr, name, dest):
    """Trim + resize to TARGET + save."""
    arr = trim(arr)
    if arr is None:
        return False
    h, w = arr.shape[:2]
    scale = TARGET / float(max(h, w))
    if abs(scale - 1.0) > 0.01:
        arr = cv2.resize(arr, (max(2, int(w * scale)), max(2, int(h * scale))),
                         interpolation=cv2.INTER_LANCZOS4)
    path = os.path.join(dest, name)
    cv2.imwrite(path, arr)
    print(f"    saved {name} ({arr.shape[1]}x{arr.shape[0]})")
    return True


# --------------------------------------------------------------------------- #
# Download pipeline
# --------------------------------------------------------------------------- #
def download_from_wikimedia(dest, wanted):
    print("[assets] downloading public-domain butterfly specimens from Wikimedia …")
    got = 0
    for title, slug in WIKIMEDIA_TITLES:
        if got >= wanted:
            break
        print(f"  • {title}")
        data = http_get(wikimedia_url(title))
        if data is None:
            continue
        arr = load_bgra(data)
        if arr is None or arr.shape[0] < 220 or arr.shape[1] < 220:
            print("    skipped (too small / unreadable)")
            continue
        if not has_real_alpha(arr):                     # photo on white bg
            arr = cutout_white(arr)
            if arr is None:
                print("    skipped (background not clean white)")
                continue
        if finalize(arr, f"butterfly_{got + 1:02d}_{slug}.png", dest):
            got += 1
            time.sleep(1.2)                             # be polite to the API
    return got


# --------------------------------------------------------------------------- #
# Procedural fallback (PIL) — stylized-real butterflies, NOT emoji
# --------------------------------------------------------------------------- #
SCHEMES = [
    ("monarch",    (232, 138, 38),  (255, 210, 125), (30, 17, 10),  True),
    ("morpho",     (25, 95, 210),   (135, 215, 255), (12, 20, 44),  False),
    ("swallowtail", (246, 210, 86), (252, 236, 165), (32, 32, 28),  True),
    ("peacock",    (188, 62, 58),   (242, 152, 82),  (26, 18, 20),  True),
    ("admiral",    (70, 78, 190),   (130, 145, 240), (22, 22, 30),  True),
    ("buckeye",    (165, 112, 62),  (224, 174, 112), (36, 28, 22),  True),
]


def _wing_half(scheme, S):
    """Build the LEFT wing pair as an RGBA canvas (numpy, RGB order)."""
    _, wing, light, edge, dots = scheme
    root = np.array([S * 0.50, S * 0.55])               # wing root (x, y)
    mask = np.zeros((S, S), np.uint8)
    # upper fore-wing + lower hind-wing (left side)
    cv2.ellipse(mask, (int(S * 0.30), int(S * 0.28)), (int(S * 0.235), int(S * 0.165)),
                38, 0, 360, 255, -1)
    cv2.ellipse(mask, (int(S * 0.34), int(S * 0.66)), (int(S * 0.19), int(S * 0.135)),
                -24, 0, 360, 255, -1)

    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    d = np.sqrt((xx - root[0]) ** 2 + (yy - root[1]) ** 2) / (S * 0.62)
    t = np.clip(d, 0, 1) ** 0.8
    grad = np.zeros((S, S, 3), np.float32)
    for c in range(3):
        grad[:, :, c] = (light[c] * (1 - t) + wing[c] * t)

    # dark margin band
    band = cv2.subtract(mask, cv2.erode(mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (23, 23))))
    for c in range(3):
        grad[:, :, c][band > 0] = edge[c]

    # white spots along the margin (species look)
    if dots:
        rng = random.Random(7)
        for _ in range(60):
            x, y = rng.randrange(4, S - 4), rng.randrange(4, S - 4)
            if band[y, x] > 0 and rng.random() < 0.30:
                r = rng.randint(3, 6)
                cv2.circle(grad, (x, y), r, (245, 245, 240), -1)

    # veins fanning from the root
    veins = np.zeros((S, S), np.uint8)
    for ang in (-72, -50, -30, -8, 12, 32, 50):
        a = math.radians(ang)
        x2 = int(root[0] + math.cos(a) * S * 0.55)
        y2 = int(root[1] + math.sin(a) * S * 0.55)
        cv2.line(veins, (int(root[0]), int(root[1])), (x2, y2), 255, 2)
    for c in range(3):
        grad[:, :, c][veins > 0] = (grad[:, :, c][veins > 0] * 0.45 + edge[c] * 0.55)

    rgba = np.dstack([grad.astype(np.uint8), mask])
    rgba[:, int(S * 0.52):, 3] = 0                      # keep left half only
    return rgba


def procedural_butterfly(scheme, seed=0):
    """Compose a full butterfly (512x512 RGBA) from mirrored wings + body."""
    from PIL import Image, ImageDraw
    S = 512
    left = _wing_half(scheme, S)
    right = np.fliplr(left)
    full = np.maximum(left, right)                      # union of both halves

    img = Image.fromarray(full)
    d = ImageDraw.Draw(img)
    body = (72, 48, 34, 255)
    d.ellipse([S * 0.493, S * 0.36, S * 0.507 + 8, S * 0.78], fill=body)   # abdomen
    d.ellipse([S * 0.488, S * 0.33, S * 0.514, S * 0.40], fill=body)       # thorax
    d.ellipse([S * 0.484, S * 0.285, S * 0.516, S * 0.325], fill=body)     # head
    # antennae with club tips
    for sgn in (-1, 1):
        pts = []
        for i in range(13):
            t = i / 12.0
            x = S * 0.50 + sgn * (t * S * 0.115)
            y = S * 0.30 - math.sin(t * math.pi * 0.5) * S * 0.115
            pts.append((x, y))
        d.line(pts, fill=body, width=3)
        d.ellipse([pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4], fill=body)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def generate_procedural(dest, needed, start=0):
    print(f"[assets] generating {needed} procedural butterflies (offline fallback) …")
    for i in range(needed):
        scheme = SCHEMES[(start + i) % len(SCHEMES)]
        arr = procedural_butterfly(scheme, seed=i)
        finalize(arr, f"butterfly_{start + i + 1:02d}_{scheme[0]}.png", dest)


# --------------------------------------------------------------------------- #
# Kiss sound (tiny synthesized "pop") — used only when pygame is present
# --------------------------------------------------------------------------- #
def make_kiss_sound(path, sr=22050):
    t = np.linspace(0.0, 0.28, int(sr * 0.28), False)
    f = 900.0 * np.exp(-t * 9.0) + 220.0                       # downward chirp
    x = np.sin(2 * np.pi * f * t) * np.exp(-t * 11.0)
    x += 0.22 * np.random.randn(len(t)) * np.exp(-t * 30.0)    # breathy noise
    x = np.tanh(x * 2.2) * 0.7
    pcm = (x * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    print(f"[assets] synthesized {path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def ensure_assets(base_dir=None):
    """Idempotent: fill assets/butterflies + assets/kiss.wav if missing."""
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(base_dir, "assets", "butterflies")
    os.makedirs(dest, exist_ok=True)

    existing = [f for f in os.listdir(dest) if f.lower().endswith(".png")]
    if len(existing) >= MIN_SPRITES:
        print(f"[assets] {len(existing)} sprites already present — nothing to do")
    else:
        got = download_from_wikimedia(dest, wanted=8)
        if got < MIN_SPRITES:
            generate_procedural(dest, needed=MIN_SPRITES - got, start=got)

    wav = os.path.join(base_dir, "assets", "kiss.wav")
    if not os.path.exists(wav):
        try:
            make_kiss_sound(wav)
        except Exception as e:
            print(f"[assets] could not synthesize kiss sound: {e}")

    n = len([f for f in os.listdir(dest) if f.lower().endswith(".png")])
    print(f"[assets] done — {n} butterfly sprites in {dest}")
    return n


if __name__ == "__main__":
    ensure_assets(sys.argv[1] if len(sys.argv) > 1 else None)
