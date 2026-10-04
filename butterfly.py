"""
butterfly.py — Butterfly particle: sprite loading, physics, and rendering.

A butterfly is a lightweight particle carrying one high-res RGBA sprite
(loaded once at startup). Each frame it advances through a small physics
model:

    vx *= 0.97 ; vy *= 0.97        # air drag
    vy -= lift                     # gentle upward flutter lift
    + organic pseudo-Perlin noise  # sum of incommensurate sines
    + wing-beat bob                # body lifts on each down-stroke

and is alpha-composited onto the webcam frame. Wing flapping is simulated
by horizontally squeezing the sprite between 0.30x and 1.00x width
(wing_phase driven), depth is faked by scaling "farther" butterflies down
and pre-blurring them (depth-of-field), and sprites are resampled with
INTER_LANCZOS4 to keep the edges crisp and HD.
"""

from __future__ import annotations

import glob
import math
import os
import random

import cv2
import numpy as np

SPRITE_TARGET = 512          # longest sprite edge at load time (HD crispness)

# --------------------------------------------------------------------------- #
# Look & feel tunables
# --------------------------------------------------------------------------- #
# Butterfly height as a fraction of the frame height.  Previously the size was
# derived from SPRITE_TARGET, which produced sprites up to 716 px tall inside a
# 720 px window -- individual butterflies were taller than the whole frame.
SIZE_FRAC_MIN = 0.085        # smallest butterfly, as a fraction of frame height
SIZE_FRAC_MAX = 0.22         # largest butterfly

# Wing flap.  The rate is deliberately modest: at 30 fps a 9 Hz flap only gets
# ~3 frames per cycle and reads as a flickering sliver rather than a flap.
FLAP_HZ_MIN = 2.6
FLAP_HZ_MAX = 5.2
FLAP_SQUEEZE_MIN = 0.45      # narrowest wing spread (fraction of full width)

# Per-frame interpolation for the wing squeeze + rotation.  INTER_LANCZOS4 is
# ~64 taps per pixel and was being run on every butterfly every frame, which
# cost >100 ms each and froze the app.  The squeeze is a smooth horizontal
# scale, so bilinear is visually indistinguishable and ~50x cheaper.
FLAP_INTERP = cv2.INTER_LINEAR

# Rotation drifts slowly, so the (expensive) arbitrary-angle warp is only
# redone once the tilt has moved this far.  Between warps the frame only pays
# for a cheap horizontal resize.
ROT_REWARP_DEG = 6.0

# Radial falloff blobs for the glow trail, keyed by diameter.  Rebuilding the
# np.mgrid for every butterfly on every frame cost ~0.9 ms each.
_BLOB_CACHE = {}
_BLOB_COLOR_CACHE = {}


def _glow_blob(size):
    """Cached soft radial falloff of the given diameter (uint8, 0..255)."""
    blob = _BLOB_CACHE.get(size)
    if blob is None:
        r = size * 0.5
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        d = np.sqrt((xx - r) ** 2 + (yy - r) ** 2) / max(1.0, float(r))
        falloff = np.clip(1.0 - d, 0.0, 1.0) ** 2          # soft radial falloff
        blob = (falloff * 255.0).astype(np.uint8)
        _BLOB_CACHE[size] = blob
    return blob


def _glow_patch(size, level):
    """Coloured (blue-ish) glow blob, cached by (size, brightness level).

    `level` quantises the fade envelope to 16 steps so the per-frame path is a
    single saturating add instead of float math plus an np.clip.
    """
    key = (size, level)
    patch = _BLOB_COLOR_CACHE.get(key)
    if patch is None:
        gain = _glow_blob(size).astype(np.float32) * (float(level) / 16.0)
        patch = gain[:, :, None] * np.array([70.0, 120.0, 190.0], np.float32) / 255.0
        patch = np.clip(patch, 0, 255).astype(np.uint8)
        _BLOB_COLOR_CACHE[key] = patch
    return patch


# --------------------------------------------------------------------------- #
# Sprite loading
# --------------------------------------------------------------------------- #
def load_sprites(folder, target=SPRITE_TARGET):
    """Load every butterfly PNG (with alpha) from *folder*.

    Returns a list of BGRA numpy arrays, trimmed to their alpha bounding box
    and rescaled so the longest edge is *target* px. Sprites are loaded ONCE
    at startup; individual butterflies then just reference them.
    """
    sprites = []
    for path in sorted(glob.glob(os.path.join(folder, "*.png"))):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[sprites] warning: unreadable PNG {path}")
            continue
        if img.ndim == 2:                                    # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        if img.shape[2] == 3:                                # BGR → BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        alpha = img[:, :, 3]
        ys, xs = np.where(alpha > 8)                         # trim transparent border
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        img = img[y0:y1, x0:x1]
        h, w = img.shape[:2]
        scale = target / float(max(h, w))
        if abs(scale - 1.0) > 0.01:
            img = cv2.resize(img, (max(2, int(w * scale)), max(2, int(h * scale))),
                             interpolation=cv2.INTER_LANCZOS4)
        sprites.append(img)
        print(f"[sprites] loaded {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")
    return sprites


# --------------------------------------------------------------------------- #
# Butterfly particle
# --------------------------------------------------------------------------- #
class Butterfly:
    """One animated butterfly particle."""

    def __init__(self, sprite, x, y, frame_w, frame_h):
        self.sprite = sprite
        self.fw, self.fh = int(frame_w), int(frame_h)

        # ---- spawn state: radial burst outward from the mouth -------------
        self.x = x + random.uniform(-26, 26)
        self.y = y + random.uniform(-22, 22)
        ang = random.uniform(0.0, 2.0 * math.pi)
        spd = random.uniform(3.5, 10.5)                      # px/frame @30 fps
        self.vx = math.cos(ang) * spd
        self.vy = math.sin(ang) * spd - 1.2                  # slight upward bias
        self.drag = random.uniform(0.955, 0.975)             # per-frame damping

        # ---- look & feel ---------------------------------------------------
        self.depth = random.uniform(0.55, 1.0)               # <1 = farther away
        # Size is a fraction of the FRAME height, not of SPRITE_TARGET, so the
        # butterflies stay butterflies regardless of sprite resolution.
        frac = random.uniform(SIZE_FRAC_MIN, SIZE_FRAC_MAX) * self.depth
        base_h = max(24, int(self.fh * frac))
        h0, w0 = sprite.shape[:2]
        base_w = max(10, int(w0 * base_h / float(h0)))
        # Pre-scale ONCE at spawn (HD LANCZOS4), pre-blur far butterflies.
        base = cv2.resize(sprite, (base_w, base_h), interpolation=cv2.INTER_LANCZOS4)
        if self.depth < 0.72:                                # depth-of-field blur
            k = 5 if self.depth < 0.62 else 3
            base = cv2.GaussianBlur(base, (k, k), 0)
        # Pad so rotation never clips the wingtips.
        pad = max(6, int(base_h * 0.30))
        canvas = np.zeros((base_h + 2 * pad, base_w + 2 * pad, 4), dtype=np.uint8)
        canvas[pad:pad + base_h, pad:pad + base_w] = base
        self.base = canvas
        self.bw, self.bh = canvas.shape[1], canvas.shape[0]

        self.rotation = random.uniform(-24.0, 24.0)          # deg
        self.rot_speed = random.uniform(-30.0, 30.0)         # deg / s
        self.wing_phase = random.uniform(0.0, 2.0 * math.pi)
        self.flap_hz = random.uniform(FLAP_HZ_MIN, FLAP_HZ_MAX)

        # Cached tilted sprite, rebuilt only when the tilt has drifted enough.
        self._tilted = None
        self._tilted_deg = None

        # Reusable blend scratch buffers, so the alpha composite does not
        # allocate two full-size temporaries per butterfly per frame.
        self._blend_a = np.empty((self.bh, self.bw, 3), dtype=np.uint8)

        # ---- life & organic motion ----------------------------------------
        self.age = 0.0
        self.lifetime = random.uniform(4.0, 6.0)             # seconds
        f1 = random.uniform(0.8, 1.6)                        # noise frequencies
        f2 = random.uniform(1.7, 2.9)
        f3 = random.uniform(0.8, 1.6)
        f4 = random.uniform(1.7, 2.9)
        t0 = random.uniform(0.0, 100.0)                      # noise seed
        self._n = (f1, f2, f3, f4, t0)

    # ------------------------------------------------------------------ #
    @property
    def alive(self):
        return self.age < self.lifetime

    def _envelope(self):
        """Fade-in on spawn, fade-out near end of life → 0..1."""
        fade_in = min(1.0, self.age / 0.25)
        fade_out = min(1.0, max(0.0, (self.lifetime - self.age) / 0.9))
        return fade_in * fade_out

    def update(self, step):
        """Advance physics. *step* = seconds since the previous frame."""
        k = step * 30.0                                      # normalise to 30 fps
        drag = self.drag ** k
        self.vx *= drag
        self.vy *= drag
        self.vy -= 0.012 * k                                 # gentle flutter lift

        # Organic pseudo-Perlin flutter (incommensurate sine mixture).
        f1, f2, f3, f4, t0 = self._n
        t = self.age + t0
        self.vx += (math.sin(t * f1 * 3.1) + 0.5 * math.sin(t * f2 * 3.7)) * 0.075 * k
        self.vy += (math.sin(t * f3 * 2.7) + 0.5 * math.cos(t * f4 * 3.3)) * 0.060 * k

        # Wing-beat bob: body lifts a touch on each down-stroke.
        self.vy -= math.sin(self.wing_phase) * 0.010 * k

        self.x += self.vx * k
        self.y += self.vy * k

        # Soft edge steering — butterflies stay INSIDE the window and are
        # gently pushed back instead of hard-bouncing off the edges.
        m = 70.0
        if self.x < m:
            self.vx += 0.28 * k
        elif self.x > self.fw - m:
            self.vx -= 0.28 * k
        if self.y < m:
            self.vy += 0.28 * k
        elif self.y > self.fh - m:
            self.vy -= 0.28 * k
        self.x = min(max(self.x, 12.0), float(self.fw - 12))
        self.y = min(max(self.y, 12.0), float(self.fh - 12))

        self.wing_phase = (self.wing_phase + self.flap_hz * 2.0 * math.pi * step) % (2.0 * math.pi)
        self.rotation += self.rot_speed * step
        self.age += step

    # ------------------------------------------------------------------ #
    def _tilted_sprite(self):
        """Base sprite rotated to the current tilt, rebuilt only when needed.

        Rotating is an arbitrary-angle warp (~4x the cost of a plain resize),
        and the tilt only creeps a degree or so per frame, so caching it and
        refreshing every ROT_REWARP_DEG keeps the cost off the hot path.
        """
        if (self._tilted is None
                or self._tilted_deg is None
                or abs(self.rotation - self._tilted_deg) >= ROT_REWARP_DEG):
            M = cv2.getRotationMatrix2D(
                (self.bw * 0.5, self.bh * 0.5), self.rotation, 1.0)
            self._tilted = cv2.warpAffine(
                self.base, M, (self.bw, self.bh),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            self._tilted_deg = self.rotation
        return self._tilted

    def render(self, frame):
        """Alpha-composite this butterfly onto *frame* (BGR uint8, in place)."""
        env = self._envelope()
        if env <= 0.01:
            return

        # Wing flap = horizontal squeeze of the already-tilted sprite.
        flap = FLAP_SQUEEZE_MIN + (1.0 - FLAP_SQUEEZE_MIN) * abs(
            math.sin(self.wing_phase))
        tw = max(6, int(self.bw * flap))
        sprite = cv2.resize(self._tilted_sprite(), (tw, self.bh),
                            interpolation=FLAP_INTERP)

        # Split BGRA into contiguous BGR + a contiguous ALPHA plane.
        # NOTE: do NOT use cv2.COLOR_BGRA2GRAY here -- it converts the *colour*
        # channels to luminance and ignores alpha completely.  These PNGs keep
        # near-white RGB under their transparent regions, so that conversion
        # reported alpha=255 on fully transparent pixels and drew every sprite
        # as an opaque white box.
        bgr = cv2.cvtColor(sprite, cv2.COLOR_BGRA2BGR)
        a8 = np.ascontiguousarray(sprite[:, :, 3])
        if env < 0.999:
            a8 = cv2.convertScaleAbs(a8, alpha=float(env))

        h, w = a8.shape
        x0 = int(self.x - w / 2.0)
        y0 = int(self.y - h / 2.0)

        # Clip the ROI against the frame edges.
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(self.fw, x0 + w), min(self.fh, y0 + h)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        sx0, sy0 = fx0 - x0, fy0 - y0
        sx1, sy1 = sx0 + (fx1 - fx0), sy0 + (fy1 - fy0)

        src = bgr[sy0:sy1, sx0:sx1]
        m3 = cv2.cvtColor(a8[sy0:sy1, sx0:sx1], cv2.COLOR_GRAY2BGR)
        i3 = cv2.bitwise_not(m3)
        roi = frame[fy0:fy1, fx0:fx1]
        hh, ww = src.shape[:2]
        buf_a = self._blend_a[:hh, :ww]
        cv2.multiply(src, m3, dst=buf_a, scale=1.0 / 255.0)
        cv2.multiply(roi, i3, dst=roi, scale=1.0 / 255.0)
        cv2.add(buf_a, roi, dst=roi)

    # ------------------------------------------------------------------ #
    def stamp_glow(self, glow):
        """Stamp a soft additive blob on the low-res glow/trail canvas."""
        env = self._envelope()
        if env <= 0.05:
            return
        gw, gh = glow.shape[1], glow.shape[0]
        sx = gw / float(self.fw)
        sy = gh / float(self.fh)
        cx, cy = self.x * sx, self.y * sy
        size = 2 * max(4, int(self.bw * 0.20 * sx))
        level = max(1, min(16, int(env * 16.0)))
        patch = _glow_patch(size, level)

        x0, y0 = int(cx - size * 0.5), int(cy - size * 0.5)
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(gw, x0 + size), min(gh, y0 + size)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        px0, py0 = fx0 - x0, fy0 - y0
        px1, py1 = px0 + (fx1 - fx0), py0 + (fy1 - fy0)
        roi = glow[fy0:fy1, fx0:fx1]
        cv2.add(roi, patch[py0:py1, px0:px1], dst=roi)   # saturating uint8 add


# --------------------------------------------------------------------------- #
# Swarm container
# --------------------------------------------------------------------------- #
class ButterflySwarm:
    """Spawns bursts and updates/renders every active butterfly."""

    def __init__(self, sprites, frame_w, frame_h, max_butterflies=80):
        if not sprites:
            raise ValueError("No butterfly sprites loaded — run: python fetch_assets.py")
        self.sprites = sprites
        self.fw, self.fh = int(frame_w), int(frame_h)
        self.max = max_butterflies
        self.items = []
        self.total_spawned = 0

    def spawn_burst(self, x, y, count=None):
        """Spawn a burst of 12–20 butterflies at pixel position (x, y)."""
        count = count if count is not None else random.randint(12, 20)
        for _ in range(count):
            if len(self.items) >= self.max:
                self.items.pop(0)                            # recycle the oldest
            sprite = random.choice(self.sprites)
            self.items.append(Butterfly(sprite, x, y, self.fw, self.fh))
        self.total_spawned += count

    def update_and_render(self, frame, step, glow=None):
        """Update physics + composite all butterflies (far ones painted first)."""
        for b in self.items:
            b.update(step)
        self.items = [b for b in self.items if b.alive]
        for b in sorted(self.items, key=lambda b: b.depth):
            b.render(frame)
            if glow is not None:
                b.stamp_glow(glow)
