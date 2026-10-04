"""
main.py — "Flying Kiss 🦋" — blow a kiss at your webcam and a burst of
photorealistic HD butterflies erupts from your lips and flutters around
inside the live camera window.

Pipeline (per frame):
    1. capture from webcam at 1280x720, flip horizontally (mirror mode)
    2. downscale to 640x360 → MediaPipe FaceMesh + Hands  (fast detection)
    3. FlyingKissDetector state machine → kiss trigger?
    4. spawn a burst of 12–20 butterflies at the mouth centre
    5. update physics + alpha-composite butterflies (wing flap, depth blur,
       glow trail via addWeighted motion blur)
    6. HUD / debug overlays → cv2.imshow

Keys:  q quit · d debug landmarks · b test burst · s screenshot

Install:  pip install opencv-python mediapipe numpy pillow
Run:      python main.py           (see --help for camera/size options)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

import butterfly
from butterfly import ButterflySwarm
from detector import FaceHandDetector
from gesture import FlyingKissDetector

ROOT = os.path.dirname(os.path.abspath(__file__))

ACCENT = (32, 164, 255)      # warm gold (BGR)
INK = (235, 235, 235)        # near-white


# --------------------------------------------------------------------------- #
# Optional kiss sound (pygame if available, silent skip otherwise)
# --------------------------------------------------------------------------- #
class KissSound:
    """Tiny wrapper so the app works with or without pygame installed."""

    def __init__(self, path):
        self.snd = None
        if not os.path.exists(path):
            return
        try:
            import pygame                                   # optional dep
            pygame.mixer.init()
            self.snd = pygame.mixer.Sound(path)
        except Exception:
            self.snd = None                                 # stay silent

    def play(self):
        if self.snd is not None:
            try:
                self.snd.play()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Flying Kiss butterfly webcam demo")
    p.add_argument("--camera", type=int, default=-1,
                   help="webcam index; -1 (default) auto-detects the first "
                        "camera that actually delivers frames")
    p.add_argument("--width", type=int, default=1280, help="render width")
    p.add_argument("--height", type=int, default=720, help="render height")
    p.add_argument("--max-butterflies", type=int, default=70,
                   help="cap on simultaneously alive butterflies")
    p.add_argument("--no-mirror", action="store_true", help="disable mirror flip")
    p.add_argument("--debug", action="store_true",
                   help="also show numeric debug text (mouth openness, "
                        "fingertips). The yellow lip/hand tracking overlay "
                        "is always drawn regardless of this flag.")
    return p.parse_args()


def ensure_sprites():
    """Load sprites; on first run auto-download/generate them."""
    folder = os.path.join(ROOT, "assets", "butterflies")
    sprites = butterfly.load_sprites(folder)
    if sprites:
        return sprites
    print("[assets] no butterfly sprites found — fetching …")
    import fetch_assets
    fetch_assets.ensure_assets(ROOT)
    sprites = butterfly.load_sprites(folder)
    if not sprites:
        sys.exit("[assets] could not obtain butterfly sprites — see README")
    return sprites


def hud(frame, fps, n_active, state, flash_until, debug):
    """Translucent top bar with FPS, butterfly count, state and key hints."""
    h, w = frame.shape[:2]
    bar = frame[0:58, 0:w]
    # One SIMD addWeighted instead of two full-bar float32 conversions.
    tint = np.empty_like(bar)
    tint[:] = (22, 20, 18)
    cv2.addWeighted(bar, 0.45, tint, 0.55, 0, dst=bar)

    cv2.putText(frame, f"FPS {fps:5.1f}   butterflies {n_active}   state: {state}",
                (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1, cv2.LINE_AA)
    tip = ("NOW flick your hand away!" if state == "ARMED"
           else "hand to lips, then flick away - or pucker & blow!")
    cv2.putText(frame, tip, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, INK, 1, cv2.LINE_AA)
    if time.monotonic() < flash_until:
        cv2.putText(frame, "xoxo ~ kiss sent ~", (14, 88),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.05, (90, 140, 255), 2, cv2.LINE_AA)
    if debug:
        cv2.putText(frame, "DEBUG", (w - 92, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 255), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def open_camera(index, W, H):
    """
    Open a camera, verifying it actually delivers frames.

    `index` may be -1, in which case indices are probed in turn.  Relying on
    index 0 alone is fragile: on many Linux machines the first /dev/videoN
    node is a capture card or a dummy device, and the real webcam is index 1.
    """
    candidates = [index] if index >= 0 else [0, 1, 2, 3]
    tried = []
    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        tried.append(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)           # lowest latency
        # isOpened() can be true for a node that never produces frames.
        ok = False
        for _ in range(10):
            got, _frame = cap.read()
            if got:
                ok = True
                break
        if ok:
            if idx != candidates[0] or len(candidates) > 1:
                print(f"[camera] using webcam index {idx}")
            return cap
        cap.release()
    sys.exit(f"ERROR: no working webcam found (tried indices {tried}). "
             f"Pass --camera N to choose one.")


def main():
    args = parse_args()
    W, H = args.width, args.height
    det_w = 640                                   # detection resolution (fast)
    det_h = max(240, round(H * det_w / float(W)))

    # ---- webcam ------------------------------------------------------------
    cap = open_camera(args.camera, W, H)

    # ---- models, sprites, sound --------------------------------------------
    print("[init] loading MediaPipe FaceMesh + Hands …")
    detector = FaceHandDetector()
    gesture = FlyingKissDetector()
    sprites = ensure_sprites()
    print(f"[init] ready — {len(sprites)} butterfly sprites")
    swarm = ButterflySwarm(sprites, W, H, max_butterflies=args.max_butterflies)
    sound = KissSound(os.path.join(ROOT, "assets", "kiss.wav"))

    # half-res glow/trail canvas (spec: motion blur via addWeighted)
    glow = np.zeros((H // 2, W // 2, 3), np.uint8)

    win = "Flying Kiss \U0001F98B"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    debug = args.debug
    flash_until = 0.0
    fps_ema = 0.0
    t_prev = time.perf_counter()
    miss = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            miss += 1
            if miss > 90:
                sys.exit("ERROR: webcam stopped delivering frames")
            time.sleep(0.01)
            continue
        miss = 0

        if not args.no_mirror:
            frame = cv2.flip(frame, 1)            # mirror mode feels natural
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        raw_dt = time.perf_counter() - t_prev
        t_prev = time.perf_counter()
        step = min(1.0 / 10.0, max(1.0 / 240.0, raw_dt))      # clamp dt
        if raw_dt > 1e-6:
            fps_ema = fps_ema * 0.9 + (1.0 / raw_dt) * 0.1

        # ---- detection on the downscaled frame --------------------------
        small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        result = detector.process(small)
        aspect = W / float(H)

        fired, reason = gesture.update(result["face"], result["hands"], aspect)
        if fired and result["face"] is not None:
            mx, my = result["face"]["mouth_center"]
            swarm.spawn_burst(mx * W, my * H)
            print(f"[kiss] {reason} -> burst of butterflies!")
            sound.play()
            flash_until = time.monotonic() + 0.9

        # ---- butterflies + glow trail ------------------------------------
        glow = cv2.addWeighted(glow, 0.82, glow, 0.0, 0)      # fade trail
        swarm.update_and_render(frame, step, glow=glow)
        if int(glow.max()) > 2:
            glow_full = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(frame, 1.0, glow_full, 0.35, 0, dst=frame)

        # ---- overlays ------------------------------------------------------
        # Lip mesh + hand skeletons (yellow lines, red dots) are always on.
        detector.draw_landmarks(frame, result)
        if debug:
            detector.draw_debug(frame, result)
        hud(frame, fps_ema, len(swarm.items), gesture.state, flash_until, debug)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            debug = not debug
        elif key == ord("b"):                                  # test burst
            swarm.spawn_burst(W * 0.5, H * 0.55)
        elif key == ord("s"):                                  # screenshot
            out = os.path.join(ROOT, "screenshot.png")
            cv2.imwrite(out, frame)
            print(f"[shot] saved {out}")

    print("[exit] releasing camera …")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[exit] interrupted — bye!")
