"""
gesture.py — Flying-kiss state machine.

Two independent triggers, whichever fires first wins:

1. HAND-FLICK
   Index/middle fingertip stays within `near_thresh` (0.15, normalized
   height-units) of the mouth centre with the palm facing the camera for
   `near_frames` (5) consecutive frames, THEN the hand moves > `away_thresh`
   (0.12) away within `release_window` (0.5 s).

2. BLOW  ("pucker & blow")
   Mouth puckered (openness < `pucker_thresh` = 0.05) for >= 2 frames, then
   opens wide (openness > `blow_thresh` = 0.25) within 0.6 s.

A `cooldown` (0.8 s) after each fire suppresses double triggers.

update(face, hands, aspect) -> (triggered: bool, reason: str | None)
"""

import math
import time

NEAR_FRAMES = 5        # consecutive near-mouth frames to charge
NEAR_THRESH = 0.15     # fingertip↔mouth distance that counts as "near"
AWAY_THRESH = 0.12     # how far the hand must move away to release
RELEASE_WINDOW = 0.5   # seconds allowed for the flick-away
PUCKER_THRESH = 0.05   # openness below this = puckered lips
BLOW_THRESH = 0.25     # openness above this = blown wide open
BLOW_WINDOW = 0.6      # seconds between pucker and wide-open
COOLDOWN = 0.8         # seconds of silence after each trigger


def aspect_dist(a, b, aspect):
    """Distance between two normalized points, x corrected by aspect (w/h).

    Working in 'height units' makes thresholds behave the same on 16:9,
    4:3 and vertical cameras.
    """
    return math.hypot((float(a[0]) - float(b[0])) * aspect,
                      float(a[1]) - float(b[1]))


class FlyingKissDetector:
    """StateMachine: IDLE → ARMED (hand charged at mouth) → fire | timeout."""

    def __init__(self,
                 near_frames=NEAR_FRAMES,
                 near_thresh=NEAR_THRESH,
                 away_thresh=AWAY_THRESH,
                 release_window=RELEASE_WINDOW,
                 pucker_thresh=PUCKER_THRESH,
                 blow_thresh=BLOW_THRESH,
                 blow_window=BLOW_WINDOW,
                 cooldown=COOLDOWN):
        self.near_frames = near_frames
        self.near_thresh = near_thresh
        self.away_thresh = away_thresh
        self.release_window = release_window
        self.pucker_thresh = pucker_thresh
        self.blow_thresh = blow_thresh
        self.blow_window = blow_window
        self.cooldown = cooldown

        self.state = "IDLE"          # IDLE | ARMED
        self.near_count = 0          # consecutive near-mouth frames
        self.armed_pos = None        # fingertip position when charged
        self.armed_time = 0.0
        self.last_tip = None         # last seen fingertip position
        self.last_seen = -1e9
        self.pucker_frames = 0
        self.last_pucker = -1e9
        self.last_trigger = -1e9
        self.reason = None

    # ------------------------------------------------------------------ #
    def update(self, face, hands, aspect, now=None):
        """Feed one frame of detections; returns (triggered, reason)."""
        if now is None:
            now = time.monotonic()
        trigger = False
        self.reason = None
        mouth = face["mouth_center"] if face else None

        # ---------------- hand path -------------------------------------
        tip, tip_d, palm_ok = None, 1e9, False
        if mouth is not None:
            for hnd in hands:
                for cand in (hnd["tip_index"], hnd["tip_middle"]):
                    d = aspect_dist(cand, mouth, aspect)
                    if d < tip_d:
                        tip_d, tip, palm_ok = d, cand, hnd["palm_facing"]

        hand_near = tip is not None and palm_ok and tip_d < self.near_thresh
        if hand_near:
            self.near_count += 1
            self.last_tip, self.last_seen = tip, now
            if self.state == "IDLE" and self.near_count >= self.near_frames:
                self.state = "ARMED"                       # charged!
                self.armed_pos, self.armed_time = tip, now
            elif self.state == "ARMED":
                if (aspect_dist(tip, self.armed_pos, aspect) > self.away_thresh
                        and now - self.armed_time <= self.release_window):
                    trigger, self.reason = True, "hand-flick"
                elif now - self.armed_time > self.release_window:
                    # Still hovering by the mouth — re-arm so a later quick
                    # flick can fire without redoing the 5-frame charge.
                    self.armed_pos, self.armed_time = tip, now
        else:
            self.near_count = max(0, self.near_count - 2)  # decay, not reset
            if self.state == "ARMED":
                if tip is not None:
                    # Hand visible again but out of the mouth zone & moved
                    # away fast enough → release the kiss.
                    if (now - self.armed_time <= self.release_window
                            and tip_d > self.near_thresh
                            and aspect_dist(tip, self.armed_pos, aspect) > self.away_thresh):
                        trigger, self.reason = True, "hand-flick"
                elif now - self.armed_time <= self.release_window:
                    # Hand vanished right after charging (fast flick out of
                    # frame / motion blur) → treat as a release too.
                    trigger, self.reason = True, "hand-flick"
                elif now - self.armed_time > self.release_window + 0.15:
                    self.state = "IDLE"

        # ---------------- mouth "pucker & blow" path ---------------------
        if face is not None:
            op = face["openness"]
            if op < self.pucker_thresh:
                self.pucker_frames += 1
                self.last_pucker = now
            else:
                if (self.pucker_frames >= 2
                        and op > self.blow_thresh
                        and now - self.last_pucker <= self.blow_window):
                    trigger, self.reason = True, "blow"
                if op > self.blow_thresh:
                    self.pucker_frames = 0      # fully opened → reset cycle

        # ---------------- cooldown + commit -------------------------------
        if trigger and (now - self.last_trigger) < self.cooldown:
            trigger, self.reason = False, None
        if trigger:
            self.last_trigger = now
            self.state = "IDLE"
            self.near_count = 0
            self.pucker_frames = 0
        return trigger, self.reason
