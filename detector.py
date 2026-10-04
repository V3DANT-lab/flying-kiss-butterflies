"""
detector.py — MediaPipe FaceMesh + Hands wrapper.

FaceHandDetector runs MediaPipe FaceMesh (468 landmarks) and Hands on one
BGR frame and returns plain numpy data:

    face:  None | {
        "landmarks":    (468, 3) float32 normalized x,y,z,
        "mouth_center": (2,)  midpoint of upper lip (13) & lower lip (14),
        "openness":     dist(13,14) / dist(61,291)  — 0 = closed/puckered,
    }
    hands: [ {
        "landmarks":   (21, 3) float32 normalized,
        "tip_index":   (2,) index fingertip (8),
        "tip_middle":  (2,) middle fingertip (12),
        "palm_center": (2,) mean of wrist + knuckles,
        "palm_facing": bool  — knuckles closer to camera than fingertips,
        "handedness":  "Left"/"Right"/"?",
    }, ... ]

All coordinates are normalized to 0..1. The raw MediaPipe protobufs are
also returned for the optional debug overlay.
"""

import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError as err:  # friendly failure
    raise SystemExit(
        "MediaPipe is required but not installed.\n"
        "    pip install mediapipe\n"
        f"(original error: {err})"
    )

# MediaPipe FaceMesh landmark indices
LIP_UPPER, LIP_LOWER = 13, 14          # inner upper / lower lip
MOUTH_LEFT, MOUTH_RIGHT = 61, 291      # mouth corners
# MediaPipe Hands landmark indices
TIP_INDEX, TIP_MIDDLE = 8, 12          # index & middle fingertips
KNUCKLE_IDS = (5, 9, 13, 17)           # MCP joints
TIP_IDS = (8, 12, 16, 20)              # fingertip ids
PALM_Z_MARGIN = 0.015                  # knuckles must be this much closer (z)


def _handedness_label(handedness) -> str:
    """
    Pull the "Left"/"Right" string out of a MediaPipe Handedness message.

    The field layout differs across MediaPipe builds: older ones exposed
    `classification[i].category_name`, while 0.10.x exposes the string in
    `classification[i].label` and leaves `display_name` empty.  Reading
    `category_name` directly raised AttributeError on 0.10.14, so probe the
    available attributes instead of assuming one layout.
    """
    if not handedness.classification:
        return "?"
    c = handedness.classification[0]
    name = getattr(c, "label", "") or ""
    if not name:
        dn = getattr(c, "display_name", None)
        name = getattr(dn, "category_name", "") or "" if dn is not None else ""
    return name or "?"


class FaceHandDetector:
    """Thin wrapper around MediaPipe FaceMesh + Hands (video tracking mode)."""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,            # lighter model → faster FPS
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ------------------------------------------------------------------ #
    def process(self, frame_bgr):
        """Run both models on a BGR frame; returns the dict documented above."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False        # mediapipe prefers read-only input
        face_res = self.face_mesh.process(rgb)
        hand_res = self.hands.process(rgb)
        return {
            "face": self._parse_face(face_res),
            "hands": self._parse_hands(hand_res),
            "raw": list(face_res.multi_face_landmarks or []),
            "raw_hands": list(hand_res.multi_hand_landmarks or []),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_face(res):
        multi = res.multi_face_landmarks
        if not multi:
            return None
        lm = multi[0].landmark
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        upper, lower = pts[LIP_UPPER, :2], pts[LIP_LOWER, :2]
        left, right = pts[MOUTH_LEFT, :2], pts[MOUTH_RIGHT, :2]
        mouth_center = (upper + lower) / 2.0
        width = float(np.linalg.norm(left - right)) + 1e-6
        openness = float(np.linalg.norm(upper - lower)) / width
        return {"landmarks": pts, "mouth_center": mouth_center, "openness": openness}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_hands(res):
        out = []
        multi = res.multi_hand_landmarks
        if not multi:
            return out
        labels = [_handedness_label(h) for h in (res.multi_handedness or [])]
        for i, hand in enumerate(multi):
            lm = hand.landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            # Palm-facing heuristic: when the palm looks at the camera the
            # fingers point AWAY from it, so knuckles are closer (smaller z
            # in MediaPipe) than the fingertips.
            knuckle_z = float(np.mean([pts[j, 2] for j in KNUCKLE_IDS]))
            tip_z = float(np.mean([pts[j, 2] for j in TIP_IDS]))
            out.append({
                "landmarks": pts,
                "tip_index": pts[TIP_INDEX, :2].copy(),
                "tip_middle": pts[TIP_MIDDLE, :2].copy(),
                "palm_center": pts[[0] + list(KNUCKLE_IDS), :2].mean(axis=0),
                "palm_facing": bool(knuckle_z + PALM_Z_MARGIN < tip_z),
                "handedness": labels[i] if i < len(labels) else "?",
            })
        return out

    # ------------------------------------------------------------------ #
    # Overlay colours (BGR)
    # ------------------------------------------------------------------ #
    TRACK_LINE = (0, 255, 255)     # yellow  — lips + hand bones
    TRACK_DOT = (0, 0, 255)        # red     — lip + hand joints
    TRACK_THICK = 2
    TRACK_RADIUS = 3

    @staticmethod
    def _draw_mesh(frame, landmarks, connections, h, w):
        """
        Draw one mesh as yellow lines + red dots on the joint endpoints.

        Rolled by hand rather than using MediaPipe's draw_landmarks(), which
        draws a circle for *every* landmark in the array -- 468 circles per
        frame for a FaceMesh, even though only the ~40 lip points are
        connected.  Restricting dots to the connected joints cuts the overlay
        from ~4.5 ms to well under 1 ms.
        """
        pts = landmarks.landmark
        coords = [(int(p.x * w), int(p.y * h)) for p in pts]
        n = len(coords)
        touched = set()
        for a, b in connections:
            if a < n and b < n:
                cv2.line(frame, coords[a], coords[b],
                         FaceHandDetector.TRACK_LINE,
                         FaceHandDetector.TRACK_THICK, cv2.LINE_AA)
                touched.add(a)
                touched.add(b)
        r = FaceHandDetector.TRACK_RADIUS
        for i in touched:
            cv2.circle(frame, coords[i], r, FaceHandDetector.TRACK_DOT,
                       -1, cv2.LINE_AA)

    def draw_landmarks(self, frame, result):
        """
        Draw the lip mesh and any hand skeletons: yellow connecting lines
        with red dots on every joint.  Drawn every frame, so it stays visible
        without toggling the debug overlay.
        """
        h, w = frame.shape[:2]
        lips = mp.solutions.face_mesh.FACEMESH_LIPS
        bones = mp.solutions.hands.HAND_CONNECTIONS
        for face_lm in result.get("raw", []):
            self._draw_mesh(frame, face_lm, lips, h, w)
        for hand_lm in result.get("raw_hands", []):
            self._draw_mesh(frame, hand_lm, bones, h, w)

    # ------------------------------------------------------------------ #
    def draw_debug(self, frame, result):
        """Extra numeric overlay (toggled with `d`): mouth centre + openness."""
        h, w = frame.shape[:2]
        face = result.get("face")
        if face is not None:
            mx = int(face["mouth_center"][0] * w)
            my = int(face["mouth_center"][1] * h)
            cv2.circle(frame, (mx, my), 6, (60, 60, 255), 2)
            cv2.putText(frame, f"open {face['openness']:.2f}", (mx + 10, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1, cv2.LINE_AA)
        for hand in result.get("hands", []):
            for tip in (hand["tip_index"], hand["tip_middle"]):
                cv2.circle(frame, (int(tip[0] * w), int(tip[1] * h)), 5,
                           (60, 220, 60), -1)

    def close(self):
        self.face_mesh.close()
        self.hands.close()
