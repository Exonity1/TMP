"""Live-Tracking der Daumen- und Zeigefingerkuppen beider Hände.

Die vier Fingerkuppen werden markiert und zu einem Viereck verbunden.
Innerhalb des Vierecks wird ein wählbarer Bildeffekt angewendet.
Effekt wechseln mit den Zifferntasten oder E, beenden mit Q oder ESC.
R schaltet den Rahmen, H die Punkte und Koordinaten an oder aus.

Mit --virtual-cam wird das Bild (ohne Statuszeile) zusätzlich an die
OBS Virtual Camera gesendet und kann so in MS Teams als Kamera gewählt werden.
"""

import argparse
import contextlib
import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# Landmark-Indizes im MediaPipe-Handmodell
THUMB_TIP = 4
INDEX_TIP = 8

WINDOW = "Finger Tracking"
COLOR_THUMB = (0, 200, 255)   # BGR
COLOR_INDEX = (255, 120, 0)
COLOR_RECT = (0, 255, 0)


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Lade Handmodell herunter nach {MODEL_PATH.name} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def find_fingertips(result, width, height):
    """Liefert eine Liste von (label, (x, y), farbe) für alle erkannten Kuppen."""
    tips = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        side = "L" if handedness[0].category_name == "Left" else "R"
        for index, name, color in (
            (THUMB_TIP, "Daumen", COLOR_THUMB),
            (INDEX_TIP, "Zeigef.", COLOR_INDEX),
        ):
            lm = landmarks[index]
            point = (int(lm.x * width), int(lm.y * height))
            tips.append((f"{side} {name}", point, color))
    return tips


def order_as_polygon(points):
    """Sortiert Punkte im Kreis um ihren Schwerpunkt, damit sich das
    Viereck nie selbst überkreuzt, egal wie die Hände gehalten werden."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def fx_grayscale(roi, t):
    return cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def fx_rainbow(roi, t):
    """Extreme Sättigung, der Farbton wandert mit der Zeit einmal im Kreis."""
    h, s, v = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    # OpenCV-Hue läuft von 0 bis 179; ein Umlauf dauert 3 Sekunden
    h = ((h.astype(np.int32) + int(t * 60)) % 180).astype(np.uint8)
    s = cv2.convertScaleAbs(s, alpha=3.0, beta=40)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def fx_negative(roi, t):
    return cv2.bitwise_not(roi)


def fx_thermal(roi, t):
    return cv2.applyColorMap(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)


def fx_pixelate(roi, t):
    height, width = roi.shape[:2]
    small = cv2.resize(
        roi, (max(1, width // 16), max(1, height // 16)), interpolation=cv2.INTER_AREA
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def fx_blur(roi, t):
    return cv2.GaussianBlur(roi, (0, 0), 15)


# (Anzeigename, Funktion) - die Position in der Liste ist die Zifferntaste
EFFECTS = [
    ("Kein Effekt", None),
    ("Schwarz-Weiss", fx_grayscale),
    ("Regenbogen", fx_rainbow),
    ("Negativ", fx_negative),
    ("Waermebild", fx_thermal),
    ("Pixel", fx_pixelate),
    ("Unscharf", fx_blur),
]


def apply_effect(frame, corners, effect, t):
    """Wendet den Effekt nur innerhalb des Vierecks an."""
    polygon = np.array(corners, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(polygon)
    # Auf das Bild begrenzen, falls eine Kuppe knapp außerhalb liegt
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon - (x0, y0)], 255)
    cv2.copyTo(effect(roi, t), mask, roi)


def draw_overlay(frame, tips, show_border=True, show_markers=True):
    """Zeichnet den Rahmen des Vierecks sowie Kuppen und Koordinaten."""
    if show_border and len(tips) == 4:
        corners = order_as_polygon([point for _, point, _ in tips])
        for start, end in zip(corners, corners[1:] + corners[:1]):
            cv2.line(frame, start, end, COLOR_RECT, 3, cv2.LINE_AA)

    if show_markers:
        for label, (x, y), color in tips:
            cv2.circle(frame, (x, y), 10, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame, f"{label} ({x}, {y})", (x + 14, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
    return frame


def mirror_tips(tips, width):
    return [(label, (width - 1 - x, y), color) for label, (x, y), color in tips]


def draw_hud(frame, lines):
    for row, text in enumerate(lines):
        origin = (10, 28 + 26 * row)
        cv2.putText(
            frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )


def open_virtual_camera(width, height):
    """Öffnet die OBS Virtual Camera als Ausgabe (OBS muss installiert sein)."""
    import pyvirtualcam

    try:
        return pyvirtualcam.Camera(
            width, height, 30, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs"
        )
    except RuntimeError as error:
        raise SystemExit(
            "Die OBS Virtual Camera konnte nicht geöffnet werden.\n"
            "- Ist OBS Studio installiert?\n"
            "- Die virtuelle Kamera darf nicht gleichzeitig in OBS gestartet sein.\n"
            f"Fehlermeldung: {error}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Index der Webcam")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--virtual-cam", action="store_true",
        help="Bild zusätzlich an die OBS Virtual Camera senden (z. B. für Teams)",
    )
    args = parser.parse_args()

    ensure_model()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Webcam {args.camera} konnte nicht geöffnet werden.")

    start = time.monotonic()
    last = start
    fps = 0.0
    effect_index = 2
    show_border = True
    # Für die Ausgabe an Teams sind Punkte und Koordinaten zunächst aus
    show_markers = not args.virtual_cam
    virtual_cam = None

    with contextlib.ExitStack() as stack:
        landmarker = stack.enter_context(create_landmarker())
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Kein Bild von der Webcam erhalten.")
                break

            # Spiegeln, damit sich das Bild wie ein Spiegel verhält
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now = time.monotonic()
            result = landmarker.detect_for_video(image, int((now - start) * 1000))

            name, effect = EFFECTS[effect_index]
            tips = find_fingertips(result, width, height)
            if effect is not None and len(tips) == 4:
                corners = order_as_polygon([point for _, point, _ in tips])
                apply_effect(frame, corners, effect, now - start)

            if args.virtual_cam:
                if virtual_cam is None:
                    virtual_cam = stack.enter_context(open_virtual_camera(width, height))
                    print(f"Sende an: {virtual_cam.device}")
                # Die Ausgabe wird zurückgespiegelt: andere sehen dich seitenrichtig
                # wie bei einer normalen Webcam. Das HUD wird nie mitgesendet.
                output = cv2.flip(frame, 1)
                draw_overlay(output, mirror_tips(tips, width), show_border, show_markers)
                virtual_cam.send(output)

            draw_overlay(frame, tips, show_border, show_markers)

            if now > last:
                fps = 0.9 * fps + 0.1 / (now - last)
            last = now
            draw_hud(frame, [
                f"{fps:4.1f} FPS  |  Effekt {effect_index}: {name}  |  "
                f"0-{len(EFFECTS) - 1} / E = Effekt  |  Q/ESC = Beenden",
                f"R = Rahmen {'an' if show_border else 'aus'}  |  "
                f"H = Punkte {'an' if show_markers else 'aus'}  |  "
                f"Virtuelle Kamera: {'sendet' if virtual_cam else 'aus'}",
            ])

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("e"):
                effect_index = (effect_index + 1) % len(EFFECTS)
            elif key == ord("r"):
                show_border = not show_border
            elif key == ord("h"):
                show_markers = not show_markers
            elif ord("0") <= key < ord("0") + len(EFFECTS):
                effect_index = key - ord("0")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
