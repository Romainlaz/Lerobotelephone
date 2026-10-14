"""
Test du suivi du regard : webcam -> visage -> angles -> tête simulée.

La webcam joue le rôle du téléphone fixé dans le torse ; la tête dessinée
à droite joue le rôle de la tête du robot, qui doit regarder le visage.

Installation : pip install opencv-python numpy
Lancement    : python suivi_visage.py   (touche q pour quitter)
"""
import math
import time

import cv2
import numpy as np

# --- Paramètres à régler -----------------------------------------------------
HFOV_DEG = 70.0          # champ de vision horizontal supposé de la webcam
FACE_WIDTH_M = 0.15      # largeur moyenne d'un visage, pour estimer la distance
DEAD_ZONE_DEG = 2.0      # en dessous, la cible ne bouge pas (anti-tremblement)
TAU_S = 0.15             # constante de lissage : plus grand = plus doux et plus lent
MAX_SPEED_DEG_S = 180.0  # vitesse max de la tête
YAW_LIMIT_DEG = 60.0     # butées logicielles
PITCH_LIMIT_DEG = 30.0
# -----------------------------------------------------------------------------


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def step(current, target, limit, dt):
    """Avance la tête vers la cible avec lissage exponentiel et limite de vitesse."""
    target = clamp(target, -limit, limit)
    delta = (target - current) * (1 - math.exp(-dt / TAU_S))
    max_delta = MAX_SPEED_DEG_S * dt
    return current + clamp(delta, -max_delta, max_delta)


def draw_head(h, yaw, pitch, has_face):
    """Dessine une tête vue de face dont les yeux suivent yaw/pitch."""
    panel = np.full((h, 320, 3), 30, np.uint8)
    cx, cy = 160, h // 2
    color = (200, 200, 200) if has_face else (90, 90, 90)
    cv2.circle(panel, (cx, cy), 100, color, 2)
    dx = int(yaw / YAW_LIMIT_DEG * 50)
    dy = int(-pitch / PITCH_LIMIT_DEG * 40)
    for ex in (cx - 35, cx + 35):
        cv2.circle(panel, (ex, cy - 10), 22, color, 2)
        cv2.circle(panel, (ex + dx // 3, cy - 10 + dy // 3), 9, (80, 220, 255), -1)
    # antennes
    cv2.line(panel, (cx - 40, cy - 95), (cx - 60 + dx // 2, cy - 150), color, 2)
    cv2.line(panel, (cx + 40, cy - 95), (cx + 60 + dx // 2, cy - 150), color, 2)
    return panel


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la webcam.")
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    head_yaw = head_pitch = 0.0
    target_yaw = target_pitch = 0.0
    last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # effet miroir, plus intuitif à l'écran
        h, w = frame.shape[:2]
        fx = (w / 2) / math.tan(math.radians(HFOV_DEG / 2))  # focale en pixels
        cx, cy = w / 2, h / 2

        now = time.time()
        dt = max(now - last, 1e-3)
        last = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

        dist = None
        if len(faces):
            # visage le plus grand = le plus proche
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            u, v = x + fw / 2, y + fh / 2
            yaw = math.degrees(math.atan((u - cx) / fx))
            pitch = -math.degrees(math.atan((v - cy) / fx))  # pixels carrés : fy ≈ fx
            dist = FACE_WIDTH_M * fx / fw

            if (abs(yaw - target_yaw) > DEAD_ZONE_DEG
                    or abs(pitch - target_pitch) > DEAD_ZONE_DEG):
                target_yaw, target_pitch = yaw, pitch

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.circle(frame, (int(u), int(v)), 4, (0, 255, 0), -1)

        head_yaw = step(head_yaw, target_yaw, YAW_LIMIT_DEG, dt)
        head_pitch = step(head_pitch, target_pitch, PITCH_LIMIT_DEG, dt)

        lines = [
            f"cible  lacet {target_yaw:+5.1f}  tangage {target_pitch:+5.1f}",
            f"tete   lacet {head_yaw:+5.1f}  tangage {head_pitch:+5.1f}",
            f"distance {dist:.2f} m" if dist else "distance  -",
            f"{1 / dt:4.0f} img/s",
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 25 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        view = np.hstack([frame, draw_head(h, head_yaw, head_pitch, len(faces) > 0)])
        cv2.imshow("Suivi du regard (q pour quitter)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()