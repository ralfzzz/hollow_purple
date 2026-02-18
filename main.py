import cv2
import mediapipe as mp
import math
import random
import time

# ===== INIT =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# ===== HELPER =====
def finger_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

# ===== STATE =====
purple_active = False
purple_exploding = False
purple_pos = None

charge_radius = 0
max_radius = 120
explosion_radius = 0
max_explosion = 500

red_particles = []
blue_particles = []
purple_particles = []

# ===== PARTICLES =====
def update_particles(particles, img):
    for p in particles[:]:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1

        if p["life"] > 0:
            cv2.circle(img, (int(p["x"]), int(p["y"])), 3, p["color"], -1)
        else:
            particles.remove(p)

# ===== MAIN LOOP =====
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    red_pos = None
    blue_pos = None
    two_finger_pos = None

    pulse = int(10 * (math.sin(time.time() * 5) + 1))

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            label = results.multi_handedness[idx].classification[0].label

            index_up = finger_up(hand_landmarks, 8, 6)
            middle_up = finger_up(hand_landmarks, 12, 10)
            ring_up = finger_up(hand_landmarks, 16, 14)
            pinky_up = finger_up(hand_landmarks, 20, 18)

            thumb_up = (
                hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
                if label == "Right"
                else hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
            )

            only_index = index_up and not middle_up and not ring_up and not pinky_up
            two_fingers = index_up and middle_up and not ring_up and not pinky_up
            fist = not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up

            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # ===== RED =====
            if label == "Right" and only_index and not purple_active:
                red_pos = (x, y)
                overlay = img.copy()
                radius = 60 + pulse

                for i in range(3):
                    cv2.circle(overlay, red_pos, radius + i*15, (0,0,255), -1)

                img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
                cv2.circle(img, red_pos, radius, (0,0,255), -1)

            # ===== BLUE =====
            if label == "Left" and only_index and not purple_active:
                blue_pos = (x, y)
                overlay = img.copy()
                radius = 60 + pulse

                for i in range(3):
                    cv2.circle(overlay, blue_pos, radius + i*15, (255,0,0), -1)

                img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
                cv2.circle(img, blue_pos, radius, (255,0,0), -1)

            # ===== CONTROL =====
            if two_fingers:
                two_finger_pos = (x, y)

            # ===== TRIGGER EXPLOSION =====
            if fist and purple_active and not purple_exploding:
                purple_exploding = True
                explosion_radius = charge_radius

    # ===== MERGE =====
    if not purple_active and red_pos and blue_pos:
        distance = math.hypot(red_pos[0]-blue_pos[0], red_pos[1]-blue_pos[1])
        if distance < 100:
            purple_active = True
            purple_pos = (
                (red_pos[0] + blue_pos[0]) // 2,
                (red_pos[1] + blue_pos[1]) // 2
            )
            charge_radius = 0

    # ===== PURPLE NORMAL =====
    if purple_active and not purple_exploding:

        if two_finger_pos:
            purple_pos = two_finger_pos

        if charge_radius < max_radius:
            charge_radius += 4

        overlay = img.copy()
        radius = charge_radius + pulse

        for i in range(4):
            cv2.circle(overlay, purple_pos, radius + i*25, (255,0,255), -1)

        img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
        cv2.circle(img, purple_pos, radius, (255,0,255), -1)

    # ===== PURPLE EXPLOSION =====
    if purple_exploding:

        explosion_radius += 40

        overlay = img.copy()

        for i in range(5):
            cv2.circle(
                overlay,
                purple_pos,
                explosion_radius + i*40,
                (255,0,255),
                -1
            )

        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

        # BIG PARTICLE BURST
        for _ in range(20):
            purple_particles.append({
                "x": purple_pos[0],
                "y": purple_pos[1],
                "vx": random.uniform(-8,8),
                "vy": random.uniform(-8,8),
                "life": 40,
                "color": (255,0,255)
            })

        if explosion_radius > max_explosion:
            purple_active = False
            purple_exploding = False
            purple_pos = None
            charge_radius = 0
            explosion_radius = 0

    update_particles(purple_particles, img)

    cv2.imshow("Hollow Purple - BLAST MODE", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("Hollow Purple - BLAST MODE", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
