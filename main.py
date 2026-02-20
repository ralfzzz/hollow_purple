import cv2
import mediapipe as mp
import math
import random
import time
import numpy as np

# ===== INIT =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ===== HELPER =====
def finger_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

# ===== PARTICLES =====
particles = []

def spawn_edge_particles(center, radius, color, amount, speed_range):
    for _ in range(amount):
        angle = random.uniform(0, 2 * math.pi)
        x = center[0] + math.cos(angle) * radius
        y = center[1] + math.sin(angle) * radius
        speed = random.uniform(*speed_range)

        particles.append({
            "x": x,
            "y": y,
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "life": random.randint(15, 30),
            "color": color
        })

def update_particles(img):
    for p in particles[:]:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1

        if p["life"] > 0:
            alpha = p["life"] / 30
            color = (
                int(p["color"][0] * alpha),
                int(p["color"][1] * alpha),
                int(p["color"][2] * alpha)
            )
            cv2.circle(img, (int(p["x"]), int(p["y"])), 2, color, -1)
        else:
            particles.remove(p)

# ===== LIGHTNING ARC =====
def draw_lightning(img, p1, p2, segments=12):
    points = []
    for i in range(segments + 1):
        t = i / segments
        x = int(p1[0] * (1 - t) + p2[0] * t + random.randint(-20, 20))
        y = int(p1[1] * (1 - t) + p2[1] * t + random.randint(-20, 20))
        points.append((x, y))

    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (255,255,255), 2)

# ===== STATE =====
red_active = False
blue_active = False
red_pos = None
blue_pos = None

purple_active = False
purple_merging = False
purple_exploding = False
purple_pos = None

merge_angle = 0
merge_radius = 0
merge_center = None

charge_radius = 0
max_radius = 120
explosion_radius = 0
max_explosion = 600

# ===== MAIN LOOP =====
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

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

            if label == "Right" and not purple_active and not purple_merging:
                if only_index:
                    red_active = True
                    red_pos = (x, y)

            if label == "Left" and not purple_active and not purple_merging:
                if only_index:
                    blue_active = True
                    blue_pos = (x, y)

            if two_fingers:
                two_finger_pos = (x, y)

            if fist and purple_active and not purple_exploding:
                purple_exploding = True
                explosion_radius = charge_radius

    # ===== DRAW NORMAL RED =====
    if red_active and red_pos and not purple_active and not purple_merging:
        radius = 60 + pulse
        overlay = img.copy()
        for i in range(3):
            cv2.circle(overlay, red_pos, radius + i*15, (0,0,255), -1)
        img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
        cv2.circle(img, red_pos, radius, (0,0,255), -1)
        spawn_edge_particles(red_pos, radius, (0,0,255), 3, (1,3))

    # ===== DRAW NORMAL BLUE =====
    if blue_active and blue_pos and not purple_active and not purple_merging:
        radius = 60 + pulse
        overlay = img.copy()
        for i in range(3):
            cv2.circle(overlay, blue_pos, radius + i*15, (255,0,0), -1)
        img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
        cv2.circle(img, blue_pos, radius, (255,0,0), -1)
        spawn_edge_particles(blue_pos, radius, (255,0,0), 3, (1,3))

    # ===== START MERGE =====
    if not purple_active and not purple_merging and red_active and blue_active and red_pos and blue_pos:
        distance = math.hypot(red_pos[0] - blue_pos[0], red_pos[1] - blue_pos[1])
        if distance < 120:
            purple_merging = True
            merge_center = ((red_pos[0]+blue_pos[0])//2, (red_pos[1]+blue_pos[1])//2)
            merge_radius = distance // 2
            merge_angle = 0

    # ===== MERGING EFFECT =====
    if purple_merging:

        dark = np.zeros_like(img)
        img = cv2.addWeighted(img, 0.3, dark, 0.7, 0)

        merge_angle += 0.25
        merge_radius *= 0.96

        red_x = int(merge_center[0] + math.cos(merge_angle) * merge_radius)
        red_y = int(merge_center[1] + math.sin(merge_angle) * merge_radius)

        blue_x = int(merge_center[0] + math.cos(merge_angle + math.pi) * merge_radius)
        blue_y = int(merge_center[1] + math.sin(merge_angle + math.pi) * merge_radius)

        cv2.circle(img, (red_x, red_y), 40, (0,0,255), -1)
        cv2.circle(img, (blue_x, blue_y), 40, (255,0,0), -1)

        draw_lightning(img, (red_x, red_y), (blue_x, blue_y))

        cv2.circle(img, merge_center, int(merge_radius*1.5), (255,255,255), 1)

        spawn_edge_particles((red_x, red_y), 40, (0,0,255), 6, (3,6))
        spawn_edge_particles((blue_x, blue_y), 40, (255,0,0), 6, (3,6))

        if merge_radius < 15:
            flash = np.ones_like(img) * 255
            img = cv2.addWeighted(img, 0.3, flash, 0.7, 0)

        if merge_radius < 8:
            purple_merging = False
            purple_active = True
            purple_pos = merge_center
            charge_radius = 0
            red_active = False
            blue_active = False

    # ===== PURPLE NORMAL =====
    if purple_active and not purple_exploding:

        if two_finger_pos:
            purple_pos = two_finger_pos

        if charge_radius < max_radius:
            charge_radius += 4

        radius = charge_radius + pulse
        overlay = img.copy()

        for i in range(4):
            cv2.circle(overlay, purple_pos, radius + i*25, (255,0,255), -1)

        img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
        cv2.circle(img, purple_pos, radius, (255,0,255), -1)

        spawn_edge_particles(purple_pos, radius, (255,0,255), 8, (2,5))

    # ===== EXPLOSION =====
    if purple_exploding:

        explosion_radius += 50
        overlay = img.copy()

        for i in range(5):
            cv2.circle(overlay, purple_pos, explosion_radius + i*40, (255,0,255), -1)

        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        spawn_edge_particles(purple_pos, explosion_radius, (255,0,255), 25, (5,10))

        if explosion_radius > max_explosion:
            purple_active = False
            purple_exploding = False
            purple_pos = None
            charge_radius = 0
            explosion_radius = 0

    update_particles(img)

    cv2.imshow("Hollow Purple - Blackhole Collision Mode", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("Hollow Purple - Blackhole Collision Mode", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()