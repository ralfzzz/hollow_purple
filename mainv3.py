import cv2
import mediapipe as mp
import math
import random
import time
import numpy as np

# ================= INIT =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ================= HELPER =================
def finger_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

# ================= EDGE PARTICLES =================
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

# ================= PARTICLE SPHERE =================
BALL_COUNT = 1500  # aman untuk realtime

class ParticleSphere:
    def __init__(self, color):
        self.color = color
        self.radius = 300
        self.points = []
        self.rot = 0
        self.speed = 10
        self.generate()

    def generate(self):
        self.points.clear()
        for _ in range(BALL_COUNT):
            r = random.random() * self.radius
            th = random.random() * 2 * math.pi
            ph = math.acos(2 * random.random() - 1)

            x = r * math.sin(ph) * math.cos(th)
            y = r * math.sin(ph) * math.sin(th)
            z = r * math.cos(ph)

            self.points.append((x, y, z))

    def draw(self, img, center, scale_factor=1, speed_mult=1, emit_amount=6):

        cx, cy = center
        self.rot += self.speed * speed_mult

        max_screen_radius = 0

        for x, y, z in self.points:

            x *= scale_factor
            y *= scale_factor
            z *= scale_factor

            rx = x * math.cos(self.rot) - z * math.sin(self.rot)
            rz = x * math.sin(self.rot) + z * math.cos(self.rot)

            focal = 400
            scale = focal / (focal + rz + 400)

            sx = int(cx + rx * scale)
            sy = int(cy + y * scale)
            size = max(1, int(4 * scale))

            if 0 < sx < img.shape[1] and 0 < sy < img.shape[0]:
                cv2.circle(img, (sx, sy), size, self.color, -1)

            if abs(rx * scale) > max_screen_radius:
                max_screen_radius = abs(rx * scale)

        # Emit edge energy
        spawn_edge_particles(center,
                             int(max_screen_radius),
                             self.color,
                             emit_amount,
                             (2, 5))

# ================= STATE =================
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

# ================= SPHERES =================
red_sphere = ParticleSphere((0,0,255))
blue_sphere = ParticleSphere((255,0,0))
purple_sphere = ParticleSphere((255,0,255))

# ================= MAIN LOOP =================
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    two_finger_pos = None

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

    # ===== RED =====
    if red_active and red_pos and not purple_active and not purple_merging:
        red_sphere.draw(img, red_pos, 0.6, 1, 6)

    # ===== BLUE =====
    if blue_active and blue_pos and not purple_active and not purple_merging:
        blue_sphere.draw(img, blue_pos, 0.6, 1, 6)

    # ===== START MERGE =====
    if not purple_active and not purple_merging and red_active and blue_active:
        dist_rb = math.hypot(red_pos[0]-blue_pos[0], red_pos[1]-blue_pos[1])
        if dist_rb < 120:
            purple_merging = True
            merge_center = ((red_pos[0]+blue_pos[0])//2,
                            (red_pos[1]+blue_pos[1])//2)
            merge_radius = dist_rb // 2
            merge_angle = 0

    # ===== MERGING =====
    if purple_merging:
        dark = np.zeros_like(img)
        img = cv2.addWeighted(img, 0.3, dark, 0.7, 0)

        merge_angle += 0.6
        merge_radius *= 0.90

        red_x = int(merge_center[0] + math.cos(merge_angle) * merge_radius)
        red_y = int(merge_center[1] + math.sin(merge_angle) * merge_radius)

        blue_x = int(merge_center[0] + math.cos(merge_angle + math.pi) * merge_radius)
        blue_y = int(merge_center[1] + math.sin(merge_angle + math.pi) * merge_radius)

        red_sphere.draw(img, (red_x, red_y), 0.5, 3, 15)
        blue_sphere.draw(img, (blue_x, blue_y), 0.5, 3, 15)

        # EXTRA ENERGY DI TENGAH
        spawn_edge_particles(merge_center, 30, (255,0,255), 20, (4,8))

        if merge_radius < 8:
            purple_merging = False
            purple_active = True
            purple_pos = merge_center
            charge_radius = 0
            red_active = False
            blue_active = False

    # ===== PURPLE =====
    if purple_active and not purple_exploding:
        if two_finger_pos:
            purple_pos = two_finger_pos

        if charge_radius < max_radius:
            charge_radius += 4

        scale = charge_radius / 120
        purple_sphere.draw(img, purple_pos, scale, 2, 12)

    # ===== EXPLOSION =====
    if purple_exploding:
        explosion_radius += 50
        spawn_edge_particles(purple_pos, explosion_radius,
                             (255,0,255), 40, (6,12))

        if explosion_radius > max_explosion:
            purple_active = False
            purple_exploding = False
            purple_pos = None
            charge_radius = 0
            explosion_radius = 0

    update_particles(img)

    cv2.imshow("Hollow Purple - Merge Particle Mode", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()