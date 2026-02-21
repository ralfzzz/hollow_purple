import cv2
import mediapipe as mp
import math
import random
import numpy as np

# ================= INIT =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ================= SETTINGS =================
BALL_COUNT = 900
FOCAL_LENGTH = 300

# ================= HELPER =================
def finger_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

# ================= EDGE PARTICLES =================
particles = []

def spawn_edge_particles(center, radius, color, amount, speed_range):
    for _ in range(amount):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*speed_range)

        particles.append({
            "x": center[0] + math.cos(angle) * radius,
            "y": center[1] + math.sin(angle) * radius,
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "life": random.randint(15, 40),
            "color": color
        })

def update_particles(img):
    for p in particles[:]:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1

        if p["life"] > 0:
            alpha = p["life"] / 40.0
            color = (
                int(p["color"][0] * alpha),
                int(p["color"][1] * alpha),
                int(p["color"][2] * alpha)
            )
            cv2.circle(img, (int(p["x"]), int(p["y"])), 3, color, -1)
        else:
            particles.remove(p)

# ================= PARTICLE SPHERE =================
class ParticleSphere:
    def __init__(self, color):
        self.color = color
        self.radius = 250
        self.points = []
        self.rot = 0
        self.speed = 0.7
        self.generate()

    def generate(self):
        for _ in range(BALL_COUNT):
            r = random.random() * self.radius
            theta = random.random() * 2 * math.pi
            phi = math.acos(2 * random.random() - 1)

            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)

            self.points.append((x, y, z))

    def draw(self, img, center, scale_factor=1, speed_mult=1,
             emit_amount=6, glow_strength=0.6, glow_size=3):

        cx, cy = center
        self.rot += self.speed * speed_mult

        glow_layer = np.zeros_like(img)
        max_screen_radius = 0

        for x, y, z in self.points:
            x *= scale_factor
            y *= scale_factor
            z *= scale_factor

            rx = x * math.cos(self.rot) - z * math.sin(self.rot)
            rz = x * math.sin(self.rot) + z * math.cos(self.rot)

            scale = FOCAL_LENGTH / (FOCAL_LENGTH + rz + 400)

            sx = int(cx + rx * scale)
            sy = int(cy + y * scale)
            size = max(1, int(4 * scale))

            if 0 < sx < img.shape[1] and 0 < sy < img.shape[0]:
                cv2.circle(img, (sx, sy), size, self.color, -1)
                cv2.circle(glow_layer, (sx, sy), size + glow_size, self.color, -1)

            if abs(rx * scale) > max_screen_radius:
                max_screen_radius = abs(rx * scale)

        glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), 6)
        img[:] = cv2.addWeighted(img, 1.0, glow_layer, glow_strength, 0)

        spawn_edge_particles(center, int(max_screen_radius), self.color, emit_amount, (2, 5))

# ================= STATE =================
red_active = False
blue_active = False
purple_active = False
purple_merging = False
purple_exploding = False

red_pos = None
blue_pos = None
purple_pos = None

merge_angle = 0
merge_radius = 0
merge_center = None

charge_radius = 0
max_radius = 140

explosion_radius = 0
max_explosion = 600

# Explosion & merge effects
shake_offset = (0, 0)
shake_duration = 0
dark_hold_duration = 0
dark_fade_alpha = 0
merge_dim_alpha = 0          # alpha untuk efek remang selama merging
just_exploded = False
explosion_cooldown = 0
initial_explosion_burst = False

# ================= SPHERES =================
red_sphere = ParticleSphere((0, 0, 255))
blue_sphere = ParticleSphere((255, 0, 0))
purple_sphere = ParticleSphere((255, 0, 255))

# ================= MAIN LOOP =================
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    if shake_duration > 0:
        shake_duration -= 1
        shake_offset = (random.randint(-20, 20), random.randint(-20, 20))
    else:
        shake_offset = (0, 0)

    img_with_shake = img.copy()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    two_finger_pos = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(img_with_shake, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            label = results.multi_handedness[idx].classification[0].label

            index_up = finger_up(hand_landmarks, 8, 6)
            middle_up = finger_up(hand_landmarks, 12, 10)
            ring_up = finger_up(hand_landmarks, 16, 14)
            pinky_up = finger_up(hand_landmarks, 20, 18)

            only_index = index_up and not middle_up and not ring_up and not pinky_up
            two_fingers = index_up and middle_up and not ring_up and not pinky_up
            fist = not index_up and not middle_up and not ring_up and not pinky_up

            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            if label == "Right" and only_index and not purple_active:
                red_active = True
                red_pos = (x, y)

            if label == "Left" and only_index and not purple_active:
                blue_active = True
                blue_pos = (x, y)

            if two_fingers:
                two_finger_pos = (x, y)

            if fist and purple_active and not purple_exploding and not just_exploded:
                purple_exploding = True
                explosion_radius = charge_radius
                shake_duration = 30
                dark_hold_duration = 45
                dark_fade_alpha = 255
                just_exploded = True
                explosion_cooldown = 90
                initial_explosion_burst = False

    if explosion_cooldown > 0:
        explosion_cooldown -= 1
    else:
        just_exploded = False

    # RED
    if red_active and red_pos and not purple_active and not purple_merging:
        red_sphere.draw(img_with_shake, (red_pos[0] + shake_offset[0], red_pos[1] + shake_offset[1]),
                        scale_factor=0.6, speed_mult=1, emit_amount=4, glow_strength=0.7, glow_size=4)

    # BLUE
    if blue_active and blue_pos and not purple_active and not purple_merging:
        blue_sphere.draw(img_with_shake, (blue_pos[0] + shake_offset[0], blue_pos[1] + shake_offset[1]),
                         scale_factor=0.6, speed_mult=1, emit_amount=4, glow_strength=0.7, glow_size=4)

    # MERGE START
    if red_active and blue_active and not purple_merging and not purple_active:
        dist = math.hypot(red_pos[0]-blue_pos[0], red_pos[1]-blue_pos[1])
        if dist < 120:
            purple_merging = True
            merge_center = ((red_pos[0]+blue_pos[0])//2, (red_pos[1]+blue_pos[1])//2)
            merge_radius = dist//2
            merge_angle = 0
            merge_dim_alpha = 140  # mulai remang saat merge dimulai

            # MERGING (dengan efek remang hitam pure)
    if purple_merging:
        merge_angle += 0.6
        merge_radius *= 0.90

        red_x = int(merge_center[0] + math.cos(merge_angle)*merge_radius)
        red_y = int(merge_center[1] + math.sin(merge_angle)*merge_radius)

        blue_x = int(merge_center[0] + math.cos(merge_angle+math.pi)*merge_radius)
        blue_y = int(merge_center[1] + math.sin(merge_angle+math.pi)*merge_radius)

        red_sphere.draw(img_with_shake, (red_x + shake_offset[0], red_y + shake_offset[1]), 0.5, 3, 12, 0.8, 5)
        blue_sphere.draw(img_with_shake, (blue_x + shake_offset[0], blue_y + shake_offset[1]), 0.5, 3, 12, 0.8, 5)

        spawn_edge_particles(merge_center, 30, (255,0,255), 15, (4,8))

        # Efek remang hitam pure
        overlay_dim = np.zeros((h, w, 3), dtype=np.uint8)
        overlay_dim[:] = (0, 0, 0)  # hitam polos
        current_alpha = merge_dim_alpha / 255.0
        cv2.addWeighted(overlay_dim, current_alpha, img_with_shake, 1.0 - current_alpha, 0, img_with_shake)

        if merge_radius < 8:
            purple_merging = False
            purple_active = True
            purple_pos = merge_center
            charge_radius = 0
            red_active = False
            blue_active = False
            merge_dim_alpha = 140  # mulai fade dari nilai ini

    # Fade out remang setelah merge selesai
    if not purple_merging and merge_dim_alpha > 0:
        merge_dim_alpha -= 6
        if merge_dim_alpha > 0:
            overlay_dim = np.zeros((h, w, 3), dtype=np.uint8)
            overlay_dim[:] = (0, 0, 0)  # hitam polos juga pas fade
            current_alpha = merge_dim_alpha / 255.0
            cv2.addWeighted(overlay_dim, current_alpha, img_with_shake, 1.0 - current_alpha, 0, img_with_shake)

    # PURPLE CHARGE
    if purple_active and not purple_exploding:
        if two_finger_pos:
            purple_pos = two_finger_pos

        if charge_radius < max_radius:
            charge_radius += 4

        scale = charge_radius / 120

        purple_sphere.draw(img_with_shake, (purple_pos[0] + shake_offset[0], purple_pos[1] + shake_offset[1]),
                           scale_factor=scale, speed_mult=2, emit_amount=15, glow_strength=1.0, glow_size=6)

    # EXPLOSION
    if purple_exploding:
        explosion_radius += 50

        if dark_hold_duration > 0:
            dark_hold_duration -= 1
            img_with_shake[:] = 0
        else:
            if dark_fade_alpha > 0:
                overlay = np.full((h, w, 3), 0, dtype=np.uint8)
                cv2.addWeighted(overlay, dark_fade_alpha / 255.0, img_with_shake, 1.0, 0, img_with_shake)
                dark_fade_alpha -= 8

        overlay_circle = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(overlay_circle,
                   (purple_pos[0] + shake_offset[0], purple_pos[1] + shake_offset[1]),
                   int(explosion_radius),
                   (0, 0, 0), -1)
        alpha_circle = min(160, int(explosion_radius / 4))
        cv2.addWeighted(overlay_circle, alpha_circle / 255.0, img_with_shake, 1.0, 0, img_with_shake)

        if not initial_explosion_burst:
            spawn_edge_particles(purple_pos, explosion_radius * 1.3, (255, 0, 255), 350, (10, 25))
            initial_explosion_burst = True

        spawn_edge_particles(purple_pos, explosion_radius, (255, 0, 255), 120, (8, 20))

        if explosion_radius > max_explosion:
            purple_active = False
            purple_exploding = False
            charge_radius = 0
            explosion_radius = 0
            dark_hold_duration = 0
            dark_fade_alpha = 0
            initial_explosion_burst = False

    update_particles(img_with_shake)

    cv2.imshow("Hollow Purple - All Spheres Glow", img_with_shake)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()