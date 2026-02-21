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

# ================= BINTANG 3D VECTORIZED (NUMPY) =================
stars_pos = np.array([])          # posisi 3D
stars_props = []                  # size & brightness
star_rot = np.array([0.0, 0.0, 0.0])

def spawn_stars_3d(amount, radius_range=(100, 600)):
    global stars_pos, stars_props
    pos_list = []
    props_list = []
    for _ in range(amount):
        r = random.uniform(radius_range[0], radius_range[1])
        theta = random.uniform(0, 2 * math.pi)
        phi = math.acos(2 * random.random() - 1)
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        size = random.randint(1, 3)
        brightness = random.randint(200, 255)
        pos_list.append([x, y, z])
        props_list.append({"size": size, "brightness": brightness})

    stars_pos = np.array(pos_list)
    stars_props = props_list

def update_stars_3d(img, center, focal_length=500, rot_speed=np.array([0.08, 0.12, 0.05])):
    global star_rot, stars_pos
    star_rot += rot_speed

    if len(stars_pos) == 0:
        return

    # Rotasi X
    cos_x, sin_x = math.cos(star_rot[0]), math.sin(star_rot[0])
    rot_x = np.array([[1, 0, 0],
                      [0, cos_x, -sin_x],
                      [0, sin_x, cos_x]])
    stars_pos = stars_pos @ rot_x.T

    # Rotasi Y
    cos_y, sin_y = math.cos(star_rot[1]), math.sin(star_rot[1])
    rot_y = np.array([[cos_y, 0, sin_y],
                      [0, 1, 0],
                      [-sin_y, 0, cos_y]])
    stars_pos = stars_pos @ rot_y.T

    # Rotasi Z
    cos_z, sin_z = math.cos(star_rot[2]), math.sin(star_rot[2])
    rot_z = np.array([[cos_z, -sin_z, 0],
                      [sin_z, cos_z, 0],
                      [0, 0, 1]])
    stars_pos = stars_pos @ rot_z.T

    # Proyeksi
    scale = focal_length / (focal_length + stars_pos[:, 2])
    sx = (center[0] + stars_pos[:, 0] * scale).astype(int)
    sy = (center[1] + stars_pos[:, 1] * scale).astype(int)
    sizes = np.maximum(1, (np.array([p["size"] for p in stars_props]) * scale).astype(int))

    # Draw
    brights = np.array([p["brightness"] for p in stars_props])
    for i in range(len(stars_pos)):
        if 0 < sx[i] < img.shape[1] and 0 < sy[i] < img.shape[0]:
            bright = int(brights[i])
            cv2.circle(img, (sx[i], sy[i]), sizes[i], (bright, bright, bright), -1)

# ================= BLACKHOLE BERPUTAR =================
blackhole_rot = 0.0

def draw_blackhole(img, center, rot_speed=0.08, radius=150):
    global blackhole_rot
    blackhole_rot += rot_speed

    cv2.circle(img, center, radius, (0, 0, 0), -1)

    glow_layer = np.zeros_like(img)
    for i in range(4):
        angle = blackhole_rot + i * (2 * math.pi / 4)
        rx = int(center[0] + math.cos(angle) * (radius + 60 + i*20))
        ry = int(center[1] + math.sin(angle) * (radius + 60 + i*20))
        cv2.circle(glow_layer, (rx, ry), 40 - i*8, (220, 100, 255), -1)

    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), 8)
    cv2.addWeighted(glow_layer, 0.9, img, 1.0, 0, img)

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
domain_expansion_active = False
domain_countdown_active = False

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

shake_offset = (0, 0)
shake_duration = 0
dark_hold_duration = 0
dark_fade_alpha = 0
domain_timer = 0
domain_duration = 40
countdown_timer = 0
countdown_duration = 25

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

    frame = img.copy()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    two_finger_pos = None
    domain_triggered = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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

            # Spawn red/blue hanya kalau domain/countdown tidak aktif
            if not domain_expansion_active and not domain_countdown_active:
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

            if two_fingers and not domain_expansion_active and not domain_countdown_active:
                if not red_active and not blue_active and not purple_active:
                    domain_triggered = True

    if domain_triggered:
        domain_countdown_active = True
        countdown_timer = countdown_duration

    if domain_countdown_active:
        countdown_timer -= 1

        overlay = np.zeros_like(frame)
        overlay[:] = (0, 0, 0)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        seconds_left = 3
        if countdown_timer <= 15:
            seconds_left = 2
        if countdown_timer <= 5:
            seconds_left = 1

        cv2.putText(frame, f"Processing Domain in {seconds_left}..", (w//2 - 500, h//2),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 6, cv2.LINE_AA)

        if countdown_timer <= 0:
            domain_countdown_active = False
            domain_expansion_active = True
            domain_timer = domain_duration
            spawn_stars_3d(200)

    if explosion_cooldown > 0:
        explosion_cooldown -= 1
    else:
        just_exploded = False

    # RED & BLUE DRAW (hanya kalau tidak domain/countdown)
    if red_active and red_pos and not purple_active and not purple_merging and not domain_expansion_active and not domain_countdown_active:
        red_sphere.draw(frame, (red_pos[0] + shake_offset[0], red_pos[1] + shake_offset[1]),
                        scale_factor=0.6, speed_mult=1, emit_amount=4, glow_strength=0.7, glow_size=4)

        # Tulisan Blue di atas tengah
        cv2.putText(frame, "the attractive forces of Red", (w//2 - 30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        
    if blue_active and blue_pos and not purple_active and not purple_merging and not domain_expansion_active and not domain_countdown_active:
        blue_sphere.draw(frame, (blue_pos[0] + shake_offset[0], blue_pos[1] + shake_offset[1]),
                         scale_factor=0.6, speed_mult=1, emit_amount=4, glow_strength=0.7, glow_size=4)
                
        # Tulisan Red di atas tengah
        cv2.putText(frame, "repulsive forces of Blue", (w//2 - 600, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

    # MERGE START
    if red_active and blue_active and not purple_merging and not purple_active:
        dist = math.hypot(red_pos[0] - blue_pos[0], red_pos[1] - blue_pos[1])
        if dist < 120:
            purple_merging = True
            merge_center = ((red_pos[0] + blue_pos[0]) // 2, (red_pos[1] + blue_pos[1]) // 2)
            merge_radius = dist // 2
            merge_angle = 0

    # MERGING
    if purple_merging:
        merge_angle += 0.6
        merge_radius *= 0.90

        red_x = int(merge_center[0] + math.cos(merge_angle) * merge_radius)
        red_y = int(merge_center[1] + math.sin(merge_angle) * merge_radius)

        blue_x = int(merge_center[0] + math.cos(merge_angle + math.pi) * merge_radius)
        blue_y = int(merge_center[1] + math.sin(merge_angle + math.pi) * merge_radius)

        red_sphere.draw(frame, (red_x + shake_offset[0], red_y + shake_offset[1]), 0.5, 3, 12, 0.8, 5)
        blue_sphere.draw(frame, (blue_x + shake_offset[0], blue_y + shake_offset[1]), 0.5, 3, 12, 0.8, 5)

        spawn_edge_particles(merge_center, 30, (255,0,255), 15, (4,8))

        if merge_radius < 8:
            purple_merging = False
            purple_active = True
            purple_pos = merge_center
            charge_radius = 0
            red_active = False
            blue_active = False

    # PURPLE CHARGE
    if purple_active and not purple_exploding:
            if two_finger_pos:
                purple_pos = two_finger_pos

            if charge_radius < max_radius:
                charge_radius += 4

            scale = charge_radius / 120

            purple_sphere.draw(frame, (purple_pos[0] + shake_offset[0], purple_pos[1] + shake_offset[1]),
                            scale_factor=scale, speed_mult=2, emit_amount=15, glow_strength=1.0, glow_size=6)
        
            # Tulisan Purple di atas tengah
            cv2.putText(frame, "Hollow Technique Purple", (w//2 - 180, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 0, 128), 2, cv2.LINE_AA)

    # DOMAIN EXPANSION
    if domain_expansion_active:
        domain_timer -= 1

        frame[:] = (40, 0, 40)
        center = (w // 2, h // 2)
        update_stars_3d(frame, center)

        draw_blackhole(frame, center)

        text_alpha = min(255, int(255 * (domain_timer / domain_duration))) if domain_timer > domain_duration // 2 else min(255, int(255 * (1 - (domain_timer / (domain_duration // 2)))))
        text_color = (text_alpha, text_alpha, text_alpha)

        cv2.putText(frame, "Infinite Void", (w//2 - 500, h//2),
                    cv2.FONT_HERSHEY_TRIPLEX, 3, text_color, 6, cv2.LINE_AA)

        info_color = (0, 0, 255)
        cv2.putText(frame, "domain last only 2sec, to much information!;)", (w//2 - 600, h//2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, info_color, 3, cv2.LINE_AA)

        if domain_timer <= 0:
            domain_expansion_active = False
            stars_3d = []

    # EXPLOSION
    if purple_exploding:
        explosion_radius += 50

        if dark_hold_duration > 0:
            dark_hold_duration -= 1
            frame[:] = 0
        else:
            if dark_fade_alpha > 0:
                overlay = np.full_like(frame, 0)
                cv2.addWeighted(overlay, dark_fade_alpha / 255.0, frame, 1.0, 0, frame)
                dark_fade_alpha -= 8

        overlay_circle = np.zeros_like(frame)
        cv2.circle(overlay_circle,
                   (purple_pos[0] + shake_offset[0], purple_pos[1] + shake_offset[1]),
                   int(explosion_radius),
                   (0, 0, 0), -1)
        alpha_circle = min(160, int(explosion_radius / 4))
        cv2.addWeighted(overlay_circle, alpha_circle / 255.0, frame, 1.0, 0, frame)

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

    update_particles(frame)

    cv2.imshow("Hollow Purple - All Spheres Glow", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()