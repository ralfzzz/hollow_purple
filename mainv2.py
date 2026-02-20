import cv2
import mediapipe as mp
import numpy as np
import math
import random

# ================= INIT =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

BALL_COUNT = 1400

# ================= CONFIG =================
class EnergyConfig:
    def __init__(self, core_color, arm_color):
        self.coreRadius = 80
        self.coreRatio = 0.6
        self.coreColor = core_color
        self.coreSize = 5
        self.spiralStartRadius = 40
        self.spiralRadius = 160
        self.spiralDepth = 120
        self.spiralSpeed = 25
        self.armColor = arm_color
        self.armSize = 2
        self.arms = 4

RED_CONFIG = EnergyConfig((0,0,255), (50,50,255))
BLUE_CONFIG = EnergyConfig((255,0,0), (255,150,150))
PURPLE_CONFIG = EnergyConfig((255,0,255), (255,150,255))

# ================= ENERGY SPHERE =================
class EnergySphere:

    def __init__(self, config):
        self.config = config
        self.points = []
        self.rot_x = 0
        self.rot_y = 0
        self.base_speed = 0.05
        self.speed_multiplier = 1.0
        self.scale_factor = 1.0
        self.pulse = 0
        self.tilt_angle = math.radians(45)
        self.generate()

    def generate(self):
        C = self.config
        self.points.clear()

        for i in range(BALL_COUNT):

            if i < BALL_COUNT * C.coreRatio:
                r = random.random() * C.coreRadius
                th = random.random() * 2 * math.pi
                ph = math.acos(2 * random.random() - 1)
                x = r * math.sin(ph) * math.cos(th)
                y = r * math.sin(ph) * math.sin(th)
                z = r * math.cos(ph)
                self.points.append((x,y,z,C.coreColor,C.coreSize+2))
            else:
                t = i / BALL_COUNT
                angle = t * C.spiralSpeed + ((i % C.arms) * (2*math.pi/C.arms))
                radius = C.spiralStartRadius + t * C.spiralRadius
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = (random.random() - 0.5) * (C.spiralDepth * t)
                self.points.append((x,y,z,C.armColor,C.armSize))

    def update_speed(self, mode):

        # ===== Dinamis normal =====
        self.base_speed += 0.0003

        if mode == "IDLE":
            self.speed_multiplier = 1.0

        elif mode == "MERGING":
            self.speed_multiplier = 3.0   # 🔥 makin cepat saat merge

        elif mode == "PURPLE":
            self.speed_multiplier = 4.0   # chaos mode

        elif mode == "EXPLOSION":
            self.speed_multiplier = 8.0   # ekstrem

    def draw(self, img, center, mode):

        self.update_speed(mode)

        cx, cy = center

        speed = self.base_speed * self.speed_multiplier

        # ===== Chaos mode =====
        if mode == "PURPLE":
            self.rot_x += speed * random.uniform(0.8, 1.5)
            self.rot_y += speed * random.uniform(0.8, 1.5)
        else:
            self.rot_x += speed
            self.rot_y += speed * 1.2

        self.pulse += 0.12 * self.speed_multiplier
        pulse_scale = 1 + 0.1 * math.sin(self.pulse)

        glow_layer = np.zeros_like(img)

        for x,y,z,color,size in self.points:

            # Tilt 45°
            ty = y * math.cos(self.tilt_angle) - z * math.sin(self.tilt_angle)
            tz = y * math.sin(self.tilt_angle) + z * math.cos(self.tilt_angle)

            # Rotasi X
            ry = ty * math.cos(self.rot_x) - tz * math.sin(self.rot_x)
            rz = ty * math.sin(self.rot_x) + tz * math.cos(self.rot_x)

            # Rotasi Y
            rx = x * math.cos(self.rot_y) - rz * math.sin(self.rot_y)
            rz2 = x * math.sin(self.rot_y) + rz * math.cos(self.rot_y)

            focal = 500
            scale = focal / (focal + rz2 + 500)

            sx = int(cx + rx * scale * self.scale_factor * pulse_scale)
            sy = int(cy + ry * scale * self.scale_factor * pulse_scale)

            s = max(1, int(size * scale * self.scale_factor))

            if 0 < sx < img.shape[1] and 0 < sy < img.shape[0]:
                cv2.circle(img, (sx, sy), s, color, -1)
                cv2.circle(glow_layer, (sx, sy), s*3, color, -1)

        glow_layer = cv2.GaussianBlur(glow_layer, (0,0), 15)
        img[:] = cv2.addWeighted(img, 1.0, glow_layer, 0.6, 0)

# ================= STATE =================
state = "IDLE"

red_active = False
blue_active = False

red_pos = None
blue_pos = None
purple_pos = None

merge_angle = 0
merge_radius = 0
merge_center = None

explosion_scale = 1.0

red_sphere = EnergySphere(RED_CONFIG)
blue_sphere = EnergySphere(BLUE_CONFIG)
purple_sphere = EnergySphere(PURPLE_CONFIG)

# ================= HELPER =================
def finger_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

# ================= MAIN LOOP =================
while True:

    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img,1)
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    two_fingers = False
    fist = False
    two_pos = None

    if results.multi_hand_landmarks and results.multi_handedness:

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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

            if label == "Right" and only_index and state == "IDLE":
                red_active = True
                red_pos = (x,y)

            if label == "Left" and only_index and state == "IDLE":
                blue_active = True
                blue_pos = (x,y)

            if two_fingers:
                two_pos = (x,y)

    if red_active and red_pos and state == "IDLE":
        red_sphere.draw(img, red_pos, state)

    if blue_active and blue_pos and state == "IDLE":
        blue_sphere.draw(img, blue_pos, state)

    if red_active and blue_active and state == "IDLE":
        dist = math.hypot(red_pos[0]-blue_pos[0], red_pos[1]-blue_pos[1])
        if dist < 150:
            state = "MERGING"
            merge_center = ((red_pos[0]+blue_pos[0])//2,
                            (red_pos[1]+blue_pos[1])//2)
            merge_radius = dist//2

    if state == "MERGING":

        red_sphere.draw(img, merge_center, state)
        blue_sphere.draw(img, merge_center, state)

        merge_radius *= 0.9

        if merge_radius < 20:
            state = "PURPLE"
            purple_pos = merge_center
            red_active = False
            blue_active = False

    if state == "PURPLE":

        if two_pos:
            purple_pos = two_pos

        purple_sphere.draw(img, purple_pos, state)

        if fist:
            state = "EXPLOSION"
            explosion_scale = 1.0

    if state == "EXPLOSION":

        explosion_scale += 0.3
        purple_sphere.scale_factor = explosion_scale
        purple_sphere.draw(img, purple_pos, state)

        if explosion_scale > 6:
            state = "IDLE"
            red_active = False
            blue_active = False
            purple_pos = None

    cv2.imshow("ULTIMATE CHAOS ENERGY", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("ULTIMATE CHAOS ENERGY", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()