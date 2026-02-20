import cv2
import mediapipe as mp
import math
import random
import time


# ==============================
# HAND TRACKER
# ==============================
class HandTracker:
    def __init__(self, max_hands=2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_hands)
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw(self, frame, hand_landmarks):
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

    def finger_up(self, landmarks, tip_id, pip_id):
        return landmarks.landmark[tip_id].y < landmarks.landmark[pip_id].y

    def close(self):
        self.hands.close()


# ==============================
# PARTICLE SYSTEM
# ==============================
class ParticleSystem:
    def __init__(self):
        self.particles = []

    def spawn_edge(self, center, radius, color, amount, speed_range):
        for _ in range(amount):
            angle = random.uniform(0, 2 * math.pi)
            x = center[0] + math.cos(angle) * radius
            y = center[1] + math.sin(angle) * radius
            speed = random.uniform(*speed_range)

            self.particles.append({
                "x": x,
                "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": random.randint(15, 30),
                "color": color
            })

    def update(self, frame):
        for p in self.particles[:]:
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
                cv2.circle(frame, (int(p["x"]), int(p["y"])), 2, color, -1)
            else:
                self.particles.remove(p)


# ==============================
# ENERGY SYSTEM
# ==============================
class EnergySystem:
    def __init__(self):
        self.purple_active = False
        self.purple_exploding = False
        self.purple_pos = None

        self.charge_radius = 0
        self.explosion_radius = 0

        self.max_charge = 120
        self.max_explosion = 600

    def try_merge(self, red_pos, blue_pos):
        if not self.purple_active and red_pos and blue_pos:
            distance = math.hypot(
                red_pos[0] - blue_pos[0],
                red_pos[1] - blue_pos[1]
            )
            if distance < 100:
                self.purple_active = True
                self.purple_pos = (
                    (red_pos[0] + blue_pos[0]) // 2,
                    (red_pos[1] + blue_pos[1]) // 2
                )
                self.charge_radius = 0

    def update(self, frame, particles, pulse, two_finger_pos, fist):
        if not self.purple_active:
            return

        # Trigger explosion
        if fist and not self.purple_exploding:
            self.purple_exploding = True
            self.explosion_radius = self.charge_radius

        # NORMAL MODE
        if not self.purple_exploding:
            if two_finger_pos:
                self.purple_pos = two_finger_pos

            if self.charge_radius < self.max_charge:
                self.charge_radius += 4

            radius = self.charge_radius + pulse

            overlay = frame.copy()
            for i in range(4):
                cv2.circle(overlay, self.purple_pos,
                           radius + i*25, (255, 0, 255), -1)

            frame[:] = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
            cv2.circle(frame, self.purple_pos, radius, (255, 0, 255), -1)

            particles.spawn_edge(
                self.purple_pos, radius, (255, 0, 255), 6, (2, 4)
            )

        # EXPLOSION MODE
        else:
            self.explosion_radius += 50

            overlay = frame.copy()
            for i in range(5):
                cv2.circle(overlay, self.purple_pos,
                           self.explosion_radius + i*40,
                           (255, 0, 255), -1)

            frame[:] = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

            particles.spawn_edge(
                self.purple_pos,
                self.explosion_radius,
                (255, 0, 255),
                20,
                (4, 8)
            )

            if self.explosion_radius > self.max_explosion:
                self.reset()

    def reset(self):
        self.purple_active = False
        self.purple_exploding = False
        self.purple_pos = None
        self.charge_radius = 0
        self.explosion_radius = 0


# ==============================
# MAIN APP
# ==============================
class HollowPurpleApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.tracker = HandTracker()
        self.particles = ParticleSystem()
        self.energy = EnergySystem()

    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            results = self.tracker.process(frame)

            red_pos = None
            blue_pos = None
            two_finger_pos = None
            fist = False

            pulse = int(10 * (math.sin(time.time() * 5) + 1))

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                    self.tracker.draw(frame, hand_landmarks)
                    label = results.multi_handedness[idx].classification[0].label

                    index_up = self.tracker.finger_up(hand_landmarks, 8, 6)
                    middle_up = self.tracker.finger_up(hand_landmarks, 12, 10)
                    ring_up = self.tracker.finger_up(hand_landmarks, 16, 14)
                    pinky_up = self.tracker.finger_up(hand_landmarks, 20, 18)

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

                    if label == "Right" and only_index and not self.energy.purple_active:
                        red_pos = (x, y)

                    if label == "Left" and only_index and not self.energy.purple_active:
                        blue_pos = (x, y)

                    if two_fingers:
                        two_finger_pos = (x, y)

            # Merge logic
            self.energy.try_merge(red_pos, blue_pos)

            # Update energy system
            self.energy.update(frame, self.particles, pulse,
                               two_finger_pos, fist)

            # Update particles
            self.particles.update(frame)

            cv2.imshow("Hollow Purple OOP", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.tracker.close()
        cv2.destroyAllWindows()


# ==============================
# RUN PROGRAM
# ==============================
if __name__ == "__main__":
    app = HollowPurpleApp()
    app.run()