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
cap.set(3, 640)
cap.set(4, 480)

BALL_COUNT = 500

# ================= CONFIG =================
class EnergyConfig:
    def __init__(self, core_color, arm_color):
        self.coreRadius = 50
        self.coreRatio = 0.5
        self.coreColor = core_color
        self.coreSize = 4
        self.armColor = arm_color
        self.armSize = 2
        self.arms = 4

RED_CONFIG = EnergyConfig((0,0,255),(80,80,255))
BLUE_CONFIG = EnergyConfig((255,0,0),(255,150,150))
PURPLE_CONFIG = EnergyConfig((255,0,255),(255,150,255))

# ================= ENERGY SPHERE =================
class EnergySphere:

    def __init__(self, config):
        self.config = config
        self.points = []
        self.rot_y = 0
        self.speed = 0.25  # 🔥 langsung cepat stabil
        self.generate()

    def generate(self):
        C = self.config
        self.points.clear()

        for i in range(BALL_COUNT):

            if i < BALL_COUNT * C.coreRatio:
                r = random.random()*C.coreRadius
                th = random.random()*2*math.pi
                ph = math.acos(2*random.random()-1)

                x = r*math.sin(ph)*math.cos(th)
                y = r*math.sin(ph)*math.sin(th)
                z = r*math.cos(ph)

                self.points.append({
                    "type":"core",
                    "x":x,"y":y,"z":z,
                    "color":C.coreColor,
                    "size":C.coreSize
                })

            else:
                arm_index = i % C.arms
                base_angle = (2*math.pi/C.arms)*arm_index
                radius = random.uniform(60,160)
                angle = base_angle + radius*0.04
                height = (random.random()-0.5)*30

                self.points.append({
                    "type":"orbit",
                    "radius":radius,
                    "angle":angle,
                    "z":height,
                    "color":C.armColor,
                    "size":C.armSize
                })

    def draw(self, img, center, speed_multiplier=1):

        cx,cy=center
        self.rot_y += self.speed * speed_multiplier  # 🔥 hanya Y

        for p in self.points:

            if p["type"]=="core":
                x,y,z = p["x"],p["y"],p["z"]
            else:
                p["angle"] += 0.1 * speed_multiplier
                x = p["radius"]*math.cos(p["angle"])
                y = p["radius"]*math.sin(p["angle"])
                z = p["z"]

            # 🔥 ROTASI HANYA SUMBU Y (horizontal)
            rx = x*math.cos(self.rot_y) - z*math.sin(self.rot_y)
            rz = x*math.sin(self.rot_y) + z*math.cos(self.rot_y)

            focal=400
            scale=focal/(focal+rz+400)

            sx=int(cx + rx*scale)
            sy=int(cy + y*scale)
            s=max(1,int(p["size"]*scale))

            if 0<sx<img.shape[1] and 0<sy<img.shape[0]:
                cv2.circle(img,(sx,sy),s,p["color"],-1)

# ================= EXPLOSION =================
class Explosion:
    def __init__(self, center):
        self.particles=[]
        for _ in range(300):
            angle=random.random()*2*math.pi
            speed=random.uniform(4,10)
            self.particles.append({
                "x":center[0],
                "y":center[1],
                "vx":math.cos(angle)*speed,
                "vy":math.sin(angle)*speed,
                "life":random.randint(30,60)
            })

    def update_draw(self,img):
        for p in self.particles:
            p["x"]+=p["vx"]
            p["y"]+=p["vy"]
            p["life"]-=1
            if p["life"]>0:
                cv2.circle(img,(int(p["x"]),int(p["y"])),2,(255,0,255),-1)

# ================= STATE =================
state="IDLE"
red_active=False
blue_active=False
red_pos=None
blue_pos=None
purple_pos=None
merge_angle=0
explosion=None

red=EnergySphere(RED_CONFIG)
blue=EnergySphere(BLUE_CONFIG)
purple=EnergySphere(PURPLE_CONFIG)

# ================= MAIN LOOP =================
while True:

    success,img=cap.read()
    if not success: break
    img=cv2.flip(img,1)
    h,w,_=img.shape

    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(rgb)

    fist=False

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx,hand_landmarks in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            label=results.multi_handedness[idx].classification[0].label

            def up(tip,pip):
                return hand_landmarks.landmark[tip].y<hand_landmarks.landmark[pip].y

            index=up(8,6)
            middle=up(12,10)
            ring=up(16,14)
            pinky=up(20,18)
            fist=not index and not middle and not ring and not pinky

            x=int(hand_landmarks.landmark[8].x*w)
            y=int(hand_landmarks.landmark[8].y*h)

            if label=="Right" and index and state=="IDLE":
                red_active=True
                red_pos=(x,y)

            if label=="Left" and index and state=="IDLE":
                blue_active=True
                blue_pos=(x,y)

    # ===== DRAW =====
    if red_active and state=="IDLE":
        red.draw(img,red_pos)

    if blue_active and state=="IDLE":
        blue.draw(img,blue_pos)

    # ===== MERGE ORBIT =====
    if red_active and blue_active and state=="IDLE":
        if math.hypot(red_pos[0]-blue_pos[0],red_pos[1]-blue_pos[1])<120:
            state="MERGING"
            purple_pos=((red_pos[0]+blue_pos[0])//2,
                        (red_pos[1]+blue_pos[1])//2)

    if state=="MERGING":
        merge_angle+=0.3
        r=60
        rx=int(purple_pos[0]+r*math.cos(merge_angle))
        ry=int(purple_pos[1]+r*math.sin(merge_angle))
        bx=int(purple_pos[0]+r*math.cos(merge_angle+math.pi))
        by=int(purple_pos[1]+r*math.sin(merge_angle+math.pi))

        red.draw(img,(rx,ry),1.5)
        blue.draw(img,(bx,by),1.5)

        if merge_angle>6:
            state="PURPLE"

    if state=="PURPLE":
        purple.draw(img,purple_pos,1.5)
        if fist:
            explosion=Explosion(purple_pos)
            state="EXPLOSION"

    if state=="EXPLOSION":
        explosion.update_draw(img)
        if all(p["life"]<=0 for p in explosion.particles):
            state="IDLE"
            red_active=False
            blue_active=False

    cv2.imshow("GALAXY HORIZONTAL FIXED",img)

    if cv2.waitKey(1)&0xFF==ord('q'): break
    if cv2.getWindowProperty("GALAXY HORIZONTAL FIXED",cv2.WND_PROP_VISIBLE)<1:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()