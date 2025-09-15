import cv2
import numpy as np
import mediapipe as mp
import time, random, math, sys

# ==========================================================
# Settings & Constants
# ==========================================================
NUM_BG_FRAMES = 35         # Frames used to estimate background
BG_WAIT_TIME = 3           # Seconds to wait before background capture
BRUSH_RADIUS = 80          # Eraser size
FEATHER_SIZE = 61          # Blur kernel for smooth blending
SMOOTHING_FACTOR = 0.75    # Hand position smoothing
LIGHTNING_LIMIT = 3        # Max lightning bolts per frame
PARTICLES_PER_FRAME = 6    # New particles spawned per move
PARTICLE_LIFETIME = 20     # Frames before a particle dies

mp_hands = mp.solutions.hands

# ==========================================================
# Camera Setup
# ==========================================================
def open_webcam(attempts=5):
    """Try to open the webcam by testing multiple indices."""
    for idx in range(attempts):
        cam = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cam.isOpened():
            success, frame = cam.read()
            if success:
                print(f"[OK] Camera {idx} started ({frame.shape[1]}x{frame.shape[0]})")
                return cam
            cam.release()
    return None


# ==========================================================
# Background Capture
# ==========================================================
def record_background(cam, frames=NUM_BG_FRAMES, delay=BG_WAIT_TIME):
    """Capture the static background when user steps aside."""
    print("[INFO] Please move away, background capture starting...")
    timeout = time.time() + delay

    # Countdown preview
    while time.time() < timeout:
        ret, shot = cam.read()
        if not ret: continue
        shot = cv2.flip(shot, 1)
        preview = shot.copy()
        countdown = int(math.ceil(timeout - time.time()))
        cv2.putText(preview, f"Capturing in {countdown}...", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Setup", preview)
        if cv2.waitKey(1) == 27:
            break

    # Collect background frames
    stack = []
    for i in range(frames):
        ret, shot = cam.read()
        if not ret: continue
        shot = cv2.flip(shot, 1)
        stack.append(shot.astype(np.float32))
        median_view = (np.median(stack, axis=0)).astype(np.uint8)
        cv2.putText(median_view, f"Capturing background {i+1}/{frames}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Setup", median_view)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyWindow("Setup")
    if not stack:
        raise RuntimeError("No background captured.")
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


# ==========================================================
# Visual Effects (Particles & Lightning)
# ==========================================================
def create_particles(particles, x, y, amount=8):
    """Spawn glowing particles at a given coordinate."""
    for _ in range(amount):
        vx, vy = random.uniform(-1.5, 1.5), random.uniform(-3.0, -0.5)
        size = random.randint(2, 6)
        life = random.randint(int(PARTICLE_LIFETIME*0.6), PARTICLE_LIFETIME)
        color = random.choice([(255,255,0),(200,200,255),(255,120,255)])
        particles.append({
            'pos': [x,y], 'vel': [vx,vy],
            'life': life, 'size': size, 'color': color
        })

def animate_particles(frame, particles):
    """Move particles each frame and draw them."""
    new_list = []
    for p in particles:
        if p['life'] <= 0: continue
        p['pos'][0] += p['vel'][0]
        p['pos'][1] += p['vel'][1]
        p['vel'][1] += 0.12  # gravity effect
        x, y = int(p['pos'][0]), int(p['pos'][1])
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            layer = frame.copy()
            cv2.circle(layer, (x,y), p['size']*2, p['color'], -1, cv2.LINE_AA)
            frame[:] = cv2.addWeighted(layer, 0.25, frame, 0.75, 0)
        p['life'] -= 1
        if p['life'] > 0: new_list.append(p)
    particles[:] = new_list

def lightning(frame, x, y, bolts=3):
    """Draw stylized lightning bolts from a point."""
    glow_layer = np.zeros_like(frame)
    core_layer = np.zeros_like(frame)
    for _ in range(bolts):
        length = random.randint(60, 140)
        angle = random.uniform(-math.pi, math.pi)
        ex, ey = int(x + length*math.cos(angle)), int(y + length*math.sin(angle))
        ex, ey = max(0,min(frame.shape[1]-1,ex)), max(0,min(frame.shape[0]-1,ey))

        pts = [(x,y)]
        for i in range(1, random.randint(3,6)+1):
            t = i / random.randint(3,6)
            pts.append((int(x+(ex-x)*t+random.randint(-25,25)),
                        int(y+(ey-y)*t+random.randint(-25,25))))
        pts = np.array(pts, np.int32)
        color = random.choice([(255,180,30),(255,255,0),(255,50,255)])
        cv2.polylines(glow_layer,[pts],False,color,10,cv2.LINE_AA)
        cv2.polylines(core_layer,[pts],False,(255,255,255),2,cv2.LINE_AA)

    frame[:] = cv2.addWeighted(frame, 1, cv2.GaussianBlur(glow_layer,(31,31),0), 0.6, 0)
    frame[:] = cv2.addWeighted(frame, 1, core_layer, 1, 0)


# ==========================================================
# Main Program
# ==========================================================
def main():
    cam = open_webcam()
    if not cam:
        print("[ERROR] No webcam detected.")
        sys.exit(1)

    # Warm-up frames & background
    for _ in range(10): cam.read()
    background = record_background(cam)

    h, w = background.shape[:2]
    erase_mask = np.zeros((h,w),dtype=np.uint8)
    hands = mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.6,min_tracking_confidence=0.6)

    fingertip_smoothed, particle_list = None, []
    erasing = False

    # On-screen button
    btn_w, btn_h = 240, 60
    bx1, by1 = w - btn_w - 20, 20
    bx2, by2 = bx1 + btn_w, by1 + btn_h
    pulse = 0

    while True:
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame,(w,h))

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        hand_data = hands.process(rgb)

        tip = None
        if hand_data.multi_hand_landmarks:
            lm = hand_data.multi_hand_landmarks[0].landmark
            tip_x, tip_y = int(lm[8].x*w), int(lm[8].y*h)
            tip = (tip_x, tip_y)

            # If button clicked, start effect
            if not erasing and bx1 < tip_x < bx2 and by1 < tip_y < by2:
                erasing = True
                print("[INFO] Magic mode activated.")

        # Draw animated button
        if not erasing:
            pulse += 0.15
            blend = (math.sin(pulse) + 1)/2
            base, flash = np.array([50,220,255]), np.array([255,50,200])
            col = (base*(1-blend) + flash*blend).astype(np.int32).tolist()
            cv2.rectangle(frame,(bx1,by1),(bx2,by2),col,-1,cv2.LINE_AA)
            cv2.putText(frame,"START MAGIC",(bx1+20,by1+40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),3)

        # Apply effects if active
        if erasing and tip:
            if fingertip_smoothed is None:
                fingertip_smoothed = np.array(tip, dtype=np.float32)
            else:
                fingertip_smoothed = (fingertip_smoothed*SMOOTHING_FACTOR +
                                      np.array(tip,dtype=np.float32)*(1-SMOOTHING_FACTOR))
            x, y = map(int, fingertip_smoothed)
            cv2.circle(erase_mask,(x,y),BRUSH_RADIUS,255,-1)
            create_particles(particle_list,x,y,PARTICLES_PER_FRAME)
            if random.random() < 0.45:
                lightning(frame,x,y,random.randint(1,LIGHTNING_LIMIT))
        else:
            fingertip_smoothed = None

        # Apply erase effect
        k = FEATHER_SIZE if FEATHER_SIZE % 2 == 1 else FEATHER_SIZE+1
        blur_mask = cv2.GaussianBlur(erase_mask,(k,k),0)
        alpha = blur_mask.astype(np.float32)/255.0
        alpha3 = cv2.merge([alpha,alpha,alpha])
        final = (frame.astype(np.float32)*(1-alpha3) +
                 background.astype(np.float32)*alpha3).astype(np.uint8)

        animate_particles(final, particle_list)
        cv2.imshow("Magic Eraser", final)

        key=cv2.waitKey(1)&0xFF
        if key in [ord('q'),27]: break
        elif key==ord('c'): erase_mask[:]=0; particle_list.clear(); erasing=False
        elif key==ord('r'): background=record_background(cam); erase_mask[:]=0; erasing=False

    cam.release(); hands.close(); cv2.destroyAllWindows()


if __name__=="__main__":
    main()
