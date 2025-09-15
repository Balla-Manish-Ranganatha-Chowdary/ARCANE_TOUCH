# Arcane Touch

Arcane Touch is an interactive computer vision project that transforms hand gestures into magical effects. Using MediaPipe for fingertip tracking and OpenCV for real time video processing, the project lets users “erase” their webcam background by waving their index finger. As the background fades, glowing particles and dynamic lightning bolts trail behind, creating a surreal augmented reality experience.

---

## 🚀 Features
- **Hand Tracking** – Uses [MediaPipe Hands](https://google.github.io/mediapipe/) to detect your index fingertip in real-time.  
- **Background Eraser** – Captures your static background and reveals it as you “wipe” the air.  
- **Magic Effects** – Lightning bolts ⚡ and colorful particles ✨ trail your finger as you erase.  
- **Gesture Toggle** – Start/stop magic mode by hovering over the on-screen button.  
- **Hotkeys**:
  - `Q` or `Esc` → Quit the app  
  - `C` → Clear effects and reset  
  - `R` → Re-capture background  

---

## 🛠️ Requirements
Install dependencies before running:

```bash
pip install opencv-python mediapipe numpy
```

---

## ▶️ Usage
Run the program:

```bash
python main.py
```
Steps:

1. When prompted, step out of the frame to let the app capture your background.
2. Hover your index finger over the glowing START MAGIC button.
3. Wave your hand to erase and draw sparks!
4. Press [Q] to quit when done.

---

## 🧩 Project Structure
```markdown
Arcane-Touch/
│── main.py          # Main program
│── README.md        # Project description

```

---
## ✨ Ideas for Future Improvements
• Multi-hand support
• Customizable particle colors
• Sound effects with each spark
• Export as a video filter

---
