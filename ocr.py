from flask import Flask, Response
import cv2
import easyocr
import threading
from queue import Queue
import time
import json
import requests

# ------------------- CONFIG ---------------------
API_URL = "http://localhost:8000/api/monitoring/store"  
VIDEO_SOURCE = 0
SCALE_FACTOR = 0.5  # was 0.5
COOLDOWN_TIME = 1  # avoid duplicate sends within seconds
latest_plate = ""

app = Flask(__name__)

reader = easyocr.Reader(['en'])
api_queue = Queue()
recent_texts = {}  # cooldown memory

cap = cv2.VideoCapture(VIDEO_SOURCE)

# ------------------- API WORKER ---------------------
def api_worker():
    session = requests.Session()
    while True:
        text = api_queue.get()
        if text is None:
            break

        try:
            payload = {"text": text}
            headers = {"Accept": "application/json"}

            res = session.post(API_URL, json=payload, headers=headers)
            print("\n📡 API Sent:", text)
            print("Status:", res.status_code)
            print("Response:", res.text)

        except Exception as e:
            print("❌ API Error:", e)

        api_queue.task_done()

api_thread = threading.Thread(target=api_worker, daemon=True)
api_thread.start()

# ------------------- CAMERA STREAM ---------------------
def generate_frames():
    global latest_plate
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        try:
            results = reader.readtext(small_frame)
        except:
            results = []

        current_time = time.time()

        for (bbox, text, prob) in results:
            if prob < 0.5:
                continue

            (tl, tr, br, bl) = bbox
            tl = tuple(int(x / SCALE_FACTOR) for x in tl)
            br = tuple(int(x / SCALE_FACTOR) for x in br)

            # Draw box & text
            # --------- PLATE INFO WINDOW ----------
            overlay_x, overlay_y = 10, 10
            overlay_w, overlay_h = 620, 70

            # Background box
            cv2.rectangle(frame,
                        (overlay_x, overlay_y),
                        (overlay_x + overlay_w, overlay_y + overlay_h),
                        (0, 0, 0),
                        -1)

            # Border
            cv2.rectangle(frame,
                        (overlay_x, overlay_y),
                        (overlay_x + overlay_w, overlay_y + overlay_h),
                        (0, 255, 0),
                        2)

            # Title
            cv2.putText(frame, "Detected Plate:",
                        (overlay_x + 10, overlay_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            # Plate text
            cv2.putText(frame, latest_plate if latest_plate else "None",
                        (overlay_x + 10, overlay_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)

            # Cooldown check
            if text not in recent_texts or (current_time - recent_texts[text] > COOLDOWN_TIME):
                recent_texts[text] = current_time
                latest_plate = text        # ✅ STORE LATEST PLATE
                api_queue.put(text)
                print(f"📤 OCR Detected & Queued: {text}")

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------- START SERVER ---------------------
if __name__ == "__main__":
    print("🚀 Flask Camera Server Running at http://localhost:5000/video")
    app.run(host="0.0.0.0", port=5000)
