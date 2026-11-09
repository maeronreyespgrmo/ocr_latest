from flask import Flask, Response
import cv2
import easyocr
import threading
from queue import Queue
import time
import requests

# ------------------- CONFIG ---------------------
API_URL = "http://localhost:8000/api/monitoring/store"
SCALE_FACTOR = 1
COOLDOWN_TIME = 1
OCR_EVERY_N_FRAMES = 3  # 🔥 KEY SPEED BOOST

# ------------------- FLASK ----------------------
app = Flask(__name__)

# ------------------- OCR ------------------------
reader = easyocr.Reader(['en'], gpu=True)

# ------------------- API QUEUE ------------------
api_queue = Queue()
recent_texts = {}

# ------------------- CAMERAS --------------------
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

for cap in (cap1, cap2):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

# ------------------- API WORKER -----------------
def api_worker():
    session = requests.Session()
    while True:
        data = api_queue.get()
        if data is None:
            break

        try:
            session.post(API_URL, json=data)
            print(f"📡 [{data['camera_id']}] {data['text']}")
        except:
            pass

        api_queue.task_done()

threading.Thread(target=api_worker, daemon=True).start()

# ------------------- FRAME GENERATOR -------------
def generate_frames(cap, camera_id):
    latest_plate = ""
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # OCR ONLY EVERY N FRAMES
        if frame_count % OCR_EVERY_N_FRAMES == 0:
            small = cv2.resize(frame, None,
                               fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            try:
                results = reader.readtext(gray)
            except:
                results = []

            now = time.time()

            for (_, text, prob) in results:
                if prob < 0.5:
                    continue

                key = (camera_id, text)
                if key not in recent_texts or now - recent_texts[key] > COOLDOWN_TIME:
                    recent_texts[key] = now
                    latest_plate = text
                    api_queue.put({
                        "camera_id": camera_id,
                        "text": text
                    })

        # ---------- OVERLAY ----------
        cv2.rectangle(frame, (10, 10), (620, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (620, 80), (0, 255, 0), 2)

        cv2.putText(frame, f"{camera_id}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Plate: {latest_plate or 'None'}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

# ------------------- ROUTES ----------------------
@app.route('/video/cam1')
def video_cam1():
    return Response(generate_frames(cap1, "CAM_1"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/cam2')
def video_cam2():
    return Response(generate_frames(cap2, "CAM_2"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------- START -----------------------
if __name__ == "__main__":
    print("🚀 Optimized server running")
    app.run(host="0.0.0.0", port=5000, threaded=True)
