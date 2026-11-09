from flask import Flask, Response
import cv2
import easyocr
import numpy as np
import requests
import time
import base64
import threading
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
CAMERA_INDEX_1 = 0
CAMERA_INDEX_2 = 1
PROCESS_EVERY_N_FRAMES = 30     # OCR every N frames
OCR_SCALE = 0.7                 # Slight upscale for better detection
CONFIDENCE_THRESHOLD = 0.4
COOLDOWN = 1.0
API_STORE_URL = "http://localhost:8000/api/monitoring/store"
BBOX_EXPAND = 10                # Expand bounding box to capture full plate
# --------------------------------------

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Flask app
app = Flask(__name__)

# Thread-safe OCR
ocr_lock = threading.Lock()

# ---------------- CAMERA SETUP ----------------
caps = {
    "CAM_1": cv2.VideoCapture(CAMERA_INDEX_1),
    "CAM_2": cv2.VideoCapture(CAMERA_INDEX_2)
}

for cap in caps.values():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

# Store OCR results for each camera
recent_plates = {"CAM_1": {}, "CAM_2": {}}
last_detections = {"CAM_1": [], "CAM_2": []}
frame_counters = {"CAM_1": 0, "CAM_2": 0}

# ---------------- API CALL ----------------
def send_to_api(payload):
    try:
        response = requests.post(API_STORE_URL, json=payload, timeout=5)
        # Print Laravel response
        if response.status_code == 200:
            try:
                data = response.json()  # decode JSON
                print("[API] Success:", data)
            except Exception:
                print("[API] Success, but invalid JSON:", response.text)
        else:
            print(f"[API] Failed {response.status_code}: {response.text}")
    except Exception as e:
        print("[API] Error:", e)

# ---------------- OCR & FRAME GENERATOR ----------------
def generate_frames(camera_id):
    global recent_plates, last_detections, frame_counters
    cap = caps[camera_id]

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_counters[camera_id] += 1
        display_frame = frame.copy()

        # OCR every N frames
        if frame_counters[camera_id] % PROCESS_EVERY_N_FRAMES == 0:
            # Resize and grayscale
            small = cv2.resize(frame, None, fx=OCR_SCALE, fy=OCR_SCALE)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # Enhance contrast for small plates
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

            try:
                with ocr_lock:
                    results = reader.readtext(gray)
            except:
                results = []

            last_detections[camera_id] = []

            # Only capture full plate
            for (bbox, text, prob) in results:
                if prob < CONFIDENCE_THRESHOLD:
                    continue

                # Cooldown
                now = time.time()
                if text in recent_plates[camera_id] and now - recent_plates[camera_id][text] < COOLDOWN:
                    continue
                recent_plates[camera_id][text] = now

                # Scale bbox back to original frame size
                bbox = np.array(bbox) / OCR_SCALE
                bbox = bbox.astype(int)

                # Expand bbox to ensure full plate
                x_min, y_min = bbox.min(axis=0)
                x_max, y_max = bbox.max(axis=0)
                x_min = max(0, x_min - BBOX_EXPAND)
                y_min = max(0, y_min - BBOX_EXPAND)
                x_max = min(frame.shape[1], x_max + BBOX_EXPAND)
                y_max = min(frame.shape[0], y_max + BBOX_EXPAND)

                plate_img = frame[y_min:y_max, x_min:x_max]
                if plate_img.size == 0:
                    continue

                # Encode whole plate image
                _, buffer = cv2.imencode('.jpg', plate_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                plate_base64 = base64.b64encode(buffer).decode()

                payload = {"text": text, "plate_image": plate_base64}
                threading.Thread(target=lambda: send_to_api(payload), daemon=True).start()

                # Save detection to draw once per plate
                last_detections[camera_id].append((bbox, text))

        # Draw detections on frame
        for bbox, text in last_detections[camera_id]:
            cv2.polylines(display_frame, [bbox], True, (0, 255, 0), 2)
            x_min, y_min = bbox.min(axis=0)
            cv2.putText(display_frame, text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---------------- FLASK ROUTES ----------------
@app.route('/video/cam1')
def video_cam1():
    return Response(generate_frames("CAM_1"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/cam2')
def video_cam2():
    return Response(generate_frames("CAM_2"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- START ----------------
if __name__ == "__main__":
    print("🚀 Flask OCR server running")
    app.run(host="0.0.0.0", port=5000, threaded=True)
