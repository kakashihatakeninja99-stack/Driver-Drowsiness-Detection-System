import sys
import time

print("Script started", flush=True)

import cv2
import numpy as np
print("cv2 and numpy imported", flush=True)

import tensorflow as tf
import requests
print("tensorflow and requests imported", flush=True)

# ----------------------------
# CONFIG
# ----------------------------
API_URL = "http://localhost:5000/api"
LOGIN_EMAIL = "test@example.com"
LOGIN_PASSWORD = "123456"

MODEL_PATH = "drowsy.keras"
IMG_SIZE = 224

CLASS_NAMES = ['Closed', 'Open', 'no_yawn', 'yawn']
DROWSY_CLASSES = ['Closed', 'yawn']

ALERT_COOLDOWN = 5
CONSEC_FRAMES_NEEDED = 3

# ----------------------------
# Face & Eye detectors (built into OpenCV)
# ----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

try:
    print("Loading model...", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!", flush=True)

    # ----------------------------
    # Login
    # ----------------------------
    print("Attempting login...", flush=True)
    response = requests.post(f"{API_URL}/auth/login", json={
        "email": LOGIN_EMAIL,
        "password": LOGIN_PASSWORD
    })
    if response.status_code == 200:
        token = response.json()["token"]
        print("Logged in successfully.", flush=True)
    else:
        print("Login failed:", response.json(), flush=True)
        token = None

    if not token:
        print("Exiting - no token.", flush=True)
        sys.exit(1)

    # ----------------------------
    # Webcam
    # ----------------------------
    print("Opening webcam...", flush=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.", flush=True)
        sys.exit(1)

    print("Webcam opened. Click the video window and press 'q' to quit.", flush=True)

    last_alert_time = 0
    consecutive_drowsy = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.", flush=True)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        label_text = "No face detected"
        color = (255, 255, 0)

        if len(faces) > 0:
            # Sab se bara face lo (closest to camera)
            (fx, fy, fw, fh) = max(faces, key=lambda r: r[2] * r[3])
            face_gray = gray[fy:fy+fh, fx:fx+fw]
            face_color = frame[fy:fy+fh, fx:fx+fw]

            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

            if len(eyes) > 0:
                # Sab se bari eye detection lo
                (ex, ey, ew, eh) = max(eyes, key=lambda r: r[2] * r[3])
                eye_crop = face_color[ey:ey+eh, ex:ex+ew]

                # Preprocess eye crop for model
                processed = cv2.resize(eye_crop, (IMG_SIZE, IMG_SIZE))
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                processed = processed.astype("float32")
                input_tensor = np.expand_dims(processed, axis=0)

                prediction = model.predict(input_tensor, verbose=0)[0]
                class_idx = int(np.argmax(prediction))
                class_name = CLASS_NAMES[class_idx]
                confidence = float(prediction[class_idx])

                print(f"Prediction: {class_name} ({confidence:.2f})", flush=True)

                if class_name in DROWSY_CLASSES:
                    consecutive_drowsy += 1
                    label_text = f"DROWSY: {class_name} ({confidence:.2f})"
                    color = (0, 0, 255)
                else:
                    consecutive_drowsy = 0
                    label_text = f"ALERT: {class_name} ({confidence:.2f})"
                    color = (0, 255, 0)

                if consecutive_drowsy >= CONSEC_FRAMES_NEEDED:
                    current_time = time.time()
                    if current_time - last_alert_time > ALERT_COOLDOWN:
                        try:
                            r = requests.post(f"{API_URL}/detections",
                                json={"eventType": "drowsy", "confidence": confidence},
                                headers={"Authorization": f"Bearer {token}"})
                            print("Alert sent:", r.json(), flush=True)
                        except Exception as e:
                            print("Error sending alert:", e, flush=True)
                        last_alert_time = current_time

                # Draw rectangle around detected eye (debug visual)
                cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), color, 2)
            else:
                label_text = "Face found, no eyes detected"
                consecutive_drowsy = 0

            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
        else:
            consecutive_drowsy = 0

        cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit key pressed. Stopping...", flush=True)
            break

    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("FATAL ERROR:", e, flush=True)
    import traceback
    traceback.print_exc()

print("Script ended.", flush=True)
input("Press Enter to close...")