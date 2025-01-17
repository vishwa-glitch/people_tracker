import cv2
import os
from flask import Flask, render_template, Response
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Initialize MediaPipe for face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# Initialize camera
camera = cv2.VideoCapture(0)  # 0 for default camera

class PeopleTracker:
    def __init__(self):
        self.total_count = 0
        self.happy_count = 0
        self.not_happy_count = 0

    def detect_faces_and_emotions(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        self.total_count = 0
        self.happy_count = 0
        self.not_happy_count = 0

        with ThreadPoolExecutor() as executor:
            future_to_face = {}

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)

                    face_roi = frame[y:y + height, x:x + width]
                    if face_roi.size == 0:
                        continue

                    self.total_count += 1

                    result_dict = {}
                    future = executor.submit(self.analyze_emotion_async, face_roi, result_dict)
                    future_to_face[future] = (x, y, width, height, result_dict)

            for future in future_to_face:
                x, y, width, height, result_dict = future_to_face[future]
                future.result()
                emotion = result_dict['emotion']

                if emotion == 'happy':
                    self.happy_count += 1
                    color = (0, 255, 0)
                else:
                    self.not_happy_count += 1
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(frame, f"Total: {self.total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Happy: {self.happy_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Not Happy: {self.not_happy_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def analyze_emotion_async(self, face_roi, result_dict):
        try:
            emotion = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            result_dict['emotion'] = emotion[0]['dominant_emotion']
        except Exception as e:
            print(f"Error analyzing emotion: {str(e)}")
            result_dict['emotion'] = "Unknown"

tracker = PeopleTracker()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        processed_frame = tracker.detect_faces_and_emotions(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def counts():
    return {
        'total': tracker.total_count,
        'happy': tracker.happy_count,
        'not_happy': tracker.not_happy_count
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



