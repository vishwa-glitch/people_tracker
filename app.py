import os
from flask import Flask, render_template, Response, request
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)


class PeopleTracker:
    def __init__(self):
        self.total_count = 0
        self.happy_count = 0
        self.not_happy_count = 0

    def detect_faces_and_emotions(self, frame):
        # Detect faces using MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        self.total_count = 0
        self.happy_count = 0
        self.not_happy_count = 0

        # Initialize a thread pool executor for emotion detection
        with ThreadPoolExecutor() as executor:
            future_to_face = {}

            if results.detections:
                for detection in results.detections:
                    # Get face bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)

                    # Extract face ROI
                    face_roi = frame[y:y + height, x:x + width]
                    if face_roi.size == 0:
                        continue

                    self.total_count += 1

                    # Submit the emotion analysis task to the thread pool
                    result_dict = {}
                    future = executor.submit(self.analyze_emotion_async, face_roi, result_dict)
                    future_to_face[future] = (x, y, width, height, result_dict)

            # Now retrieve results and add to the frame
            for future in future_to_face:
                x, y, width, height, result_dict = future_to_face[future]
                future.result()  # Wait for the result
                emotion = result_dict['emotion']
                if emotion == 'happy':
                    self.happy_count += 1
                    color = (0, 255, 0)  # Green for happy
                else:
                    self.not_happy_count += 1
                    color = (0, 0, 255)  # Red for not happy

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                # Add emotion label
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Add counts to frame
        cv2.putText(frame, f"Total: {self.total_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Happy: {self.happy_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Not Happy: {self.not_happy_count}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def analyze_emotion_async(self, face_roi, result_dict):
        try:
            # Analyze emotion using DeepFace in the background
            emotion = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            result_dict['emotion'] = emotion[0]['dominant_emotion']
        except Exception as e:
            print(f"Error analyzing emotion: {str(e)}")
            result_dict['emotion'] = None

tracker = PeopleTracker()

# This route will just render the main page with the video stream
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    data = request.data  # You will receive video frames from the client here
    # Convert the incoming frame to an image (e.g., using numpy)
    frame = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Process the frame and detect faces and emotions
    processed_frame = tracker.detect_faces_and_emotions(frame)

    # Convert to jpg format
    ret, buffer = cv2.imencode('.jpg', processed_frame)
    frame = buffer.tobytes()

    return Response(frame, mimetype='image/jpeg')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Ensure it uses the correct port for Render
    app.run(host="0.0.0.0", port=port)


