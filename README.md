# People Tracker with Expression Analysis

A real-time people tracking and expression analysis application built with Python, Flask, and deep learning. This application detects faces in a video stream and analyzes their emotions, providing live counts of happy and unhappy people.

## 🌟 Features

- Real-time face detection using MediaPipe
- Emotion analysis using DeepFace
- Live counting of total people, happy, and unhappy faces
- Visual indicators with color-coded bounding boxes
- Web-based interface for easy access
- Responsive design that works on different screen sizes

## 🛠️ Technologies Used

- Python 3.8+
- Flask
- OpenCV
- TensorFlow
- DeepFace
- MediaPipe
- HTML/CSS

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam or video input device
- Modern web browser
- Required Python packages (see requirements.txt)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/vishwa-glitch/people-tracker.git
cd people-tracker
```

2. Create and activate virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. Allow camera access when prompted by your browser

## 📊 Features in Detail

- **Face Detection**: Uses MediaPipe's face detection model for accurate and fast face detection
- **Emotion Analysis**: Implements DeepFace for real-time emotion recognition
- **Live Counting**: Tracks the number of people and their emotional states in real-time
- **Visual Feedback**: 
  - Green boxes: Happy expressions
  - Red boxes: Not happy expressions
  - On-screen counters for total, happy, and unhappy faces

## 🔧 Configuration

The application can be configured by modifying the following parameters in `app.py`:

- Face detection confidence threshold
- Emotion analysis settings
- Video capture parameters

## 🚀 Deployment

The application can be deployed on various platforms:

1. **Hugging Face**
   - Follow Hugging Face Spaces deployment guidelines
   - Include requirements.txt and Dockerfile

2. **Render**
   - Connect to GitHub repository
   - Set environment variables
   - Choose Python environment

3. **Google Colab**
   - Modify code for Colab environment
   - Use ngrok for tunneling

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Future Improvements

- Add support for multiple camera inputs
- Implement emotion tracking over time
- Add data visualization for emotion trends
- Improve performance optimization
- Add support for recorded video analysis

## ⚠️ Note

This application requires camera access and sufficient computational resources for real-time processing. Performance may vary based on hardware capabilities.

## 👥 Author

[Your Name]
- GitHub: [@vishwa-glitch](https://github.com/vishwa-glitch)
- LinkedIn: [LinkedIn](https://linkedin.com/in/vishwa55)

## 🙏 Acknowledgments

- MediaPipe team for their face detection model
- DeepFace library contributors
- Flask framework community
