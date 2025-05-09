# Sign Language Recognition System

## 🎯 Overview

A real-time sign language recognition system that converts hand gestures to text using computer vision and deep learning. The system processes video input from a webcam and provides instant text output.

## ✨ Features

- 🔄 Real-time hand gesture recognition
- 🤖 High-accuracy sign language interpretation
- 🌐 FastAPI backend with RESTful endpoints
- 📱 User-friendly interface
- 🚀 Optimized performance with GPU support
- 🔍 Advanced hand tracking with MediaPipe
- 📊 Real-time performance metrics
- ✨ Auto-correct functionality
- 🎯 Gesture stability detection

## 🛠️ Technical Stack

- **Backend**: FastAPI, Python 3.8+
- **Computer Vision**: OpenCV, MediaPipe
- **Deep Learning**: PyTorch, CNN
- **Data Processing**: NumPy
- **API Communication**: Requests
- **Text Processing**: SpellChecker

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam
- CUDA-capable GPU (optional, for better performance)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the server:
```bash
python app.py
```

2. Start the client:
```bash
python client.py
```

## 📁 Project Structure

```
├── app.py              # FastAPI server
├── client.py           # Webcam client
├── modules/            # Core modules
│   ├── sign_processor.py  # Sign processing logic
│   └── utils.py          # Utility functions
└── model/             # ML models
    └── trained_models/   # Pre-trained models
```

## 🔧 Configuration

### Environment Variables

- `API_HOST`: Server host (default: "0.0.0.0")
- `API_PORT`: Server port (default: 8000)
- `CAMERA_ID`: Webcam ID (default: 0)

### Model Parameters

- `CONFIDENCE_THRESHOLD`: 0.75
- `LETTER_DELAY`: 0.8s
- `REPEAT_DELAY`: 0.8s
- `DELETE_DELAY`: 0.2s
- `STABILITY_THRESHOLD`: 0.12

## 📊 Performance

- Real-time processing (30+ FPS)
- Low latency response
- High accuracy in sign recognition
- Stable gesture detection

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push branch (`git push origin feature/NewFeature`)
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- MediaPipe for hand tracking
- PyTorch team
- FastAPI framework
- OpenCV community 