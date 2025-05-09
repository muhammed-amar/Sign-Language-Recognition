# Sign Language Recognition System

## 🎯 Overview

A real-time sign language recognition system that converts hand gestures to text using computer vision and deep learning. The system processes video input from a webcam and provides instant text output with auto-correction capabilities.

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

### Core Technologies
- **Backend**: FastAPI 0.104.1
- **Computer Vision**: OpenCV 4.8.1, MediaPipe 0.10.8
- **Deep Learning**: PyTorch 2.1.1, TorchVision 0.16.1
- **Data Processing**: NumPy 1.24.3
- **API Communication**: Requests 2.31.0
- **Text Processing**: PySpellChecker 0.7.2

### Additional Tools
- **Visualization**: Matplotlib 3.8.2, Seaborn 0.13.0
- **Image Processing**: Pillow 10.1.0
- **Progress Tracking**: tqdm 4.66.1
- **Environment Management**: python-dotenv 1.0.0

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam
- CUDA-capable GPU (optional, for better performance)
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
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
├── model/             # ML models
│   └── trained_models/   # Pre-trained models
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
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
- Low latency response (<100ms)
- High accuracy in sign recognition (>90%)
- Stable gesture detection
- GPU acceleration support

## 🔍 Usage

1. Start the application
2. Position your hand in front of the webcam
3. Make sign language gestures
4. View the recognized text in real-time
5. Use delete gesture to remove characters
6. System will auto-correct words when space is detected

## 🛠️ Troubleshooting

### Common Issues

1. **Webcam not detected**
   - Check camera connection
   - Verify camera permissions
   - Try different `CAMERA_ID`

2. **Low performance**
   - Enable GPU support
   - Reduce video resolution
   - Close background applications

3. **Installation errors**
   - Update pip: `pip install --upgrade pip`
   - Install build tools: `sudo apt-get install build-essential`
   - For CUDA issues, check NVIDIA drivers

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
- PyTorch team for deep learning framework
- FastAPI framework
- OpenCV community
- All contributors and users 
