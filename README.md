# Sign Language Recognition System

## 🎯 Overview

A real-time sign language recognition system that converts hand gestures to text using computer vision and deep learning. The system processes video input from a webcam and provides instant text output with auto-correction capabilities.

## Contributors

- Muhammad Amar
- Mahmoud Elnagar [Mahmoud Elnagar](https://github.com/ahmedmohamed)
 
  
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
- **Backend**: FastAPI 0.115.12
- **Computer Vision**: OpenCV 4.11.0.86, MediaPipe 0.10.21
- **Deep Learning**: PyTorch 2.6.0 (CUDA 12.4), TorchVision 0.21.0
- **Data Processing**: NumPy 1.26.4
- **API Communication**: Requests 2.32.3
- **Text Processing**: PySpellChecker 0.8.2

### Additional Tools
- **Visualization**: Matplotlib 3.10.1, Seaborn 0.13.2
- **Image Processing**: Pillow 11.0.0
- **Environment Management**: python-dotenv 1.1.0

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam
- CUDA 12.4 compatible GPU (recommended)
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
- GPU acceleration with CUDA 12.4

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
   - Enable GPU support (CUDA 12.4)
   - Reduce video resolution
   - Close background applications

3. **Installation errors**
   - Update pip: `pip install --upgrade pip`
   - Install build tools: `sudo apt-get install build-essential`
   - For CUDA issues, check NVIDIA drivers (CUDA 12.4)

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
