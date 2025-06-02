# Sign Language Recognition System

Hey there! 👋 This is a cool project that helps you convert sign language into text in real-time. Just wave your hands in front of your webcam, and it'll show you the text right away. Pretty neat, right?

---

## What Can It Do? 🤔

- Watch your hands and understand sign language instantly
- Super accurate - it gets it right most of the time
- Easy to use - no complicated setup needed
- Works faster if you have a good graphics card
- Tracks your hands really well
- Shows you how well it's working in real-time
- Fixes typos automatically
- Won't get confused by shaky hands

---

## What's Under the Hood? 🛠️

### The Main Stuff
- **Backend**: FastAPI (it's super fast!)
- **Computer Vision**: OpenCV and MediaPipe (for tracking your hands)
- **Deep Learning**: PyTorch (the brain of the operation)
- **Data Stuff**: NumPy (for number crunching)
- **API Stuff**: Requests (for talking to the server)
- **Text Fixing**: PySpellChecker (for fixing typos)

### Extra Goodies
- **Charts**: Matplotlib and Seaborn (for pretty graphs)
- **Images**: Pillow (for handling pictures)
- **Setup**: python-dotenv (for keeping secrets)

---

## Let's Get Started! 🚀

### What You'll Need
- Python 3.8 or newer
- A webcam (most laptops have one built-in)
- A graphics card with CUDA 12.4 (optional, but makes it faster)
- pip (comes with Python)

### Setting It Up

1. **Get the code**:
```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

2. **Make a virtual environment** (keeps things clean):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the requirements**:
```bash
pip install -r requirements.txt
```

### Running It

1. **Start the server**:
```bash
python app.py
```

2. **Open the camera**:
```bash
python client.py
```

---

## What's in the Box? 📁

```
├── app.py                 # The server part
├── client.py              # The camera part
├── modules/               # The brainy stuff
│   ├── sign_processor.py  # Understands your signs
│   └── utils.py           # Helper functions
├── model/                 # The trained brain
│   └── trained_models/    # Pre-trained models
├── tests/                 # Making sure everything works
│   ├── test_sign_processor.py
│   ├── test_utils.py
│   └── conftest.py
├── requirements.txt       # List of needed packages
└── README.md              # This file
```

---

## Settings ⚙️

### Environment Stuff
Make a file called `.env` and put this in it:
```env
API_HOST=0.0.0.0
API_PORT=5000
CAMERA_ID=0
```

### Model Settings
- How sure it needs to be: 75%
- How long to wait between letters: 0.8 seconds
- How long to wait before repeating: 0.8 seconds
- How long to wait before deleting: 0.2 seconds
- How stable your hand needs to be: 0.12

---

## How Fast Is It? 📊

- Keeps up with your hands (30+ frames per second)
- Responds super quick (less than 100ms)
- Gets it right most of the time (90%+)
- Doesn't get confused by shaky hands
- Goes even faster with a good graphics card

---

## How to Use It 🖐️

1. Start it up
2. Put your hand in front of the camera
3. Start signing
4. Watch the text appear
5. Use the delete sign if you make a mistake
6. It'll fix typos when you add a space

---

## Having Trouble? 🛠️

### Common Issues

**1. Camera not working?**
- Make sure nothing else is using it
- Try changing the `CAMERA_ID` in the `.env` file

**2. Running slow?**
- Use a graphics card if you have one
- Turn down the video quality
- Close other programs

**3. Can't install it?**
- Update pip:
  ```bash
  pip install --upgrade pip
  ```
- On Linux, get the build tools:
  ```bash
  sudo apt-get install build-essential
  ```
- Make sure your graphics drivers are up to date

---

## Want to Help? 🤝

1. Fork the repo
2. Make a new branch:
   ```bash
   git checkout -b feature/NewFeature
   ```
3. Save your changes:
   ```bash
   git commit -m "Added something cool"
   ```
4. Push it up:
   ```bash
   git push origin feature/NewFeature
   ```
5. Send us a pull request

---

## Legal Stuff 📝

This project is under the MIT License.  
Check the [LICENSE](LICENSE) file for the details.
