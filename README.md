
# 🤟 Hand Gesture Recognition & Translation System

This project is a real-time hand gesture recognition and translation system using computer vision and deep learning. It recognizes hand signs via webcam, translates them into text, optionally converts them into speech, and can also translate the recognized words/sentences into multiple languages.

## 🧠 Powered By
- OpenCV
- CVZone
- TensorFlow (via Google Teachable Machine)
- Tkinter (for GUI)
- Google Translate API (`googletrans`)
- pyttsx3 (Text-to-Speech)

---

## 📁 Project Structure

```
├── data/                 # Collected images for training, e.g., data/hi, data/A, etc.
├── Model/
│   ├── keras_model.h5    # Trained model using Teachable Machine
│   └── labels.txt        # Corresponding gesture labels
├── datacollection.py     # Script to collect and save gesture images
├── test.py               # Main GUI app for gesture recognition and translation
├── README.md             # Project documentation
```

---

## 📹 Features

- ✋ **Hand Gesture Recognition** using webcam.
- 🗣 **Text-to-Speech** conversion of recognized gestures.
- 🌐 **Multi-language Translation** (Hindi, Japanese, Korean, Italian, French).
- 💾 **Save Gestures** into a sentence structure.
- 🔁 **Replay** the gestures from scratch.
- 🖼️ Clean and modern **Tkinter GUI** for interactive use.

---

## ⚙️ How It Works

### 1. Data Collection (`datacollection.py`)
- Uses OpenCV and CVZone to detect a hand and crop it to a 300x300 image.
- Saves captured hand gestures into folders inside `/data` (e.g., `data/hi`, `data/A`, etc.)
- Press **`s`** key to save a frame.

### 2. Gesture Recognition & Translation (`test.py`)
- Loads trained model from `Model/keras_model.h5`.
- Uses a live webcam feed to detect and classify gestures.
- Allows user to:
  - Save multiple gestures (as words).
  - Translate them to another language.
  - Hear them via text-to-speech.
  - Reset the session.

---

## 🛠 Requirements

Install dependencies using pip:

```bash
pip install opencv-python cvzone numpy pillow googletrans==4.0.0-rc1 pyttsx3 tensorflow
```

> Make sure you have Python 3.6 or higher.

---

## 🧪 How to Run

### For Data Collection:

```bash
python datacollection.py
```

> Save images by pressing `s`. Images go inside `data/<gesture-name>` folder.

---

### For GUI App & Recognition:

```bash
python test.py
```

> Make sure the trained model and labels are in the `Model/` directory.

---

## 🎓 Model Training

The model (`keras_model.h5`) was trained using **[Google Teachable Machine](https://teachablemachine.withgoogle.com/)** with hand gesture images collected via `datacollection.py`. Each class (like A, B, hi, ok, etc.) must have sufficient samples (ideally 100+).

---

## 🌍 Supported Languages for Translation

- 🇮🇳 Hindi (`hi`)
- 🇫🇷 French (`fr`)
- 🇮🇹 Italian (`it`)
- 🇯🇵 Japanese (`ja`)
- 🇰🇷 Korean (`ko`)

---

## 📝 Future Enhancements

- Support for dynamic gestures or full ASL signs.
- Improved model with more gesture classes.
- Support for user-created gesture-label mapping.
- Add gesture-to-sentence auto-completion using NLP.

---

## 📷 Screenshots

> *(Add screenshots of the app interface, training data, live predictions, etc.)*

---

## 🤝 Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License.
