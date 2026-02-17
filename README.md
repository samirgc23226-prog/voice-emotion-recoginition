
# Voice Emotion Recognition

A Python-based project that detects emotions from voice audio using **TensorFlow** and **Flask**.  
Users can upload a `.wav` audio file, play it, and the web app predicts the emotion in the voice.

---

## 🔹 Features

- Predict emotions: **Angry, Happy, Sad, Neutral, Fear, Disgust, Surprise**
- Upload `.wav` audio files via a web interface
- Play audio before prediction
- Real-time emotion prediction with **TensorFlow**
- Modern UI with gradient background and animations
- Fully deployable on **free platforms** like Render or HuggingFace Spaces

---

## 🔹 Project Structure

```

Voice-Emotion-Recognition/
│── app.py                 # Flask backend
│── model/
│     └── emotion_model.keras   # Trained TensorFlow model
│── templates/
│     └── index.html        # Frontend HTML
│── static/                 # Optional: CSS or images
│── requirements.txt        # Python dependencies

````

---

## 🔹 Installation

1. Clone the repository:

```bash
git clone https://github.com/samirghumde/Voice-Emotion-Recognition.git
cd Voice-Emotion-Recognition
````

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

* **Windows:**

```bash
venv\Scripts\activate
```

* **Linux/Mac:**

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔹 Run the App Locally

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000/
```

Upload a `.wav` audio file, play it, and see the predicted emotion.

---

## 🔹 Deployment

You can deploy this app for free using:

* **Render:** [https://render.com](https://render.com)
* **HuggingFace Spaces:** [https://huggingface.co/spaces](https://huggingface.co/spaces)

Simply push your repository and configure Python + Flask runtime.

---

## 🔹 How It Works

1. **Upload Audio:** User uploads a `.wav` file via web interface.
2. **Feature Extraction:** Flask extracts MFCC features using `librosa`.
3. **Prediction:** Pre-trained TensorFlow model predicts the emotion.
4. **Display Result:** Shows emotion on the web page with animation.

---

## 🔹 Requirements

* Python 3.10+
* TensorFlow
* Flask
* librosa
* numpy
* soundfile
* gunicorn (for deployment)

---

## 🔹 License

This project is free to use and modify.




