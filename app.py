from flask import Flask, render_template, request
import tensorflow as tf
import librosa
import numpy as np
import os

app = Flask(__name__)

# Load model once (path must match your saved model)
model = tf.keras.models.load_model("model/emotion_model.keras")

# Emotion labels according to your training order
EMOTIONS = ["angry", "happy", "sad", "neutral", "fear", "disgust", "surprise"]


def extract_segment_features(file_path, segment_duration=1.0, max_duration=60.0):
    """
    Slice the audio into fixed-length segments and extract MFCC features
    for each segment. Returns a batch of features and corresponding
    start-times (in seconds) for each segment.
    """
    audio, sr = librosa.load(file_path, sr=None)

    # Optionally limit very long files for robustness
    max_samples = int(max_duration * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    segment_length = int(segment_duration * sr)
    if segment_length <= 0:
        raise ValueError("segment_duration must be > 0")

    features = []
    timestamps = []

    num_segments = max(1, int(np.ceil(len(audio) / segment_length)))
    for i in range(num_segments):
        start = i * segment_length
        end = min(start + segment_length, len(audio))
        segment = audio[start:end]

        if len(segment) == 0:
            continue

        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        features.append(mfcc_scaled)

        start_time_sec = start / sr
        timestamps.append(start_time_sec)

    if not features:
        return None, None

    return np.array(features), timestamps


def summarize_timeline(predicted_emotions, timestamps):
    """
    Build a detailed timeline of predictions and a compact list
    of 'changes' (only when emotion switches).
    """
    timeline = []
    for emotion, t in zip(predicted_emotions, timestamps):
        timeline.append(
            {
                "time": float(t),
                "emotion": emotion,
            }
        )

    changes = []
    prev = None
    for point in timeline:
        if point["emotion"] != prev:
            changes.append(point)
            prev = point["emotion"]

    return timeline, changes


@app.route("/")
def index():
    return render_template("index.html", emotions=EMOTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    file_path = "uploaded.wav"
    file.save(file_path)

    # Extract segment-wise features for timeline analysis
    features_batch, timestamps = extract_segment_features(file_path)
    if features_batch is None:
        return "Could not extract features from audio"

    # Run model on all segments at once
    predictions = model.predict(features_batch)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_emotions = [EMOTIONS[idx] for idx in predicted_indices]

    # Majority emotion across the whole clip (for overall summary)
    unique, counts = np.unique(predicted_indices, return_counts=True)
    dominant_idx = unique[np.argmax(counts)]
    overall_emotion = EMOTIONS[int(dominant_idx)]

    # Build detailed timeline and condensed list of changes
    timeline, changes = summarize_timeline(predicted_emotions, timestamps)

    return render_template(
        "index.html",
        prediction=overall_emotion,
        emotions=EMOTIONS,
        timeline=timeline,
        changes=changes,
    )

if __name__ == "__main__":
    app.run(debug=True)
