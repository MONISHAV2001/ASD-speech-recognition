import streamlit as st
import numpy as np
import librosa
import pickle
import os
import matplotlib.pyplot as plt

MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except:
        st.error("Model not found!")
        return None

model = load_model()

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

st.title("🧠 Autism Prediction from Speech")

uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded_file is not None and model is not None:
    file_path = "temp.wav"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(file_path)

    if st.button("Predict"):
        features = extract_features(file_path).reshape(1, -1)

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        # TEXT OUTPUT
        if prediction == 1:
            st.error("Prediction: Autism")
        else:
            st.success("Prediction: Non-Autism")

        # NUMERIC OUTPUT
        st.write("Numeric (0 = Non-Autism, 1 = Autism):", int(prediction))

        # CONFIDENCE
        confidence = float(np.max(prob))
        st.write("Confidence Score:", round(confidence, 3))

        # GRAPH
        st.subheader("Prediction Probability Graph")
        labels = ["Non-Autism (0)", "Autism (1)"]
        values = prob

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

    if os.path.exists(file_path):
        os.remove(file_path)