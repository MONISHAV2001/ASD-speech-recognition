import streamlit as st
import numpy as np
import librosa
import librosa.display
import pickle
import os
import matplotlib.pyplot as plt
import time

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Autism Detection", layout="centered")

# ==============================
# ANIMATED UI (CSS)
# ==============================
st.markdown("""
<style>
body {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
    color: white;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

h1 {
    text-align: center;
    animation: fadeIn 2s ease-in-out;
}

.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except:
        st.error("❌ Model not found!")
        return None

model = load_model()

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0), audio, sr

# ==============================
# HEADER
# ==============================
st.title("🧠 Autism Prediction from Speech")
st.markdown("### 🎤 Upload or Record Audio")

# ==============================
# INPUT
# ==============================
uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])
audio_bytes = st.audio_input("Or Record Live Audio")

file_path = None

if uploaded_file:
    file_path = "temp.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(file_path)

elif audio_bytes:
    file_path = "recorded.wav"
    with open(file_path, "wb") as f:
        f.write(audio_bytes.read())
    st.audio(file_path)

# ==============================
# PREDICTION
# ==============================
if file_path and model:

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing audio... 🎧"):
            time.sleep(2)

        features, audio, sr = extract_features(file_path)
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        st.markdown("## 🧾 Result")

        if prediction == 1:
            st.error("🧠 Autism Detected")
        else:
            st.success("✅ Non-Autism")

        st.write("**Numeric (0 = Non-Autism, 1 = Autism):**", int(prediction))
        st.write("**Confidence Score:**", round(float(np.max(prob)), 3))

        # ==============================
        # PIE CHART
        # ==============================
        st.markdown("## 📊 Prediction Distribution")

        labels = ["Non-Autism", "Autism"]
        fig1, ax1 = plt.subplots()
        ax1.pie(prob, labels=labels, autopct='%1.2f%%')
        ax1.set_title("Confidence Distribution")
        st.pyplot(fig1)

        # ==============================
        # SPECTROGRAM
        # ==============================
        st.markdown("## 🎧 Spectrogram")

        fig2, ax2 = plt.subplots()
        S = librosa.feature.melspectrogram(y=audio, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)

        img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
        fig2.colorbar(img, ax=ax2)
        ax2.set_title("Mel Spectrogram")

        st.pyplot(fig2)

        # ==============================
        # REAL-TIME WAVEFORM ANIMATION
        # ==============================
        st.markdown("## 🎵 Waveform Animation")

        fig_wave, ax_wave = plt.subplots()
        wave_placeholder = st.empty()

        audio_norm = audio / np.max(np.abs(audio))
        chunk_size = int(len(audio_norm) / 50)

        for i in range(0, len(audio_norm), chunk_size):
            ax_wave.clear()

            ax_wave.plot(audio_norm[:i])
            ax_wave.set_ylim(-1, 1)
            ax_wave.set_title("Audio Waveform (Real-Time Simulation)")
            ax_wave.set_xlabel("Samples")
            ax_wave.set_ylabel("Amplitude")

            wave_placeholder.pyplot(fig_wave)
            time.sleep(0.05)

        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
