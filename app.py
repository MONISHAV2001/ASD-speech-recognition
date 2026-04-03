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
# ANIMATED UI
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
.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("ℹ️ About Project")
st.sidebar.write("""
Autism Prediction using Speech  
Model: Random Forest  
Features: MFCC  
""")

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
    audio = librosa.effects.preemphasis(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0), audio, sr, mfcc

# ==============================
# HEADER
# ==============================
st.title("🧠 Autism Prediction from Speech")

# ==============================
# INPUT
# ==============================
uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])
audio_bytes = st.audio_input("Or Record Audio")

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

        with st.spinner("Analyzing audio..."):
            time.sleep(2)

        features, audio, sr, mfcc = extract_features(file_path)
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]
        confidence = float(np.max(prob))

        # RESULT
        st.markdown("## 🧾Result")

        if prediction == 1:
            st.error("🧠 Autism Detected")
        else:
            st.success("✅ Non-Autism")

        st.write("Numeric Score (0-1):", int(prediction))
        st.write("Confidence Score:", round(confidence, 3))

        # PIE CHART
        st.markdown("## 📊 Prediction Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie(prob, labels=["Non-Autism", "Autism"], autopct='%1.2f%%')
        st.pyplot(fig1)

        # SPECTROGRAM
        st.markdown("## 🎧 Spectrogram")
        fig3, ax3 = plt.subplots()
        S = librosa.feature.melspectrogram(y=audio, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax3)
        fig3.colorbar(img, ax=ax3)
        st.pyplot(fig3)

        # ANIMATED WAVEFORM
        st.markdown("## 🎵 Waveform Animation")
        fig_wave, ax_wave = plt.subplots()
        wave_placeholder = st.empty()

        audio_norm = audio / np.max(np.abs(audio))
        chunk = int(len(audio_norm) / 50)

        for i in range(0, len(audio_norm), chunk):
            ax_wave.clear()
            ax_wave.plot(audio_norm[:i])
            ax_wave.set_ylim(-1, 1)
            wave_placeholder.pyplot(fig_wave)
            time.sleep(0.05)

        # CLEANUP
        if os.path.exists(file_path):
            os.remove(file_path)
