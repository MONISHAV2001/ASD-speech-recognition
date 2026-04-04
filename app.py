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
st.sidebar.title("ℹ️ About")
st.sidebar.write("""
Autism Prediction using Speech  
Model: XGBoost  
Features: MFCC + Delta + Spectral  
""")

# ==============================
# LOAD MODEL + SCALER
# ==============================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except:
        st.error("❌ Model or Scaler not found!")
        return None, None

model, scaler = load_model()

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spec = librosa.feature.spectral_contrast(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.mean(delta2, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spec, axis=1),
        np.mean(zcr, axis=1)
    ])

    return features, audio, sr

# ==============================
# UI
# ==============================
st.title("🧠 Autism Prediction from Speech")

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

        features, audio, sr = extract_features(file_path)
        features = scaler.transform([features])

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]
        confidence = float(np.max(prob))

        # RESULT
        st.markdown("## 🧾 Result")

        if prediction == 1:
            st.error("🧠 Autism Detected")
        else:
            st.success("✅ Non-Autism")

        st.write("Numeric:", int(prediction))
        st.write("Confidence:", round(confidence, 3))

        # PIE CHART
        st.markdown("## 📊 Prediction Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie(prob, labels=["Non-Autism", "Autism"], autopct='%1.2f%%')
        st.pyplot(fig1)

        # SPECTROGRAM
        st.markdown("## 🎧 Spectrogram")
        fig2, ax2 = plt.subplots()
        S = librosa.feature.melspectrogram(y=audio, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
        fig2.colorbar(img, ax=ax2)
        st.pyplot(fig2)

        # WAVEFORM ANIMATION
        st.markdown("## 🎵 Waveform Animation")

        fig_wave, ax_wave = plt.subplots()
        wave_placeholder = st.empty()

        max_val = np.max(np.abs(audio)) if np.max(np.abs(audio)) != 0 else 1
        audio_norm = audio / max_val

        chunk = int(len(audio_norm) / 50)

        for i in range(0, len(audio_norm), chunk):
            ax_wave.clear()
            ax_wave.plot(audio_norm[:i])
            ax_wave.set_ylim(-1, 1)
            wave_placeholder.pyplot(fig_wave)
            time.sleep(0.05)

        # DOWNLOAD RESULT
        result_text = f"""Autism Prediction Report

Prediction: {"Autism" if prediction == 1 else "Non-Autism"}
Numeric Output: {int(prediction)}
Confidence Score: {round(confidence, 3)}
"""

        st.download_button(
            "📥 Download Report",
            result_text,
            "report.txt"
        )

        # CLEANUP
        if os.path.exists(file_path):
            os.remove(file_path)
