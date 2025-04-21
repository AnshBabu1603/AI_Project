import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from youtube_search import YoutubeSearch
import os

# Load the model and labels
model = load_model("models/model.h5")
label = np.load("models/labels.npy")

# Initialize MediaPipe for face and hand landmarks
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

# Initialize emotion state in session_state
if "emotion" not in st.session_state:
    st.session_state["emotion"] = None

# If we have a previously saved emotion in session_state, restore it
if "emotion" in st.session_state and st.session_state["emotion"] is None:
    if os.path.exists("emotion.npy"):
        try:
            detected_emotion = np.load("emotion.npy", allow_pickle=True)[0]
            if isinstance(detected_emotion, str) and detected_emotion.strip():
                st.session_state["emotion"] = detected_emotion
        except Exception as e:
            st.error(f"Error loading saved emotion: {e}")
            st.session_state["emotion"] = None

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        result = holistic.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        features = []

        if result.face_landmarks:
            base = result.face_landmarks.landmark[1]
            features.extend([(lm.x - base.x, lm.y - base.y) for lm in result.face_landmarks.landmark])

            for hand_lm, ref_idx in [
                (result.left_hand_landmarks, 8),
                (result.right_hand_landmarks, 8),
            ]:
                if hand_lm:
                    base = hand_lm.landmark[ref_idx]
                    features.extend([(lm.x - base.x, lm.y - base.y) for lm in hand_lm.landmark])
                else:
                    features.extend([(0.0, 0.0)] * 21)

            features_np = np.array(features).flatten().reshape(1, -1)
            prediction = model.predict(features_np, verbose=0)
            emotion = label[np.argmax(prediction)]
            st.session_state["emotion"] = emotion
            np.save("emotion.npy", np.array([emotion]))  # Save emotion for future sessions

            cv2.putText(frm, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw landmarks on face and hands
        drawing.draw_landmarks(frm, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, result.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, result.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


def get_youtube_video(query):
    try:
        results = YoutubeSearch(query, max_results=1).to_dict()
        if results:
            return f"https://www.youtube.com/embed/{results[0]['id']}"
    except Exception as e:
        st.error(f"🎥 YouTube search error: {e}")
    return None


# ---------- UI Starts Here ----------
st.title("🎵 Emotion-Based Music Recommender 🎶")

# Emotion background and emoji color coding
emotion_color_map = {
    "happy": "#FFD700",
    "sad": "#4682B4",
    "angry": "#FF4500",
    "shocked": "#32CD32",
    "rock": "#000000"
}

emotion_emoji_map = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😡",
    "shocked": "😲",
    "rock": "🤘"
}

# Display background color based on detected emotion
emotion = st.session_state.get("emotion")
if emotion in emotion_color_map:
    st.markdown(f"""
        <style>
            .stApp {{
                background-color: {emotion_color_map[emotion]};
                color: white;
            }}
        </style>
    """, unsafe_allow_html=True)

if emotion in emotion_emoji_map:
    st.markdown(f"## Mood Detected: {emotion_emoji_map[emotion]} {emotion.capitalize()}")

# Inputs for language and singer
lang = st.text_input("Enter Language (e.g., English, Hindi)")
singer = st.text_input("Enter Singer Name")

# Start camera only after inputs
if lang and singer:
    st.subheader("🎥 Detecting Emotion from Camera...")
    webrtc_streamer(
        key="emotion_stream",
        video_processor_factory=EmotionProcessor,
        async_processing=True
    )

# Button to fetch songs based on detected emotion
if st.button("Recommend me songs"):
    emotion = st.session_state.get("emotion")

    if not emotion:
        st.warning("⚠️ Please let me capture your emotion first!")
    elif lang and singer:
        st.success(f"✅ Detected Emotion: {emotion}")
        query = f"{singer} {emotion} {lang} song"
        video_url = get_youtube_video(query)

        if video_url:
            st.success(f"🎵 Now Playing: {singer}'s {emotion} song")
            st.video(video_url)
        else:
            st.error("❌ No matching songs found.")

        # Clear state after recommendation
        st.session_state["emotion"] = None
        np.save("emotion.npy", np.array([""]))
    else:
        st.warning("⚠️ Please enter both language and singer!")
