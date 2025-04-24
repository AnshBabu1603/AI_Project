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

# Insert the custom CSS and JS
st.markdown("""
    <style>
        :root {
            --text-strong: #0d0d0d;  
            --text-primary: #272727;  
            --text-secondary: #6b6b6b;  
            --page-primary: #fff;
        }

        :root[data-theme="dark"] {
            color-scheme: dark;
            --text-strong: #fff;
            --text-primary: #f0f0f0;  
            --text-secondary: #c7c7c7;  
            --page-primary: #0d0d0d;  
        }

        .type-hero {
            font-size: 128px;
            font-weight: 400;
            line-height: 1;
            letter-spacing: -1.28px;
        }

        .type-body-01 {
            font-size: 14px;
            font-weight: 400;
            line-height: 20px;
            letter-spacing: 0.14px;
        }

        .type-heading-04 {
            font-size: 28px;
            font-weight: 400;
            line-height: 32px;
            letter-spacing: -0.28px;
        }

        .text-primary {
            color: var(--text-primary);
        }

        .text-secondary {
            color: var(--text-secondary);
        }

        body {
            font-family: Roobert, sans-serif;
            margin: 0;
            padding: 0 1rem;
            background: var(--page-primary);
            color: var(--text-strong);
        }

        body {
            display: flex;
            flex-direction: column;
            text-align: center;
            min-height: 100vh;
        }

        h2 {
            margin: 0;
        }

        a {
            color: inherit;
        }

        .site-content {
            margin: auto;
        }

        header {
            margin: 1.5rem 0;
        }

        main > * {
            margin-bottom: 1rem;
        }

        .request-id {
            margin-bottom: 2rem;
        }

        footer {
            padding: 1.75rem 0;
        }

        .logo-render {
            margin-left: 0.25rem;
        }
    </style>

    <script>
        (function () {
            try {
                var prefersDark = window.matchMedia(
                    "(prefers-color-scheme: dark)"
                ).matches;
                if (!prefersDark) return;
                document.documentElement.setAttribute("data-theme", "dark");
                var favicon = document.getElementById("favicon");
                if (favicon)
                    favicon.setAttribute(
                        "href",
                        "data:image/svg+xml,%3Csvg width='16' height='16' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cg clip-path='url(%23clip0_459_963)'%3E%3Cpath d='M11.4327 1.00388C9.64526 0.919753 8.14218 2.21231 7.88574 3.91533C7.87559 3.99436 7.86035 4.07085 7.84766 4.14733C7.44904 6.26845 5.59303 7.87459 3.3638 7.87459C2.5691 7.87459 1.82263 7.67064 1.17265 7.31372C1.09394 7.27038 1 7.32647 1 7.4157V7.87204V14.7479H7.84512V9.59291C7.84512 8.64452 8.61189 7.87459 9.5564 7.87459H11.2677C13.2049 7.87459 14.7639 6.2608 14.6877 4.29774C14.6191 2.53099 13.1922 1.08802 11.4327 1.00388Z' fill='white'/%3E%3C/g%3E%3Cdefs%3E%3CclipPath id='clip0_459_963'%3E%3Crect width='14' height='14' fill='white' transform='translate(1 1)'/%3E%3C/clipPath%3E%3C/defs%3E%3C/svg%3E%0A"
                    );
            } catch (e) {}
        })();
    </script>
""", unsafe_allow_html=True)

# Display the title and emotion-based background and emoji
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
