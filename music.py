import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from youtube_search import YoutubeSearch
import os


model = load_model(r"C:\Users\coola\OneDrive\Desktop\AI Project\models\model.h5")
label = np.load(r"C:\Users\coola\OneDrive\Desktop\AI Project\models\labels.npy")



holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils


if "emotion" not in st.session_state:
    st.session_state["emotion"] = None


if os.path.exists(r"C:\Users\coola\OneDrive\Desktop\AI Project\data\emotion.npy"):
    try:
        detected_emotion = np.load(r"C:\Users\coola\OneDrive\Desktop\AI Project\data\emotion.npy")[0]
        if isinstance(detected_emotion, str) and detected_emotion.strip():
            st.session_state["emotion"] = detected_emotion
    except:
        st.session_state["emotion"] = None


class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)

            
            pred = label[np.argmax(model.predict(lst))]
            print("Detected Emotion:", pred)
            st.session_state["emotion"] = pred
            np.save(r"C:\Users\coola\OneDrive\Desktop\AI Project\data\emotion.npy", np.array([pred]))

            
            cv2.putText(frm, f"{pred}", (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


def get_youtube_video(song_query):
    results = YoutubeSearch(song_query, max_results=1).to_dict()
    if results:
        video_id = results[0]['id']
        return f"https://www.youtube.com/embed/{video_id}"
    return None


st.title("🎵 Emotion-Based Music Recommender 🎶")


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


if st.session_state["emotion"] in emotion_color_map:
    bg_color = emotion_color_map[st.session_state["emotion"]]
    st.markdown(f"""
        <style>
            .stApp {{
                background-color: {bg_color};
                color: white; /* Adjust text color for contrast */
            }}
        </style>
    """, unsafe_allow_html=True)


if st.session_state["emotion"] in emotion_emoji_map:
    st.markdown(f"## Mood Detected: {emotion_emoji_map[st.session_state['emotion']]} {st.session_state['emotion'].capitalize()}")


lang = st.text_input("Enter Language (e.g., English, Hindi)")
singer = st.text_input("Enter Singer Name")


if lang and singer:
    st.subheader("🎥 Detecting Emotion from Camera...")
    webrtc_streamer(key="emotion_stream", video_processor_factory=lambda: EmotionProcessor())


if st.button("Recommend me songs"):
    detected_emotion = st.session_state["emotion"]

    if not detected_emotion:
        st.warning("⚠️ Please let me capture your emotion first!")
    elif lang and singer:
        st.success(f"✅ Detected Emotion: {detected_emotion}")

        
        song_query = f"{singer} {detected_emotion} {lang} song"
        video_url = get_youtube_video(song_query)

        if video_url:
            st.success(f"🎵 Now Playing: {singer}'s {detected_emotion} song")
            st.video(video_url)  
        else:
            st.error("❌ No matching songs found!")

        
        np.save("emotion.npy", np.array([""]))
        st.session_state["emotion"] = None
    else:
        st.warning("⚠️ Please enter both language and singer!")
