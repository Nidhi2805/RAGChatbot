import streamlit as st
import requests
import speech_recognition as sr
import tempfile
import os
import subprocess

from streamlit_mic_recorder import mic_recorder


FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Financial RAG Chatbot")
st.title("Financial RAG Chatbot")


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.header("Setup")

    if st.button("📥 Ingest Documents"):

        try:

            response = requests.post(f"{BACKEND_URL}/ingest", timeout=120)
            data = response.json()
            st.success(
                f"Ingested {data['documents_loaded']} documents "
                f"and created {data['chunks_created']} chunks."
            )

        except Exception as e:

            st.error(f"Cannot connect to backend: {e}")


# -----------------------------
# CHAT SECTION
# -----------------------------
st.subheader("Ask a question")


# -----------------------------
# TEXT INPUT
# -----------------------------
question = st.text_input(
    "Question",
    placeholder="What is the investment objective of Bajaj Finserv Large Cap Fund?",
    label_visibility="collapsed"
)


# -----------------------------
# VOICE INPUT
# -----------------------------
st.markdown("### 🎤 Voice Input")

audio = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    key="mic_recorder"
)

if audio:

    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw_file:
        raw_file.write(audio["bytes"])
        raw_path = raw_file.name

    wav_path = raw_path.replace(".webm", ".wav")

    try:

        # Call ffmpeg directly — bypasses pydub entirely
        result = subprocess.run(
            [FFMPEG_PATH, "-y", "-i", raw_path, wav_path],
            check=True,
            capture_output=True
        )

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            voice_text = recognizer.recognize_google(audio_data)
            st.success(f"You said: {voice_text}")
            question = voice_text

    except subprocess.CalledProcessError as e:
        st.error(f"ffmpeg conversion failed: {e.stderr.decode()}")

    except sr.UnknownValueError:
        st.error("Could not understand the audio. Please try speaking more clearly.")

    except sr.RequestError as e:
        st.error(f"Google Speech API error: {e}")

    except Exception as e:
        st.error(f"Speech recognition failed: {str(e)}")

    finally:
        for path in [raw_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)


# -----------------------------
# SUBMIT BUTTON
# -----------------------------
if st.button("Submit"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={"question": question},
                timeout=120
            )

            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

        except requests.exceptions.Timeout:
            st.error("API request timed out.")

        except ValueError:
            st.error("Invalid response from API. Check if backend is properly initialized.")

        except Exception as e:
            st.error(f"Backend error: {e}")
