import streamlit as st
import speech_recognition as sr
import tempfile
import os
import subprocess

from streamlit_mic_recorder import mic_recorder

from rag.retriever import retrieve_docs
from rag.local_llm import get_local_llm


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Financial RAG Chatbot")

st.title("Financial RAG Chatbot")


# -----------------------------
# FFMPEG CONFIG
# -----------------------------
# On cloud deployment use just "ffmpeg"
FFMPEG_PATH = "ffmpeg"


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

        # Convert webm -> wav
        subprocess.run(
            [FFMPEG_PATH, "-y", "-i", raw_path, wav_path],
            check=True,
            capture_output=True
        )

        # Speech to text
        with sr.AudioFile(wav_path) as source:

            audio_data = recognizer.record(source)

            voice_text = recognizer.recognize_google(audio_data)

            st.success(f"You said: {voice_text}")

            question = voice_text

    except subprocess.CalledProcessError as e:

        st.error(f"ffmpeg conversion failed: {e.stderr.decode()}")

    except sr.UnknownValueError:

        st.error("Could not understand the audio.")

    except sr.RequestError as e:

        st.error(f"Speech API error: {e}")

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

            # -----------------------------
            # RETRIEVE DOCUMENTS
            # -----------------------------
            docs = retrieve_docs(question)

            if not docs:

                st.error("No relevant documents found.")

            else:

                # -----------------------------
                # BUILD CONTEXT
                # -----------------------------
                context = "\n\n".join([
                    f"Page {d.metadata.get('page')}:\n{d.page_content[:500]}"
                    for d in docs
                ])

                # -----------------------------
                # PROMPT
                # -----------------------------
                prompt = f"""
You are a financial assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
say:
"I could not find this information in the documents."

Context:
{context}

Question:
{question}

Give a concise factual answer.
"""

                # -----------------------------
                # LLM
                # -----------------------------
                llm = get_local_llm()

                answer = llm.generate_response(prompt)

                # -----------------------------
                # OUTPUT
                # -----------------------------
                st.subheader("Answer")

                st.write(answer)

                # -----------------------------
                # SOURCES
                # -----------------------------
                st.subheader("Sources")

                for d in docs:

                    st.json({
                        "file": d.metadata.get("source"),
                        "page": d.metadata.get("page")
                    })

        except Exception as e:

            st.error(f"Error: {str(e)}")