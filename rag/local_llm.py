"""
rag/local_llm.py

Uses Groq's free API (llama-3.3-70b-versatile) for fast, accurate responses.
Get a free API key at: https://console.groq.com
"""

import os
import requests
import traceback

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama-3.3-70b-versatile"   # Free, fast, very capable
MAX_TOKENS   = 1024


# ──────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────

class LocalLLM:
    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self._validate_key()

    def _validate_key(self):
        if not GROQ_API_KEY:
            print(
                "⚠️  GROQ_API_KEY is not set.\n"
                "    Add it to your .env file:\n"
                "    GROQ_API_KEY=gsk_..."
            )
        else:
            print(f"✅ Groq LLM ready  (model: {self.model_name})")

    def generate_response(self, prompt: str) -> str:
        if not GROQ_API_KEY:
            return (
                "Error: GROQ_API_KEY is not configured. "
                "Please add it to your .env file and restart the server."
            )

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        }

        payload = {
            "model": self.model_name,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful financial assistant. "
                        "Answer questions based ONLY on the context provided. "
                        "If the answer is not in the context, say so clearly. "
                        "Be concise, factual, and professional."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        }

        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            return answer if answer else "I could not find this information in the documents."

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            body   = e.response.text       if e.response else ""
            print(f"❌ Groq API HTTP error {status}: {body}")
            traceback.print_exc()
            return f"Error calling Groq API (HTTP {status}). Check your API key."

        except requests.exceptions.Timeout:
            return "Error: The request to Groq API timed out. Please try again."

        except Exception as e:
            traceback.print_exc()
            return f"Error generating response: {str(e)}"


# ──────────────────────────────────────────────
# Lazy singleton
# ──────────────────────────────────────────────

_llm_instance: "LocalLLM | None" = None

def get_local_llm(model_name: str = MODEL) -> LocalLLM:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM(model_name=model_name)
    return _llm_instance