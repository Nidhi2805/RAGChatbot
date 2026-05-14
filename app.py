from dotenv import load_dotenv
import os

load_dotenv()

print("GROQ API:", os.getenv("GROQ_API_KEY"))

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from ingestion.loader import load_documents
from ingestion.cleaner import clean_documents
from ingestion.chunker import chunk_documents

from rag.retriever import (
    create_vectorstore,
    retrieve_docs
)

from rag.prompt import build_prompt
from rag.local_llm import get_local_llm

load_dotenv()

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG Chatbot Running"}


@app.post("/ingest")
def ingest_documents():
    try:
        docs = load_documents()
        cleaned_docs = clean_documents(docs)
        chunks = chunk_documents(cleaned_docs)
        create_vectorstore(chunks)

        return {
            "status": "success",
            "documents_loaded": len(docs),
            "chunks_created": len(chunks)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/chat")
def chat(request: QueryRequest):
    try:
        # Step 1: Retrieve docs
        docs = retrieve_docs(request.question)

        if not docs:
            return {
                "answer": "I could not find this information in the documents.",
                "sources": []
            }

        # Step 2: Build context
        context_parts = []
        for i, d in enumerate(docs):
            page = d.metadata.get('page', 'Unknown')
            source = d.metadata.get('source_file', 'Unknown')
            context_parts.append(
                f"[Source {i+1} | File: {source} | Page: {page}]\n"
                f"{d.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Build prompt
        prompt = build_prompt(context, request.question)

        # Step 4: Get LLM response
        llm = get_local_llm()
        
        # Validate LLM response
        raw_answer = llm.generate_response(prompt)
        
        if not raw_answer:
            return {
                "answer": "The model returned an empty response. Please try again.",
                "sources": []
            }

        if not isinstance(raw_answer, str):
            raw_answer = str(raw_answer)

        # Step 5: Build sources
        sources = []
        for d in docs:
            source_entry = {
                "file": d.metadata.get("source_file", "Unknown"),
                "page": d.metadata.get("page", "Unknown")
            }
            if source_entry not in sources:
                sources.append(source_entry)

        return {
            "answer": raw_answer.strip(),
            "sources": sources
        }

    except KeyError as e:
        # Handle missing keys
        return {
            "answer": f"Data error: Missing key {str(e)}. Please re-ingest documents.",
            "sources": []
        }

    except FileNotFoundError as e:
        # Handle missing vectorstore
        return {
            "answer": "Documents not ingested yet. Please click ingest first.",
            "sources": []
        }

    except ValueError as e:
        return {
            "answer": f"Value error: {str(e)}",
            "sources": []
        }

    except Exception as e:
        # Catch all - always return 'answer' key
        return {
            "answer": f"An error occurred: {str(e)}. Please try again.",
            "sources": []
        }