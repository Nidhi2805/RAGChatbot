from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=[
            "\n\n\n",   # Section breaks
            "\n\n",     # Paragraph breaks
            "\n",       # Line breaks
            ". ",       # Sentence breaks
            ", ",       # Clause breaks
            " ",        # Word breaks
            ""          # Character breaks
        ],
        length_function=len,
        is_separator_regex=False
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index metadata for better tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    return chunks