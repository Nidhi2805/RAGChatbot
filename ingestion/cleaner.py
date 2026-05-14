import re


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+", "", text)
    text = text.strip()

    return text


def clean_documents(documents):
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    return documents