# TODO: Suggest changes to documentation here: https://docs.langchain.com/oss/python/langchain/knowledge-base

import os
import pypdf

from dotenv import load_dotenv
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

load_dotenv()

def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]


file_path = os.path.join(os.path.dirname(__file__), "nke-10k-2023.pdf")
docs = load_pdf_pages(str(file_path))

# uv add langchain-text-splitters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embeddings)

vector_store.add_documents(documents=all_splits)

# ImportError: cosine_similarity requires numpy to be installed. Please install numpy with `pip install numpy`.
results = vector_store.similarity_search_with_score(
    "How many distribution centers does Nike have in the US?"
)

doc, score = results[0]

print(doc.page_content)
print(doc.metadata["source"])
print(score)

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

print(results)

store_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

results = store_retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

print(results)