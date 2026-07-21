from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from loader import chunks


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory= "./database"
)

# Temp RAG testing
results = db.similarity_search(
    "What is Continuous Integration?",
    k=3
)

for i, doc in enumerate(results):
    print(f"\nResult {i+1}")
    print(doc.page_content[:300])