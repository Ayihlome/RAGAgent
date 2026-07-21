from langchain_ollama import OllamaEmbeddings
from loader import chunks

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


def load_vectors(chunks):
    vectors = embeddings.embed_documents(
        [chunk.page_content for chunk in chunks]
    )
    print("Chunks Embedded...\n")
    return vectors

