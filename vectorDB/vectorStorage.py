from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from pathlib import Path

class VectorDB:
    def __init__(self):
        self.FILE_PATH = Path("./docs").glob("*.pdf")
        self.chunks = []
        self.kScore = 0.9
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        self.db = None
    
    def load_chunks(self):
        for pdf in self.FILE_PATH:
            print("FILE PROCESSING: ", pdf)
            loader = PyPDFLoader(file_path=pdf)
            self.chunks.extend(loader.load())
        
        print(f"Chunks loaded...(Chunks: {len([doc for doc in self.chunks])})")
    
    def initDB(self):
        self.db = Chroma.from_documents(
            documents=[doc for doc in self.chunks],
            embedding=self.embeddings,
            persist_directory="./database"
        )
    
    def search(self, prompt: str, kValue: int) -> list[dict]:
        print("[DB] user prompt: ", prompt)
        result = self.db.similarity_search_with_score(query=prompt, k=kValue)
        
        documents = []
        
        for doc, score in result:
            if score <= self.kScore:
                # Score Key
                # 0.0 = identical
                # 0.2 = very similar
                # 0.8 = weak match
                # 1.5 = unrelated
                documents.append({
                    "content": doc.page_content,
                    "metadata":doc.metadata,
                    "score":score
                })
        
        if not documents:
            raise NoRelevantDocumentsError(f"No relevant document on topic: '{prompt}'.")
        
        return documents


class NoRelevantDocumentsError(Exception):
    """Raised when no relevant documents are found."""
    pass