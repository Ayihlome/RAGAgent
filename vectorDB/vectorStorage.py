from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader


class VectorDB:
    def __init__(self):
        self.FILE_PATH = "docs\\CICIPipelineArticle.pdf"
        self.chunks = None
        self.kScore = 0.9
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        self.db = None
    
    def load_chunks(self):
        loader = PyPDFLoader(file_path=self.FILE_PATH)
        
        self.chunks = loader.load()
        print(f"Chunks loaded...(Chunks: {len(self.chunks)})")
    
    def initDB(self):
        self.db = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory="./database"
        )
    
    def search(self, prompt: str, kValue: int) -> list[dict]:
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
        
        return documents

# vDB = VectorDB()
# vDB.load_chunks()
# vDB.initDB()


# # Temp RAG testin
# results = vDB.search(
#     "What is  Multi-Layer Automated Testing",
#     kValue=3
# )

# for i, doc in enumerate(results):
#     print(f"\nResult {i+1}")
#     print(doc["content"])
#     print("Metadata: ",doc["metadata"])
#     print("Score: ",doc["score"])