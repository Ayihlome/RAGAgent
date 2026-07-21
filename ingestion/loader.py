from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = "docs\\CICIPipelineArticle.pdf"
# print(FILE_PATH)
# print(FILE_PATH.exists())
# print(FILE_PATH.is_file())

loader = PyPDFLoader(file_path=FILE_PATH)
# returns chunks

document = loader.load()

print("Chunks: \n")
for chunk in document:
    print(f"- {chunk.page_content=}\n")