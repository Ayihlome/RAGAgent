from langchain_docling.loader import DoclingLoader

FILE_PATH = "docs\CI\CICDPipelineResearchPaper.pdf"


loader = DoclingLoader(file_path=FILE_PATH) 
# returns chunks

document = loader.load()

for chunk in document:
    print(f"- {chunk.page_content=}")