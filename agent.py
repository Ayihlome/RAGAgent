from vectorDB.vectorStorage import VectorDB
from langchain_ollama import OllamaLLM


# Startup DB
print("Starting up DB...")
vectorDB = VectorDB()
vectorDB.load_chunks()
vectorDB.initDB()

# Setup LangChain agent
model = OllamaLLM(
    model="gemma4:e2b",
    temperature=0.3
)
print("Model loaded...")

# context builder
prompt = "What is RollBacks?"
rag = vectorDB.search(prompt=prompt, kValue=3)

def contextBuilder(userPrompt: str, RAGresponse:list[dict]) -> str:
    sysPrompt = f"""You are a software engineer assistant that answers user questions, 
    in short and easy to understand paragraphs, based off of documents uploaded. 
    Here is the users question: {userPrompt}
    Here is the relevant information that you should use to answer the question: {[txt["content"] for txt in RAGresponse]}
    At the end of every response also include this information: 
    - page: {[txt["metadata"]["page_label"] for txt in RAGresponse]}
    - source: {[txt["metadata"]["source"] for txt in RAGresponse]}"""
    
    return sysPrompt

# call model

print("Processing response....")
response = model.invoke(contextBuilder(userPrompt=prompt, RAGresponse=rag))
print("Response:\n", response)