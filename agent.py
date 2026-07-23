from vectorDB.vectorStorage import VectorDB, NoRelevantDocumentsError
from langchain_ollama import OllamaLLM
from contextManager.contextManager import ConversationManger
from logger.agentLogger import AgentLogger

log = AgentLogger()
messages = ConversationManger()

log.startSession()
# Startup DB
print("Starting up DB...")
vectorDB = VectorDB()
vectorDB.load_chunks()
vectorDB.initDB()


# Setup LangChain agent
model = OllamaLLM(
    model="gemma4:e2b",
    temperature=0.1
)
print("Model loaded...\n")

while True:
    try:
        prompt = str(input("\n    >>"))
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    
    
    try:
        retrieved_docs = vectorDB.search(prompt=prompt, kValue=3)
    except NoRelevantDocumentsError:
        print("Sorry, I couldn't find information about that in the knowledge base.")
        continue
    
    # context builder
    context = messages.build_context(prompt=prompt, docs=retrieved_docs)
    print("Current Context: ", context)
    print("Processing response....\n")
    
    # response = model.invoke(context)
    chunks = model.stream(context)

    for chunk in chunks:
        print(chunk, end='', flush=True)
    
    

log.retrievalCheck(prompt=prompt, finalResponse=chunks, RAGresponse=retrieved_docs)
log.endSession()
log.saveLog()


