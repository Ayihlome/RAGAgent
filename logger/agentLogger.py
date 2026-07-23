from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import json
from langchain_ollama import OllamaLLM


class AgentLogger:
    def __init__(self):
        self.trace_id = uuid.uuid4()
        self.session_id = uuid.uuid4()
        self.messages = []
        
        
        self.retrievalDiagnostic = RetrieverDiagnostic(
            trace_id= self.trace_id,
            retrieval= []
        )
        self.conversation = Conversation(
            session_id= self.session_id
        )
        self.session = Session(
            session_id= self.session_id,
            trace_id= self.trace_id
        )
    
    def startSession(self):
        self.session.start_time = datetime.now()
    
    def endSession(self):
        self.session.end_time = datetime.now()
        
        model = OllamaLLM(
            model="gemma4:e2b",
            temperature=0.1
        )
        self.session.total_tokens = len(self.messages[0].split()) * 1.3
        self.session.token_usage = (self.session.total_tokens / 128000) * 100
    
    def retrievalCheck(self, prompt: str, finalresponse:str,  RAGresponse: list[dict]):
        self.conversation.user_prompt = prompt
        self.retrievalDiagnostic.user_prompt = prompt
        self.messages.append(prompt)
        self.conversation.final_response = finalresponse
        

        for txt in RAGresponse:
            self.retrievalDiagnostic.doc_tokens +=  len(self.messages[0].split()) * 1.3
        
        for txtObj in RAGresponse:
            self.retrievalDiagnostic.retrieval.append({
                "score": txtObj["score"],
                "page":txtObj["metadata"]["page_label"],
                "source":txtObj["metadata"]["source"]
            })
    
    def saveLog(self):
        trace = {
            "session_info":asdict(self.session),
            "conversation":asdict(self.conversation),
            "Retrieval Diagnostics":asdict(self.retrievalDiagnostic),
        }
        with open("log.json", "a+") as file:
            json.dump(trace, file, indent=4, default=str)
            file.write("\n")


@dataclass
class RetrieverDiagnostic:
    trace_id : int
    
    user_prompt : str | None = None
    doc_tokens : float = 0
    retrieval : list[dict] | None = None

@dataclass
class Session:
    session_id : int
    trace_id: int
    
    model: str = "gemma4:e2b"
    start_time : datetime | None = None
    end_time : datetime | None = None
    
    total_tokens : float = 0
    token_usage : float  = 0 # A percentage of the context window used for prompt and response

@dataclass
class Conversation:
    session_id : int
    
    user_prompt : str | None = None
    final_response : str | None = None
    # citation : str # or a dictionary of {"page": 3, "doc":"thisdoc.pdf"}