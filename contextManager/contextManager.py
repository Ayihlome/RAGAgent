from langchain_core.messages import SystemMessage, HumanMessage


class ConversationManger:
    def __init__(self):
        self.messages = []
        self.max_messages = 8
        self.sysPrompt = f"""You are a software engineer assistant that answers user questions, 
        in short and easy to understand paragraphs, based off of documents uploaded. You will be given
        the context and then the question to which you must formulate a response based on the context. 
        Do not guess or fabricate a response of your own, If the context does not answer or relate to the
        question, simply say: "I do not have enough information on this topic"
        The format you should expect is as follows:
        Context:
        [information on a specific topic here]
        
        User Question:
        [the users question here]
        
        Formulate your answer in 2-4 short and simple paragraphs
        At the end of every response also include this information: 
        - Citation: Page[will be included in the Context metadata], Source [will be included in the Context metadata]"""
        
    def build_context(self, prompt: str, docs: list[dict]) -> list:
        messages = [
            SystemMessage(content=self.sysPrompt)
        ]
        
        messages.extend(self.messages)
        
        if len(self.messages) >= self.max_messages:
            # snip the 2 oldest messages
            del self.messages[:2]
        
        # add the new messages
        messages.append(
            HumanMessage(content=f"""Content:{[text for text in docs]}\nUser Question:{prompt}""")
        )
        
        return messages


