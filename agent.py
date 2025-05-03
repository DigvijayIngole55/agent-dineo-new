"""SmolaGents Implementation"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from smolagents import Agent, Conversation, Reasoner

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage



#Calculation tools

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b

def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

def modulus(a: float, b: float) -> float:
    """Get the modulus of two numbers."""
    return a % b

class ToolAgent(Agent):
    """Agent that can use tools to solve problems."""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        
    def run(self, query: str) -> str:
        """Process a query using available tools."""
        # Basic implementation - in practice, you'd have
        # logic to determine which tool to use
        # response = self.llm.invoke(query)
        # return response.content
        try:
            response = self.llm.invoke([HumanMessage(content=query)])
            return response.content
        except Exception as e:
            print(f"Tool agent error: {e}")
            return f"I encountered an error: {str(e)}"

def get_llm(provider: str = "google"):
    if provider == "google":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "groq":
        return ChatGroq(model="qwen-qwq-32b", temperature=0)
    elif provider == "huggingface":
        return ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
            ),
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")

def create_agent(provider: str = "google"):
    """Create a SmolaGents application"""
    llm = get_llm(provider)
    
    # Define available tools
    tools = [
        multiply,
        add,
        subtract,
        divide,
        modulus,
    ]
    
    tool_agent = ToolAgent(llm, tools)
    
    # Create a conversation flow using reasoner
    reasoner = Reasoner([
        tool_agent
    ])

    return reasoner

if __name__ == "__main__":
    agent_app = create_agent("google")
    response = agent_app("What is the capital of France?")
    print(response)