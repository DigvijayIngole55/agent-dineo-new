"""SmolaGents Implementation"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from smolagents import CodeAgent, DuckDuckGoSearchTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from smolagents import ChatMessage

load_dotenv()


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.
    
    Args:
        a: First number to multiply
        b: Second number to multiply
    """
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers.
    
    Args:
        a: First number to add
        b: Second number to add
    """
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers.
    
    Args:
        a: Number to subtract from
        b: Number to subtract
    """
    return a - b

@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers.
    
    Args:
        a: Numerator
        b: Denominator
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: float, b: float) -> float:
    """Get the modulus of two numbers.
    
    Args:
        a: First number
        b: Second number
    """
    return a % b

@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo for a query and return results.
    
    Args:
        query: The search query.
    """
    search_tool = DuckDuckGoSearchTool()
    search_results = search_tool(query)
    
    
    return {"web_results": search_results}

def get_llm(provider: str = "groq"):
    """Get language model based on provider"""
    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model="qwen-qwq-32b", temperature=0)
        except ImportError:
            print("langchain_groq not available. Using HfApiModel instead.")
            from smolagents import HfApiModel
            return HfApiModel()
    elif provider == "huggingface":
        try:
            from smolagents import HfApiModel
            return HfApiModel()
        except ImportError:
            print("HfApiModel not available.")
            raise ImportError("No suitable model found")
    else:
        raise ValueError("Invalid provider. Choose 'groq' or 'huggingface'.")
def create_agent(provider: str = "google"):
    """Create a SmolaGents application"""
    try:
        from smolagents import HfApiModel, CodeAgent
        
        try:
            model = get_llm(provider)
        except:
            print(f"Failed to get model for {provider}, falling back to HfApiModel")
            model = HfApiModel()
        
        tools = [
            multiply,
            add,
            subtract,
            divide,
            modulus,
            web_search
        ]
        
        agent = CodeAgent(tools=tools, model=model)
        return agent
        
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None

if __name__ == "__main__":
    try:
        agent_app = create_agent("google")
        if agent_app:
            print("Agent created successfully!")
            
            print("Type 'exit' to quit")
            while True:
                query = input("\nYour question: ")
                if query.lower() in ['exit', 'quit']:
                    break
                    
                try:
                    response = agent_app.run(query)
                    print(f"\nResponse: {response}")
                except Exception as e:
                    print(f"\nError running agent: {e}")
        else:
            print("Failed to create agent.")
    except Exception as e:
        print(f"Error: {e}")