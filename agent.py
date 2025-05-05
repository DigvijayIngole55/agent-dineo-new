"""SmolaGents Implementation"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, tool, ChatMessage, LiteLLMModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_groq import ChatGroq


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
def wiki_search(query: str) -> dict:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query.
    """
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        
        results = []
        for doc in search_docs:
            title = doc.metadata.get("title", "")
            source = doc.metadata.get("source", "")
            page_content = doc.page_content
            
            results.append({
                "title": title,
                "url": source,
                "content": page_content
            })
        
        if not results:
            return {"wiki_results": "No Wikipedia results found for the query: " + query}
        
        return {"wiki_results": results}
    
    except Exception as e:
        return {"wiki_results": f"Error searching Wikipedia: {str(e)}"}

@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo for a query and return results.
    
    Args:
        query: The search query.
    """
    search_tool = DuckDuckGoSearchTool()
    search_results = search_tool(query)
    
    
    return {"web_results": search_results}

def get_llm(provider: str = "google"):
    """Get language model based on provider"""
    if provider == "google":
        try:
            return LiteLLMModel(
                model_id="gemini/gemini-2.0-flash",
                api_key=os.getenv("GOOGLE_API_KEY")
            )
        except ImportError:
            print("litellm not available. Using HfApiModel instead.")
            from smolagents import HfApiModel
            return HfApiModel()
        except Exception as e:
            print(f"Error initializing Google model: {e}")
            print("Falling back to HfApiModel")
            from smolagents import HfApiModel
            return HfApiModel()
    elif provider == "huggingface":
        try:
            from smolagents import HfApiModel
            return HfApiModel()
        except ImportError:
            print("HfApiModel not available.")
            raise ImportError("No suitable model found")
    elif provider == "groq":
        try:
            return LiteLLMModel(
                model_id="groq/llama3-70b-8192",
                api_key=os.getenv("GROQ_API_KEY"),
                api_base="https://api.groq.com/openai/v1"
            )
        except Exception as e:
            print(f"Error initializing Groq model: {e}")
            print("Falling back to HfApiModel")
            from smolagents import HfApiModel
            return HfApiModel()
    else:
        raise ValueError("Invalid provider. Choose 'google', 'huggingface', or 'groq'.")

def create_agent(provider: str = "google"):
    """Create a SmolaGents application"""
    try:
        try:
            model = get_llm(provider)
        except:
            print(f"Failed to get model for {provider}, falling back to HfApiModel")
            from smolagents import HfApiModel
            model = HfApiModel()
        
        tools = [
            multiply,
            add,
            subtract,
            divide,
            modulus,
            web_search,
            wiki_search
        ]
        
        agent = ToolCallingAgent(tools=tools, model=model)
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