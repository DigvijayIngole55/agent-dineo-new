import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from functools import wraps
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, tool, ChatMessage, LiteLLMModel, CodeAgent, PythonInterpreterTool
from langchain_community.document_loaders import WikipediaLoader
import pandas as pd
import json
import uuid
import time
import numpy as np
from datetime import datetime
from urllib.parse import urlparse
import tempfile
import requests

load_dotenv()

prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
with open(prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()

def limit_calls(max_calls: int):
    """Decorator to limit function calls to max_calls."""
    def decorator(func):
        calls = {'count': 0}
        @wraps(func)
        def wrapper(*args, **kwargs):
            if calls['count'] >= max_calls:
                raise RuntimeError(
                    f"Call limit reached: {func.__name__} may only be called {max_calls} times"
                )
            calls['count'] += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

@tool
def download_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    
    Args:
        url: The URL to download from
        filename: Optional filename, will generate one based on URL if not provided
        
    Returns:
        Path to the downloaded file
    """
    try:
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                import uuid
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"
        
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return f"File downloaded to {filepath}. You can now process this file."
    except Exception as e:
        return f"Error downloading file: {str(e)}"

@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using pytesseract.
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text or error message
    """
    try:
        import pytesseract
        from PIL import Image
        
        image = Image.open(image_path)
        
        text = pytesseract.image_to_string(image)
        
        return f"Extracted text from image:\n\n{text}"
    except ImportError:
        return "Error: pytesseract is not installed."
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

@tool
def analyze_tabular_file(file_path: str) -> str:
    """
    Analyze a tabular file (CSV or Excel) using pandas.
    
    Args:
        file_path: Path to the CSV or Excel file
        
    Returns:
        Analysis result or error message
    """
    try:
        import pandas as pd
        import os
        
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        if file_extension in ['.csv', '.txt']:
            df = pd.read_csv(file_path)
            file_type = "CSV"
        elif file_extension in ['.xlsx', '.xls', '.xlsm']:
            df = pd.read_excel(file_path)
            file_type = "Excel"
        else:
            return f"Unsupported file extension: {file_extension}. Please provide a CSV or Excel file."
        
        result = f"{file_type} file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        result += "Summary statistics:\n"
        result += str(df.describe())
        
        return result
    
    except ImportError as e:
        if "openpyxl" in str(e):
            return "Error: openpyxl is not installed. Please install it with 'pip install openpyxl'."
        else:
            return "Error: pandas is not installed. Please install it with 'pip install pandas'."
    except Exception as e:
        return f"Error analyzing file: {str(e)}"

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
def arxiv_search(query: str, max_results: int = 3) -> dict:
    """Search arXiv for academic papers based on a query and return results.

    Args:
        query: The search query for academic papers.
        max_results: Maximum number of results to return (default: 3).
    """
    try:
        import arxiv
        
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=3,  
            num_retries=3
        )
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in client.results(search):
            paper_info = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else None,
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "categories": paper.categories
            }
            results.append(paper_info)
        
        if not results:
            return {"arxiv_results": "No arXiv papers found for the query: " + query}
        
        return {"arxiv_results": results}
    
    except ImportError:
        return {"arxiv_results": "Error: The arxiv package is not installed. Install it with 'pip install arxiv'."}
    except Exception as e:
        return {"arxiv_results": f"Error searching arXiv: {str(e)}"}

# @tool
# def web_search(query: str) -> dict:
#     """Search DuckDuckGo for a query and return results.

#     Args:
#         query: The search query.
#     """
#     search_tool = DuckDuckGoSearchTool()
#     search_results = search_tool(query)
    
#     return {"web_results": search_results}

_last_web_call = 0.0
_web_retries   = 5
_min_interval  = 2.0

@tool
def web_search(query: str) -> dict:
    """Rate-limited DuckDuckGo search (2 s min interval + 5× retries).
    
    Args:
        query: The search query string to look up on DuckDuckGo.
    """
    global _last_web_call

    # Throttle: ensure ≥2 s between actual HTTP calls
    elapsed = time.time() - _last_web_call
    if elapsed < _min_interval:
        time.sleep(_min_interval - elapsed)

    # Attempt + exponential back-off on rate-limit
    for attempt in range(_web_retries):
        try:
            ddg = DuckDuckGoSearchTool()
            results = ddg(query)
            _last_web_call = time.time()
            return {"web_results": results}

        except Exception as e:
            msg = str(e)
            if "202 Ratelimit" in msg and attempt < _web_retries - 1:
                backoff = (2 ** attempt) * _min_interval
                time.sleep(backoff)
                continue

            # Final failure
            return {"web_results": f"Search failed after {attempt+1} attempts: {msg}"}
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

def create_agent(provider: str = "google") -> ToolCallingAgent:
    model = get_llm(provider)
    tools = [
        wiki_search,
        arxiv_search,
        web_search,
        download_from_url,
        analyze_tabular_file,
        extract_text_from_image,
        PythonInterpreterTool()
    ]
    
    agent = ToolCallingAgent(tools=tools, model=model)
    agent.system_prompt = system_prompt
    print(agent.system_prompt)
    return agent

def check_search_results(response: str) -> bool:
    """Check if any search results were found in the response"""
    search_indicators = [
        "wiki_results", 
        "arxiv_results", 
        "web_results",
        "From Wikipedia",
        "From arXiv",
        "From DuckDuckGo"
    ]
    
    for indicator in search_indicators:
        if indicator in response:
            return True
    return False


if __name__ == "__main__":
    agent = create_agent("google")
    print("Agent created. Type 'exit' to quit.")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() in ("exit", "quit"):
            break
        
        try:
            print("\n[Searching and generating answer...]")
            
            resp = agent.run(query)
            print(f"\nResponse: {resp}")
                
        except Exception as e:
            print(f"\n[Error]: An unexpected error occurred: {str(e)}")
            print("Please try again or check your configuration.")