import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from functools import wraps
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, tool, ChatMessage, LiteLLMModel, CodeAgent, PythonInterpreterTool
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import pandas as pd
import json
import uuid
import time
import numpy as np
from datetime import datetime
from urllib.parse import urlparse
import tempfile
import requests
from bs4 import BeautifulSoup

# Web search rate limiting and retry parameters
_web_retries = 3
_min_interval = 1.0  # Minimum time between web search calls in seconds
_last_web_call = 0   # Last time a web search was made

print("[DEBUG] Loading environment variables...")
load_dotenv()

prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
print(f"[DEBUG] Loading system prompt from: {prompt_path}")
with open(prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()
print(f"[DEBUG] System prompt loaded, length: {len(system_prompt)} characters")

def limit_calls(max_calls: int):
    """Decorator to limit function calls to max_calls."""
    print(f"[DEBUG] Setting up limit_calls decorator with max_calls={max_calls}")
    def decorator(func):
        calls = {'count': 0}
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[DEBUG] Function {func.__name__} call count: {calls['count']}/{max_calls}")
            if calls['count'] >= max_calls:
                print(f"[DEBUG] Call limit reached for {func.__name__}: {calls['count']}/{max_calls}")
                raise RuntimeError(
                    f"Call limit reached: {func.__name__} may only be called {max_calls} times"
                )
            calls['count'] += 1
            print(f"[DEBUG] Executing {func.__name__} (call {calls['count']}/{max_calls})")
            result = func(*args, **kwargs)
            print(f"[DEBUG] Function {func.__name__} execution completed")
            return result
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
    print(f"[DEBUG] download_from_url called with URL: {url}, filename: {filename}")
    try:
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                import uuid
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"
            print(f"[DEBUG] Generated filename: {filename}")
        
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        print(f"[DEBUG] File will be saved to: {filepath}")
        
        print(f"[DEBUG] Sending HTTP request to: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        print(f"[DEBUG] HTTP request successful, status code: {response.status_code}")
        
        with open(filepath, 'wb') as f:
            print(f"[DEBUG] Downloading and writing file content...")
            chunks_count = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                chunks_count += 1
            print(f"[DEBUG] Download complete, wrote {chunks_count} chunks")
        
        print(f"[DEBUG] File successfully downloaded to {filepath}")
        return f"File downloaded to {filepath}. You can now process this file."
    except Exception as e:
        print(f"[DEBUG] Error in download_from_url: {str(e)}")
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
    print(f"[DEBUG] extract_text_from_image called with path: {image_path}")
    try:
        import pytesseract
        from PIL import Image
        
        print(f"[DEBUG] Opening image from: {image_path}")
        image = Image.open(image_path)
        print(f"[DEBUG] Image opened successfully, size: {image.size}, format: {image.format}")
        
        print(f"[DEBUG] Extracting text using pytesseract...")
        text = pytesseract.image_to_string(image)
        print(f"[DEBUG] Text extraction completed, extracted {len(text)} characters")
        
        return f"Extracted text from image:\n\n{text}"
    except ImportError as e:
        print(f"[DEBUG] ImportError in extract_text_from_image: {str(e)}")
        return "Error: pytesseract is not installed."
    except Exception as e:
        print(f"[DEBUG] Error in extract_text_from_image: {str(e)}")
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
    print(f"[DEBUG] analyze_tabular_file called with path: {file_path}")
    try:
        import pandas as pd
        import os
        
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        print(f"[DEBUG] File extension detected: {file_extension}")
        
        if file_extension in ['.csv', '.txt']:
            print(f"[DEBUG] Loading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            file_type = "CSV"
        elif file_extension in ['.xlsx', '.xls', '.xlsm']:
            print(f"[DEBUG] Loading Excel file: {file_path}")
            df = pd.read_excel(file_path)
            file_type = "Excel"
        else:
            print(f"[DEBUG] Unsupported file extension: {file_extension}")
            return f"Unsupported file extension: {file_extension}. Please provide a CSV or Excel file."
        
        print(f"[DEBUG] File loaded successfully. Shape: {df.shape}")
        result = f"{file_type} file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        print(f"[DEBUG] Generating summary statistics...")
        result += "Summary statistics:\n"
        result += str(df.describe())
        
        return result
    
    except ImportError as e:
        print(f"[DEBUG] ImportError in analyze_tabular_file: {str(e)}")
        if "openpyxl" in str(e):
            return "Error: openpyxl is not installed. Please install it with 'pip install openpyxl'."
        else:
            return "Error: pandas is not installed. Please install it with 'pip install pandas'."
    except Exception as e:
        print(f"[DEBUG] Error in analyze_tabular_file: {str(e)}")
        return f"Error analyzing file: {str(e)}"

@tool
def wiki_search(query: str) -> dict:
    """Search Wikipedia for a query and return maximum 2 results.

    Args:
        query: The search query.
    """
    print(f"[DEBUG] wiki_search called with query: {query}")
    try:
        print(f"[DEBUG] Using WikipediaLoader with max_docs=2")
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        print(f"[DEBUG] WikipediaLoader returned {len(search_docs)} documents")
        
        results = []
        for i, doc in enumerate(search_docs):
            title = doc.metadata.get("title", "")
            source = doc.metadata.get("source", "")
            page_content = doc.page_content
            print(f"[DEBUG] Document {i+1}: title='{title}', source='{source}', content length={len(page_content)}")
            
            results.append({
                "title": title,
                "url": source,
                "content": page_content
            })
        
        if not results:
            print(f"[DEBUG] No Wikipedia results found for query: {query}")
            return {"wiki_results": "No Wikipedia results found for the query: " + query}
        
        print(f"[DEBUG] Returning {len(results)} Wikipedia results")
        return {"wiki_results": results}
    
    except Exception as e:
        print(f"[DEBUG] Error in wiki_search: {str(e)}")
        return {"wiki_results": f"Error searching Wikipedia: {str(e)}"}

@tool
def arxiv_search(query: str, max_results: int = 3) -> dict:
    """Search arXiv for academic papers based on a query and return results.

    Args:
        query: The search query for academic papers.
        max_results: Maximum number of results to return (default: 3).
    """
    print(f"[DEBUG] arxiv_search called with query: {query}, max_results: {max_results}")
    try:
        import arxiv
        
        print(f"[DEBUG] Creating arxiv Client with page_size={max_results}")
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=3,  
            num_retries=3
        )
        
        print(f"[DEBUG] Creating arxiv Search for query: {query}")
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        print(f"[DEBUG] Fetching search results...")
        result_count = 0
        for paper in client.results(search):
            result_count += 1
            print(f"[DEBUG] Processing paper {result_count}: {paper.title}")
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
            print(f"[DEBUG] No arXiv papers found for query: {query}")
            return {"arxiv_results": "No arXiv papers found for the query: " + query}
        
        print(f"[DEBUG] Returning {len(results)} arXiv results")
        return {"arxiv_results": results}
    
    except ImportError as e:
        print(f"[DEBUG] ImportError in arxiv_search: {str(e)}")
        return {"arxiv_results": "Error: The arxiv package is not installed. Install it with 'pip install arxiv'."}
    except Exception as e:
        print(f"[DEBUG] Error in arxiv_search: {str(e)}")
        return {"arxiv_results": f"Error searching arXiv: {str(e)}"}
@tool
def web_search(query: str) -> dict:
    """Search Serper API for a query and return results.

    Args:
        query: The search query.
    """
    print(f"[DEBUG] web_search called with query: {query}")
    try:
        # Get Serper API key from environment variables
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            print(f"[DEBUG] SERPER_API_KEY not found in environment variables")
            return {"web_results": "Error: SERPER_API_KEY not found in environment variables"}
        
        # Add rate limiting
        global _last_web_call
        elapsed = time.time() - _last_web_call
        if elapsed < _min_interval:
            sleep_time = _min_interval - elapsed
            print(f"[DEBUG] Rate limiting - sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        # Execute the search with retries
        for attempt in range(_web_retries):
            try:
                print(f"[DEBUG] Serper search attempt {attempt+1}/{_web_retries}")
                
                # Make request to Serper API
                headers = {
                    'X-API-KEY': serper_api_key,
                    'Content-Type': 'application/json'
                }
                payload = json.dumps({
                    "q": query,
                    "num": 5  # Number of results to return
                })
                
                response = requests.post('https://google.serper.dev/search', 
                                       headers=headers, 
                                       data=payload)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                search_results = response.json()
                _last_web_call = time.time()
                print(f"[DEBUG] Search successful, updating last_web_call to {_last_web_call}")
                
                # Format the results in a readable way
                formatted_results = format_serper_results(search_results)
                
                if not formatted_results:
                    return {"web_results": "No search results found."}
                    
                return {"web_results": formatted_results}
                
            except requests.exceptions.RequestException as e:
                msg = str(e)
                print(f"[DEBUG] Search attempt {attempt+1} failed with error: {msg}")
                if "429" in msg and attempt < _web_retries - 1:  # 429 is the status code for rate limiting
                    backoff = (2 ** attempt) * _min_interval
                    print(f"[DEBUG] Rate limit hit, backing off for {backoff:.2f}s")
                    time.sleep(backoff)
                    continue
                
                # Final failure
                if attempt == _web_retries - 1:
                    print(f"[DEBUG] All attempts failed, returning error message")
                    return {"web_results": f"Search failed after {attempt+1} attempts: {msg}"}
                
    except Exception as e:
        print(f"[DEBUG] Error in web_search: {str(e)}")
        return {"web_results": f"Search error: {str(e)}"}

def format_serper_results(results):
    """Format Serper API results into a readable string."""
    formatted = ""
    
    # Process organic results
    if "organic" in results:
        for i, result in enumerate(results["organic"]):
            title = result.get("title", "No Title")
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            
            formatted += f"{i+1}. {title}\n"
            formatted += f"   URL: {link}\n"
            formatted += f"   {snippet}\n\n"
    
    # Process knowledge graph if present
    if "knowledgeGraph" in results:
        kg = results["knowledgeGraph"]
        title = kg.get("title", "")
        description = kg.get("description", "")
        if title:
            formatted += f"Knowledge Graph: {title}\n"
            if description:
                formatted += f"{description}\n\n"
    
    # Process related searches if present
    if "relatedSearches" in results and results["relatedSearches"]:
        formatted += "Related Searches:\n"
        for i, related in enumerate(results["relatedSearches"][:5]):  # Limit to 5 related searches
            formatted += f"- {related.get('query', '')}\n"
    
    return formatted.strip()

def get_llm(provider: str = "google"):
    """Get language model based on provider"""
    print(f"[DEBUG] get_llm called with provider: {provider}")
    if provider == "google":
        try:
            print(f"[DEBUG] Attempting to initialize Google's Gemini model")
            api_key = os.getenv("GOOGLE_API_KEY")
            print(f"[DEBUG] API key present: {bool(api_key)}")
            model = LiteLLMModel(
                model_id="gemini/gemini-2.0-flash",
                api_key=api_key
            )
            print(f"[DEBUG] Google Gemini model initialized successfully")
            return model
        except ImportError as e:
            print(f"[DEBUG] ImportError initializing Google model: {e}")
            print("litellm not available. Using HfApiModel instead.")
            from smolagents import HfApiModel
            print(f"[DEBUG] Falling back to HfApiModel")
            return HfApiModel()
        except Exception as e:
            print(f"[DEBUG] Error initializing Google model: {e}")
            print("Falling back to HfApiModel")
            from smolagents import HfApiModel
            print(f"[DEBUG] Falling back to HfApiModel")
            return HfApiModel()
    elif provider == "huggingface":
        try:
            print(f"[DEBUG] Initializing HuggingFace model")
            from smolagents import HfApiModel
            model = HfApiModel()
            print(f"[DEBUG] HuggingFace model initialized successfully")
            return model
        except ImportError as e:
            print(f"[DEBUG] ImportError initializing HuggingFace model: {e}")
            print("HfApiModel not available.")
            raise ImportError("No suitable model found")
    elif provider == "groq":
        try:
            print(f"[DEBUG] Initializing Groq model")
            api_key = os.getenv("GROQ_API_KEY")
            print(f"[DEBUG] API key present: {bool(api_key)}")
            model = LiteLLMModel(
                model_id="groq/llama3-70b-8192",
                api_key=api_key,
                api_base="https://api.groq.com/openai/v1"
            )
            print(f"[DEBUG] Groq model initialized successfully")
            return model
        except Exception as e:
            print(f"[DEBUG] Error initializing Groq model: {e}")
            print("Falling back to HfApiModel")
            from smolagents import HfApiModel
            print(f"[DEBUG] Falling back to HfApiModel")
            return HfApiModel()
    else:
        print(f"[DEBUG] Invalid provider: {provider}")
        raise ValueError("Invalid provider. Choose 'google', 'huggingface', or 'groq'.")

def create_agent(provider: str = "google") -> ToolCallingAgent:
    print(f"[DEBUG] create_agent called with provider: {provider}")
    print(f"[DEBUG] Getting LLM for provider: {provider}")
    model = get_llm(provider)
    print(f"[DEBUG] LLM initialized")
    
    print(f"[DEBUG] Setting up tools")
    tools = [
        wiki_search,
        arxiv_search,
        web_search,
        download_from_url,
        analyze_tabular_file,
        extract_text_from_image,
        PythonInterpreterTool()
    ]
    print(f"[DEBUG] Configured {len(tools)} tools")
    
    print(f"[DEBUG] Creating ToolCallingAgent")
    agent = ToolCallingAgent(tools=tools, model=model)
    agent.system_prompt = system_prompt
    print(f"[DEBUG] System prompt set, agent created")
    print(agent.system_prompt)
    return agent

def check_search_results(response: str) -> bool:
    """Check if any search results were found in the response"""
    print(f"[DEBUG] check_search_results called for response of length: {len(response)}")
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
            print(f"[DEBUG] Search indicator found: '{indicator}'")
            return True
    print(f"[DEBUG] No search indicators found in response")
    return False


if __name__ == "__main__":
    print(f"[DEBUG] Starting main execution")
    print(f"[DEBUG] Creating agent with Google provider")
    agent = create_agent("google")
    print("Agent created. Type 'exit' to quit.")
    
    while True:
        print(f"[DEBUG] Waiting for user input")
        query = input("\nYour question: ")
        print(f"[DEBUG] Received query: '{query}'")
        
        if query.lower() in ("exit", "quit"):
            print(f"[DEBUG] Exit command detected, terminating")
            break
        
        try:
            print("\n[Searching and generating answer...]")
            print(f"[DEBUG] Running agent with query: '{query}'")
            
            start_time = time.time()
            resp = agent.run(query)
            end_time = time.time()
            print(f"[DEBUG] Agent run completed in {end_time - start_time:.2f} seconds")
            print(f"[DEBUG] Response length: {len(resp)}")
            
            print(f"\nResponse: {resp}")
                
        except Exception as e:
            print(f"[DEBUG] Exception in agent.run: {type(e).__name__}: {str(e)}")
            print(f"\n[Error]: An unexpected error occurred: {str(e)}")
            print("Please try again or check your configuration.")