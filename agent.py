import os
import tempfile
import time
import re
import json
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool, Tool
from langchain_community.document_loaders import WikipediaLoader
import google.generativeai as genai
from pydantic import Field

# Set up debug flag - set to True to enable detailed debugging
DEBUG = True

def debug_print(message):
    """Helper function to print debug messages when DEBUG is True"""
    if DEBUG:
        print(f"[DEBUG] {message}")

debug_print("Starting agent initialization")

# Load environment variables
load_dotenv()
debug_print("Environment variables loaded")

# Load system prompt from environment variable or file
system_prompt = os.getenv("SYSTEM_PROMPT")
if system_prompt:
    print(f"System prompt loaded from environment variable, length: {len(system_prompt)} characters")
else:
    system_prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
    print(f"Loading system prompt from file: {system_prompt_path}")
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        print(f"System prompt loaded from file, length: {len(system_prompt)} characters")
    except Exception as e:
        print(f"Error loading system prompt: {str(e)}")
        system_prompt = "You are a helpful assistant."
        debug_print(f"Using default system prompt due to error: {str(e)}")

class WebSearchTool:
    """Tool for performing web searches with rate limiting and retries with fallback to Serper API."""
    
    def __init__(self):
        self.name = "web_search"
        self.description = "Search the web for a query and return results."
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds
        self.max_retries = 3
        # Check if Serper API key is available
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        if self.serper_api_key:
            debug_print("Serper API key found, will use as fallback or primary search provider")
        else:
            debug_print("No Serper API key found, will only use DuckDuckGo")
    
    def __call__(self, query: str) -> str:
        """Perform web search with rate limiting and retries."""
        debug_print(f"Web searching for: {query}")
        
        # If Serper API key is available and specified as primary, use it directly
        if self.serper_api_key and os.getenv("USE_SERPER_PRIMARY", "false").lower() == "true":
            debug_print("Using Serper as primary search provider")
            return self._serper_search(query)
        
        # Otherwise try DuckDuckGo first with fallback to Serper
        for attempt in range(self.max_retries):
            # Implement rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                debug_print(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            try:
                # Try DuckDuckGo first
                debug_print("Initializing DuckDuckGoSearchAPIWrapper")
                search = DuckDuckGoSearchAPIWrapper(max_results=5)
                debug_print("Executing DuckDuckGo search")
                results = search.run(query)
                
                if not results or results.strip() == "":
                    debug_print("No DuckDuckGo search results found")
                    if self.serper_api_key:
                        debug_print("Falling back to Serper API")
                        return self._serper_search(query)
                    return {"web_results": "No search results found."}
                    
                debug_print(f"Web search returned results (length: {len(results)})")
                self.last_request_time = time.time()
                return {"web_results": results}
            
            except Exception as e:
                debug_print(f"DuckDuckGo search error in attempt {attempt+1}: {str(e)}")
                if self.serper_api_key:
                    debug_print("Falling back to Serper API due to DuckDuckGo error")
                    return self._serper_search(query)
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * self.min_request_interval
                    debug_print(f"Waiting {wait_time:.2f}s before retry")
                    time.sleep(wait_time)
                else:
                    return {"web_results": f"Search error after {self.max_retries} attempts: {str(e)}"}
        
        # If we get here and have Serper as fallback, try it
        if self.serper_api_key:
            debug_print("Falling back to Serper API after all DuckDuckGo retries failed")
            return self._serper_search(query)
            
        return {"web_results": "Search failed due to rate limiting"}
    
    def _serper_search(self, query: str) -> dict:
        """Perform search using Serper API."""
        debug_print(f"Searching with Serper API: {query}")
        try:
            url = "https://google.serper.dev/search"
            payload = json.dumps({
                "q": query,
                "gl": "us",
                "hl": "en",
                "num": 5
            })
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            
            search_results = response.json()
            
            # Format results
            formatted_results = []
            
            # Extract organic results
            if "organic" in search_results:
                for result in search_results["organic"][:5]:
                    title = result.get("title", "")
                    link = result.get("link", "")
                    snippet = result.get("snippet", "")
                    formatted_results.append(f"[{title}]({link})\n{snippet}\n")
            
            # Extract knowledge graph if available
            if "knowledgeGraph" in search_results:
                kg = search_results["knowledgeGraph"]
                title = kg.get("title", "")
                type_text = kg.get("type", "")
                description = kg.get("description", "")
                formatted_results.append(f"Knowledge Graph: {title} ({type_text})\n{description}\n")
            
            if not formatted_results:
                return {"web_results": "No search results found from Serper."}
                
            return {"web_results": "## Search Results\n\n" + "\n".join(formatted_results)}
            
        except Exception as e:
            debug_print(f"Serper API search error: {str(e)}")
            return {"web_results": f"Serper search error: {str(e)}"}

class WikiSearchTool:
    """Tool for searching Wikipedia with rate limiting."""
    
    def __init__(self):
        self.name = "wiki_search"
        self.description = "Search Wikipedia for a query and return maximum 2 results."
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds
        self.max_calls = 3
        self.call_count = 0
    
    def __call__(self, query: str) -> str:
        """Search Wikipedia for a query."""
        debug_print(f"Searching Wikipedia for: {query}")
        
        # Check call limit
        self.call_count += 1
        if self.call_count > self.max_calls:
            debug_print(f"Call limit reached for wiki_search: {self.call_count}/{self.max_calls}")
            return {"wiki_results": f"Call limit reached: wiki_search may only be called {self.max_calls} times"}
        
        # Implement rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            debug_print(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        try:
            debug_print("Initializing WikipediaLoader")
            search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
            debug_print(f"Found {len(search_docs)} Wikipedia documents")
            
            results = []
            for i, doc in enumerate(search_docs):
                title = doc.metadata.get("title", "")
                source = doc.metadata.get("source", "")
                page_content = doc.page_content
                
                debug_print(f"Wiki result {i+1}: Title={title}, URL={source}, Content length={len(page_content)}")
                
                results.append({
                    "title": title,
                    "url": source,
                    "content": page_content
                })
            
            if not results:
                debug_print("No Wikipedia results found")
                return {"wiki_results": "No Wikipedia results found for the query: " + query}
            
            self.last_request_time = time.time()
            return {"wiki_results": results}
        
        except Exception as e:
            debug_print(f"Error searching Wikipedia: {str(e)}")
            return {"wiki_results": f"Error searching Wikipedia: {str(e)}"}

def download_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    
    Args:
        url: The URL to download from
        filename: Optional filename, will generate one based on URL if not provided
        
    Returns:
        Path to the downloaded file
    """
    debug_print(f"Attempting to download from URL: {url}, filename: {filename}")
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                # Generate a random name if we couldn't extract one
                import uuid
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"
            debug_print(f"Generated filename: {filename}")
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        debug_print(f"Saving to filepath: {filepath}")
        
        # Download the file
        debug_print(f"Starting download request for {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        debug_print(f"Request status code: {response.status_code}")
        
        # Save the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        debug_print(f"File successfully downloaded to {filepath}")
        return f"File downloaded to {filepath}. You can now process this file."
    except Exception as e:
        debug_print(f"Error downloading file: {str(e)}")
        return f"Error downloading file: {str(e)}"

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using pytesseract.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text or error message
    """
    debug_print(f"Attempting to extract text from image: {image_path}")
    try:
        # Try to import pytesseract
        import pytesseract
        from PIL import Image
        
        debug_print("Successfully imported pytesseract and PIL")
        
        # Open the image
        debug_print(f"Opening image file: {image_path}")
        image = Image.open(image_path)
        debug_print(f"Image opened: {image.format}, size={image.size}, mode={image.mode}")
        
        # Extract text
        debug_print("Starting OCR text extraction")
        text = pytesseract.image_to_string(image)
        debug_print(f"Extracted text length: {len(text)}")
        
        return f"Extracted text from image:\n\n{text}"
    except ImportError:
        debug_print("pytesseract not installed")
        return "Error: pytesseract is not installed. Please install it with 'pip install pytesseract' and ensure Tesseract OCR is installed on your system."
    except Exception as e:
        debug_print(f"Error extracting text from image: {str(e)}")
        return f"Error extracting text from image: {str(e)}"

def analyze_tabular_file(file_path: str) -> str:
    """
    Analyze a tabular file (CSV or Excel) using pandas.
    
    Args:
        file_path: Path to the CSV or Excel file
        
    Returns:
        Analysis result or error message
    """
    debug_print(f"Attempting to analyze tabular file: {file_path}")
    try:
        import pandas as pd
        import os
        
        debug_print("Successfully imported pandas")
        
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        debug_print(f"File extension: {file_extension}")
        
        if file_extension in ['.csv', '.txt']:
            debug_print(f"Reading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            file_type = "CSV"
        elif file_extension in ['.xlsx', '.xls', '.xlsm']:
            debug_print(f"Reading Excel file: {file_path}")
            df = pd.read_excel(file_path)
            file_type = "Excel"
        else:
            debug_print(f"Unsupported file extension: {file_extension}")
            return f"Unsupported file extension: {file_extension}. Please provide a CSV or Excel file."
        
        debug_print(f"File loaded successfully. Shape: {df.shape}")
        
        result = f"{file_type} file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        result += "Summary statistics:\n"
        result += str(df.describe())
        
        return result
    
    except ImportError as e:
        debug_print(f"Import error: {str(e)}")
        if "openpyxl" in str(e):
            return "Error: openpyxl is not installed. Please install it with 'pip install openpyxl'."
        else:
            return "Error: pandas is not installed. Please install it with 'pip install pandas'."
    except Exception as e:
        debug_print(f"Error analyzing file: {str(e)}")
        return f"Error analyzing file: {str(e)}"
    
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a temporary file and return the path.
    Useful for processing files from the GAIA API.
    
    Args:
        content: The content to save to the file
        filename: Optional filename, will generate a random name if not provided
        
    Returns:
        Path to the saved file
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)
    
    # Write content to the file
    with open(filepath, 'w') as f:
        f.write(content)
    
    return f"File saved to {filepath}. You can read this file to process its contents."

def open_and_read_file(file_path: str) -> str:
    """
    Open and read the contents of a file.
    
    Args:
        file_path: Path to the file to be read
        
    Returns:
        File contents as a string or error message
    """
    debug_print(f"Attempting to open and read file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        debug_print(f"Successfully read file, content length: {len(content)}")
        return content
    except UnicodeDecodeError:
        debug_print("Unicode decode error, trying binary mode")
        try:
            # Try binary mode for non-text files
            with open(file_path, 'rb') as file:
                binary_content = file.read()
            return f"Binary file read successfully. File size: {len(binary_content)} bytes"
        except Exception as e:
            debug_print(f"Error reading file in binary mode: {str(e)}")
            return f"Error reading file in binary mode: {str(e)}"
    except Exception as e:
        debug_print(f"Error reading file: {str(e)}")
        return f"Error reading file: {str(e)}"


def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and answer a question about it.
    
    Args:
        file_path: Path to the CSV file
        query: Question about the data
        
    Returns:
        Analysis result or error message
    """
    try:
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Run various analyses based on the query
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())
        
        return result
    except ImportError:
        return "Error: pandas is not installed. Please install it with 'pip install pandas'."
    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"


def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    
    Args:
        file_path: Path to the Excel file
        query: Question about the data
        
    Returns:
        Analysis result or error message
    """
    try:
        import pandas as pd
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Run various analyses based on the query
        result = f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())
        
        return result
    except ImportError:
        return "Error: pandas and openpyxl are not installed. Please install them with 'pip install pandas openpyxl'."
    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"

def arxiv_search(query: str, max_results: int = 3) -> dict:
    """Search arXiv for academic papers based on a query and return results.

    Args:
        query: The search query for academic papers.
        max_results: Maximum number of results to return (default: 3).
    """
    debug_print(f"Searching arXiv for: {query}, max_results={max_results}")
    try:
        import arxiv
        
        debug_print("Successfully imported arxiv package")
        
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=3,  
            num_retries=3
        )
        
        debug_print(f"Created arXiv client with page_size={max_results}")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        debug_print("Executing arXiv search")
        
        results = []
        for i, paper in enumerate(client.results(search)):
            debug_print(f"Processing arXiv result {i+1}: {paper.title}")
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
        
        debug_print(f"Found {len(results)} arXiv papers")
        
        if not results:
            return {"arxiv_results": "No arXiv papers found for the query: " + query}
        
        return {"arxiv_results": results}
    
    except ImportError:
        debug_print("arXiv package not installed")
        return {"arxiv_results": "Error: The arxiv package is not installed. Install it with 'pip install arxiv'."}
    except Exception as e:
        debug_print(f"Error searching arXiv: {str(e)}")
        return {"arxiv_results": f"Error searching arXiv: {str(e)}"}

def analyze_video(url: str) -> str:
    """Analyze video content using Gemini's video understanding capabilities."""
    debug_print(f"Attempting to analyze video from URL: {url}")
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            debug_print(f"Invalid URL format: {url}")
            return "Please provide a valid video URL with http:// or https:// prefix."
        
        # Check if it's a YouTube URL
        if 'youtube.com' not in url and 'youtu.be' not in url:
            debug_print(f"Not a YouTube URL: {url}")
            return "Only YouTube videos are supported at this time."

        try:
            # Configure yt-dlp with minimal extraction
            import yt_dlp
            debug_print("Successfully imported yt-dlp")
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'no_playlist': True,
                'youtube_include_dash_manifest': False
            }

            debug_print(f"yt-dlp options: {ydl_opts}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    # Try basic info extraction
                    debug_print(f"Extracting info for {url}")
                    info = ydl.extract_info(url, download=False, process=False)
                    if not info:
                        debug_print("Could not extract video information")
                        return "Could not extract video information."

                    title = info.get('title', 'Unknown')
                    description = info.get('description', '')
                    
                    debug_print(f"Video title: {title}")
                    debug_print(f"Description length: {len(description) if description else 0}")
                    
                    # Create results summary
                    result = f"YouTube Video Analysis:\n"
                    result += f"Title: {title}\n"
                    result += f"URL: {url}\n"
                    if description:
                        result += f"Description: {description}\n"
                    
                    return result

                except Exception as e:
                    debug_print(f"Error in yt-dlp extraction: {str(e)}")
                    if 'Sign in to confirm' in str(e):
                        return "This video requires age verification or sign-in. Please provide a different video URL."
                    return f"Error accessing video: {str(e)}"

        except ImportError:
            debug_print("yt-dlp not installed")
            return "Error: yt-dlp is not installed. Please install it with 'pip install yt-dlp'."
        except Exception as e:
            debug_print(f"Error with yt-dlp: {str(e)}")
            return f"Error extracting video info: {str(e)}"

    except Exception as e:
        debug_print(f"Error analyzing video: {str(e)}")
        return f"Error analyzing video: {str(e)}"

class GeminiAgent:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        debug_print(f"Initializing GeminiAgent with model: {model_name}")
        # Suppress warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*will be deprecated.*")
        warnings.filterwarnings("ignore", "LangChain.*")
        
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        debug_print("Configuring Gemini API")
        genai.configure(api_key=api_key)
        
        # Initialize the LLM
        debug_print("Setting up LLM")
        self.llm = self._setup_llm()
        
        # Setup tools
        debug_print("Setting up tools")
        self.tools = self._setup_tools()
        
        # Setup memory
        debug_print("Setting up conversation memory")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize agent
        debug_print("Setting up agent")
        self.agent = self._setup_agent()
        debug_print("GeminiAgent initialization complete")

    def run(self, query: str) -> str:
        """Run the agent on a query with incremental retries."""
        debug_print(f"Running agent with query: {query}")
        max_retries = 3
        base_sleep = 1  # Start with 1 second sleep
        
        for attempt in range(max_retries):
            try:
                debug_print(f"Attempt {attempt + 1}/{max_retries}")
                response = self.agent.run(query)
                debug_print(f"Agent response received (length: {len(response)})")
                return response

            except Exception as e:
                debug_print(f"Error in attempt {attempt + 1}: {str(e)}")
                sleep_time = base_sleep * (attempt + 1)  # Incremental sleep: 1s, 2s, 3s
                if attempt < max_retries - 1:
                    debug_print(f"Retrying in {sleep_time} seconds...")
                    print(f"Attempt {attempt + 1} failed. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                debug_print(f"All {max_retries} attempts failed")
                return f"Error processing query after {max_retries} attempts: {str(e)}"

    def _setup_llm(self):
        """Set up the language model."""
        debug_print(f"Setting up {self.model_name} LLM")
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0,
                max_output_tokens=2000,
                convert_system_message_to_human=True,
                system=system_prompt
            )
            debug_print("LLM setup successful")
            return llm
        except Exception as e:
            debug_print(f"Error setting up LLM: {str(e)}")
            raise
        
    def _setup_tools(self):
        """Set up the tools for the agent."""
        debug_print("Setting up agent tools")
        
        # Create instances of class-based tools
        wiki_search_tool = WikiSearchTool()
        web_search_tool = WebSearchTool()
        
        tools = [
            Tool(
                name=wiki_search_tool.name,
                func=wiki_search_tool,
                description=wiki_search_tool.description
            ),
            Tool(
                name=web_search_tool.name,
                func=web_search_tool,
                description=web_search_tool.description
            ),
            Tool(
                name="arxiv_search",
                func=arxiv_search,
                description="Search arXiv for academic papers based on a query and return results."
            ),
            Tool(
                name="download_from_url",
                func=download_from_url,
                description="Download a file from a URL"
            ),
            Tool(
                name="extract_text_from_image",
                func=extract_text_from_image,
                description="Extract text from an image"
            ),
            Tool(
                name="analyze_tabular_file",
                func=analyze_tabular_file,
                description="Analyze a tabular file (CSV or Excel)"
            ),
            Tool(
                name="analyze_video",
                func=analyze_video,
                description="Analyze YouTube video content"
            ),
            Tool(
            name="open_and_read_file",
            func=open_and_read_file,
            description="Open and read the contents of a file"
        ),
        Tool(
            name="save_and_read_file",
            func=save_and_read_file,
            description="Save content to a file and return the path"
        ),
        Tool(
            name="analyze_csv_file",
            func=analyze_csv_file,
            description="Analyze a CSV file using pandas"
        ),
        Tool(
            name="analyze_excel_file",
            func=analyze_excel_file,
            description="Analyze an Excel file using pandas"
        ),
        ]
        debug_print(f"Set up {len(tools)} tools")
        return tools
        
    def _setup_agent(self) -> AgentExecutor:
        """Set up the agent with tools and system message."""
        debug_print("Setting up agent executor")
        try:
            # Initialize agent executor with no prefix/suffix/safety settings
            agent = initialize_agent(
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                tools=self.tools,
                llm=self.llm,
                verbose=True,
                memory=self.memory,
                max_iterations=5,
                handle_parsing_errors=True,
                return_only_outputs=True
            )
            debug_print("Agent executor setup successful")
            return agent
        except Exception as e:
            debug_print(f"Error setting up agent: {str(e)}")
            raise

def create_agent(provider: str = "google") -> GeminiAgent:
    """
    Create an agent using the specified provider.
    Currently only supports Google's Gemini.
    
    Args:
        provider: The provider to use (only "google" is supported)
        
    Returns:
        A GeminiAgent instance
    """
    debug_print(f"Creating agent with provider: {provider}")
    if provider.lower() != "google":
        debug_print(f"Provider {provider} not supported, defaulting to Google Gemini")
        print(f"Warning: Provider {provider} not supported. Using Google Gemini.")
    
    # Get the API key from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        debug_print("GOOGLE_API_KEY not found in environment variables")
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    debug_print("API key found, creating GeminiAgent")
    # Create and return the agent
    return GeminiAgent(api_key=api_key)

if __name__ == "__main__":
    debug_print("Script started as main")
    print("Creating agent with Google provider")
    try:
        agent = create_agent("google")
        print("Agent created. Type 'exit' to quit.")
        debug_print("Agent created successfully")
        
        while True:
            query = input("\nYour question: ")
            debug_print(f"User input: {query}")
            
            if query.lower() in ("exit", "quit"):
                debug_print("User requested exit")
                print("Goodbye!")
                break
            
            try:
                print("\n[Searching and generating answer...]")
                
                start_time = time.time()
                debug_print(f"Starting agent run at {start_time}")
                resp = agent.run(query)
                end_time = time.time()
                elapsed = end_time - start_time
                debug_print(f"Agent run completed in {elapsed:.2f} seconds")
                print(f"Response generated in {elapsed:.2f} seconds")
                
                print(f"\nResponse: {resp}")
                    
            except RuntimeError as e:
                debug_print(f"RuntimeError: {str(e)}")
                if "Call limit reached" in str(e):
                    print("\n[Note]: Search limit reached. Moving on with partial results.")
                else:
                    print(f"\n[Error]: {str(e)}")
            except Exception as e:
                debug_print(f"Unexpected error: {str(e)}")
                print(f"\n[Error]: An unexpected error occurred: {str(e)}")
                print("Please try again or check your configuration.")
    except Exception as e:
        debug_print(f"Error during agent creation: {str(e)}")
        print(f"Failed to create agent: {str(e)}")
        print("Please try again or check your configuration.")