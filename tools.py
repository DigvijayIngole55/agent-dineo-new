import os
import tempfile
import time
import json
import requests
from typing import Optional
from urllib.parse import urlparse
from langchain_community.document_loaders import WikipediaLoader

# Debug flag
DEBUG = True

def debug_print(message):
    """Helper function to print debug messages when DEBUG is True"""
    if DEBUG:
        print(f"[DEBUG] {message}")

class GlobalRateLimiter:
    """Global rate limiter to coordinate between different tools."""
    
    def __init__(self):
        self.last_google_api_call = 0
        self.google_api_delay = 65  # 65 seconds between Google API calls
        self.consecutive_rate_limits = 0
        self.max_consecutive_limits = 3
    
    def should_wait_for_google_api(self):
        """Check if we should wait before making a Google API call."""
        current_time = time.time()
        time_since_last = current_time - self.last_google_api_call
        
        if time_since_last < self.google_api_delay:
            wait_time = self.google_api_delay - time_since_last
            debug_print(f"Global rate limiter: waiting {wait_time:.2f}s for Google API")
            return wait_time
        return 0
    
    def record_google_api_call(self, success=True):
        """Record a Google API call."""
        self.last_google_api_call = time.time()
        if success:
            self.consecutive_rate_limits = 0
        else:
            self.consecutive_rate_limits += 1
    
    def is_heavily_rate_limited(self):
        """Check if we're being heavily rate limited."""
        return self.consecutive_rate_limits >= self.max_consecutive_limits

# Global rate limiter instance
global_rate_limiter = GlobalRateLimiter()

class WebSearchTool:
    """Tool for performing web searches with rate limiting using Serper API."""
    
    def __init__(self):
        self.name = "web_search"
        self.description = "Search the web for a query and return results."
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Reduced from 1.0 to 0.5 seconds
        self.max_retries = 5  # Increased retries
        self.google_api_delay = 65  # 65 seconds for Google API rate limits
        # Check if Serper API key is available
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        if self.serper_api_key:
            debug_print("Serper API key found")
        else:
            debug_print("No Serper API key found - web search will not work")
    
    def __call__(self, query: str) -> str:
        """Perform web search with rate limiting and retries."""
        debug_print(f"Web searching for: {query}")
        
        if not self.serper_api_key:
            return {"web_results": "No Serper API key configured. Please set SERPER_API_KEY environment variable."}
        
        # Implement rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            debug_print(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        for attempt in range(self.max_retries):
            try:
                result = self._serper_search(query)
                self.last_request_time = time.time()
                global_rate_limiter.record_google_api_call(success=True)
                return result
            except Exception as e:
                debug_print(f"Serper search error in attempt {attempt+1}: {str(e)}")
                
                # Check for rate limit errors
                if "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower():
                    global_rate_limiter.record_google_api_call(success=False)
                    if attempt < self.max_retries - 1:
                        debug_print(f"Rate limit hit. Waiting {self.google_api_delay} seconds...")
                        print(f"Search rate limit reached. Waiting {self.google_api_delay} seconds before retry...")
                        time.sleep(self.google_api_delay)
                        continue
                
                # For other errors, use exponential backoff
                if attempt < self.max_retries - 1:
                    wait_time = min((2 ** attempt) * self.min_request_interval, 30)
                    debug_print(f"Waiting {wait_time:.2f}s before retry")
                    time.sleep(wait_time)
                    continue
                else:
                    return {"web_results": f"Search error after {self.max_retries} attempts: {str(e)}"}
        
        return {"web_results": "Search failed after all retries"}
    
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
            raise

class WikiSearchTool:
    """Tool for searching Wikipedia with rate limiting."""
    
    def __init__(self):
        self.name = "wiki_search"
        self.description = "Search Wikipedia for a query and return maximum 2 results."
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds
        self.google_api_delay = 65  # 65 seconds for Google API rate limits
        self.max_calls = 5  # Increased from 3
        self.call_count = 0
    
    def reset(self):
        """Reset call count for new query"""
        self.call_count = 0
        debug_print("Wiki search tool reset")
    
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
            global_rate_limiter.record_google_api_call(success=True)
            return {"wiki_results": results}
        
        except Exception as e:
            debug_print(f"Error searching Wikipedia: {str(e)}")
            
            # Check for rate limit errors
            if "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower():
                global_rate_limiter.record_google_api_call(success=False)
                debug_print(f"Wikipedia rate limit hit. Waiting {self.google_api_delay} seconds...")
                time.sleep(self.google_api_delay)
                return {"wiki_results": f"Rate limit hit. Please try again in {self.google_api_delay} seconds."}
            
            return {"wiki_results": f"Error searching Wikipedia: {str(e)}"}

def find_uploaded_files(file_type: str = None) -> str:
    """Find uploaded files in the temp directory."""
    debug_print(f"Finding uploaded files of type: {file_type}")
    temp_dir = tempfile.gettempdir()
    files = []
    
    try:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                if file_type:
                    if file_type.lower() == 'excel' and filename.endswith(('.xlsx', '.xls', '.xlsm')):
                        files.append(file_path)
                    elif file_type.lower() == 'csv' and filename.endswith('.csv'):
                        files.append(file_path)
                    elif file_type.lower() == 'audio' and filename.endswith(('.mp3', '.wav', '.m4a')):
                        files.append(file_path)
                    elif file_type.lower() == 'image' and filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        files.append(file_path)
                else:
                    files.append(file_path)
        
        if files:
            file_list = [os.path.basename(f) for f in files]
            return f"Found {len(files)} files: {', '.join(file_list)}"
        else:
            return f"No files found of type: {file_type}" if file_type else "No files found in temp directory"
            
    except Exception as e:
        debug_print(f"Error finding files: {str(e)}")
        return f"Error finding files: {str(e)}"

def handle_file_reference(query: str) -> str:
    """Handle queries that reference files that may not be directly accessible."""
    file_references = [
        'attached', 'attachment', 'file', 'excel', 'csv', 'mp3', 'audio',
        'image', 'picture', 'document', 'pdf', 'video'
    ]
    
    if any(ref in query.lower() for ref in file_references):
        # Look for available files
        temp_files = find_uploaded_files()
        if "Found" in temp_files and "files:" in temp_files:
            return f"I found these files that might be relevant: {temp_files}"
        else:
            return "I cannot access the file you're referring to. Please ensure the file is properly uploaded or provide the content directly in your message."
    
    return None

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
        response = requests.get(url, stream=True, timeout=30)
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

def analyze_csv_file(file_path_or_query: str, query: Optional[str] = None) -> str:
   """
   Analyze a CSV file using pandas and answer a question about it.
   
   Args:
       file_path_or_query: Either the file path or a query (for backward compatibility)
       query: Optional question about the data
       
   Returns:
       Analysis result or error message
   """
   debug_print(f"Analyzing CSV file: {file_path_or_query}, query: {query}")
   try:
       import pandas as pd
       
       # Handle case where first argument might be the query
       if file_path_or_query and not os.path.exists(file_path_or_query):
           # Assume it's a query, try to find CSV files in temp directory
           temp_dir = tempfile.gettempdir()
           csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
           if csv_files:
               file_path = os.path.join(temp_dir, csv_files[0])  # Use first found
               query = file_path_or_query  # The first argument was actually the query
           else:
               return "No CSV file found. Please specify a valid file path."
       else:
           file_path = file_path_or_query
       
       # Read the CSV file
       df = pd.read_csv(file_path)
       
       # Basic analysis
       result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
       result += f"Columns: {', '.join(df.columns)}\n\n"
       
       # Show first few rows
       result += "First 5 rows:\n"
       result += str(df.head()) + "\n\n"
       
       # Add summary statistics for numeric columns
       numeric_cols = df.select_dtypes(include=['number']).columns
       if len(numeric_cols) > 0:
           result += "Summary statistics for numeric columns:\n"
           result += str(df[numeric_cols].describe()) + "\n\n"
       
       # If there's a specific query, try to answer it
       if query:
           result += f"Query: {query}\n"
           # Add basic query processing here
           if any(word in query.lower() for word in ['total', 'sum']):
               for col in numeric_cols:
                   result += f"Total {col}: {df[col].sum()}\n"
       
       return result
       
   except ImportError:
       return "Error: pandas is not installed. Please install it with 'pip install pandas'."
   except Exception as e:
       debug_print(f"Error analyzing CSV file: {str(e)}")
       return f"Error analyzing CSV file: {str(e)}"

def analyze_excel_file(file_path_or_query: str, query: Optional[str] = None) -> str:
   """
   Analyze an Excel file using pandas and answer a question about it.
   
   Args:
       file_path_or_query: Either the file path or a query (for backward compatibility)
       query: Optional question about the data
       
   Returns:
       Analysis result or error message
   """
   debug_print(f"Analyzing Excel file: {file_path_or_query}, query: {query}")
   try:
       import pandas as pd
       
       # Handle case where first argument might be the query
       if file_path_or_query and not os.path.exists(file_path_or_query):
           # Assume it's a query, try to find Excel files in temp directory
           temp_dir = tempfile.gettempdir()
           excel_files = [f for f in os.listdir(temp_dir) if f.endswith(('.xlsx', '.xls', '.xlsm'))]
           if excel_files:
               file_path = os.path.join(temp_dir, excel_files[0])  # Use first found
               query = file_path_or_query  # The first argument was actually the query
           else:
               return "No Excel file found. Please specify a valid file path."
       else:
           file_path = file_path_or_query
       
       # Read the Excel file
       df = pd.read_excel(file_path)
       
       # Basic analysis
       result = f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
       result += f"Columns: {', '.join(df.columns)}\n\n"
       
       # Show first few rows
       result += "First 5 rows:\n"
       result += str(df.head()) + "\n\n"
       
       # Add summary statistics for numeric columns
       numeric_cols = df.select_dtypes(include=['number']).columns
       if len(numeric_cols) > 0:
           result += "Summary statistics for numeric columns:\n"
           result += str(df[numeric_cols].describe()) + "\n\n"
           
           # Calculate totals for numeric columns
           result += "Column totals:\n"
           for col in numeric_cols:
               total = df[col].sum()
               result += f"Total {col}: {total:.2f}\n"
       
       # If there's a specific query about food vs drinks, try to categorize
       if query and any(word in query.lower() for word in ['food', 'drink', 'beverage']):
           result += "\nAttempting to categorize items as food vs drinks:\n"
           
           # Look for columns that might contain item names or categories
           text_cols = df.select_dtypes(include=['object']).columns
           for col in text_cols:
               if any(keyword in col.lower() for keyword in ['item', 'product', 'name', 'category']):
                   unique_items = df[col].unique()
                   result += f"Items in {col}: {', '.join(map(str, unique_items))}\n"
       
       # If there's a specific query, try to answer it
       if query:
           result += f"\nQuery: {query}\n"
           # Add basic query processing here
           if any(word in query.lower() for word in ['total', 'sum']):
               if 'food' in query.lower() and 'not' in query.lower() and 'drink' in query.lower():
                   result += "Note: To properly calculate food sales excluding drinks, I would need to know which items are food vs drinks.\n"
       
       return result
       
   except ImportError:
       return "Error: pandas and openpyxl are not installed. Please install them with 'pip install pandas openpyxl'."
   except Exception as e:
       debug_print(f"Error analyzing Excel file: {str(e)}")
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

def analyze_video(input_data: str) -> str:
   """Analyze video content using Gemini's video understanding capabilities."""
   debug_print(f"Raw input to analyze_video: {input_data}")
   
   try:
       # Try to parse JSON input
       url = input_data
       question = ""
       
       # Check if input looks like JSON
       if isinstance(input_data, str) and input_data.strip().startswith('{'):
           try:
               params = json.loads(input_data)
               url = params.get('url', input_data)
               question = params.get('question', '')
               debug_print(f"Parsed JSON - URL: {url}, Question: {question}")
           except json.JSONDecodeError:
               debug_print("JSON parsing failed, treating as URL")
               url = input_data
       
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
                       result += f"Description: {description[:500]}...\n" if len(description) > 500 else f"Description: {description}\n"
                   
                   if question:
                       result += f"\nQuestion: {question}\n"
                       result += "Note: Video content analysis requires advanced features not available in this implementation."
                   
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
   