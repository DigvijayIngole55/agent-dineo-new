import os
import traceback
import requests
import time
import tempfile
import uuid
import pandas as pd
from urllib.parse import urlparse
from PIL import Image
import base64
import io
from typing import Optional
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client

load_dotenv()

# Log environment variables (without revealing secrets)
print("=== ENVIRONMENT SETUP ===")
print(f"SUPABASE_URL exists: {'SUPABASE_URL' in os.environ}")
print(f"SUPABASE_KEY exists: {'SUPABASE_KEY' in os.environ}")
print(f"GROQ_API_KEY exists: {'GROQ_API_KEY' in os.environ}")
print(f"GOOGLE_API_KEY exists: {'GOOGLE_API_KEY' in os.environ}")
print(f"SERPER_API_KEY exists: {'SERPER_API_KEY' in os.environ}")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    print(f"TOOL: multiply({a}, {b})")
    result = a * b
    print(f"TOOL RESULT: multiply = {result}")
    return result

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    Args:
        a: first int
        b: second int
    """
    print(f"TOOL: add({a}, {b})")
    result = a + b
    print(f"TOOL RESULT: add = {result}")
    return result

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    Args:
        a: first int
        b: second int
    """
    print(f"TOOL: subtract({a}, {b})")
    result = a - b
    print(f"TOOL RESULT: subtract = {result}")
    return result

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    Args:
        a: first int
        b: second int
    """
    print(f"TOOL: divide({a}, {b})")
    if b == 0:
        print("TOOL ERROR: Division by zero attempted")
        raise ValueError("Cannot divide by zero.")
    result = a / b
    print(f"TOOL RESULT: divide = {result}")
    return result

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    Args:
        a: first int
        b: second int
    """
    print(f"TOOL: modulus({a}, {b})")
    result = a % b
    print(f"TOOL RESULT: modulus = {result}")
    return result

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    Args:
        query: The search query."""
    print(f"TOOL: wiki_search('{query[:50]}...')")
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        print(f"TOOL: wiki_search found {len(search_docs)} documents")
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ])
        result = {"wiki_results": formatted_search_docs}
        print(f"TOOL RESULT: wiki_search returned: {result}")
        return result
    except Exception as e:
        print(f"TOOL ERROR: wiki_search failed - {str(e)}")
        print(f"TOOL ERROR: wiki_search traceback - {traceback.format_exc()}")
        return {"wiki_results": f"Error searching Wikipedia: {str(e)}"}

@tool
def web_search(query: str) -> str:
    """Search the web using Serper API and return maximum 5 results.
    Args:
        query: The search query."""
    print(f"TOOL: serper_web_search('{query[:50]}...')")
    api_key = os.getenv('SERPER_API_KEY')
    if not api_key:
        print("TOOL ERROR: SERPER_API_KEY not found in environment variables")
        return {"web_results": "Error: SERPER_API_KEY not configured"}
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": query,
            "num": 5
        }
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        print(f"TOOL: Making request to Serper API")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"TOOL: serper_web_search got response with {len(data.get('organic', []))} results")
        # Format the results
        formatted_results = []
        for res in data.get('organic', [])[:5]:
            formatted_result = f'<Document source="{res.get("link", "")}" title="{res.get("title", "")}"/>\n{res.get("snippet", "")}\n</Document>'
            formatted_results.append(formatted_result)
        formatted_search_docs = "\n\n---\n\n".join(formatted_results)
        result = {"web_results": formatted_search_docs}
        print(f"TOOL RESULT: serper_web_search returned: {result}")
        return result
    except requests.exceptions.RequestException as e:
        print(f"TOOL ERROR: serper_web_search network error - {str(e)}")
        print(f"TOOL ERROR: serper_web_search traceback - {traceback.format_exc()}")
        return {"web_results": f"Network error during web search: {str(e)}"}
    except Exception as e:
        print(f"TOOL ERROR: serper_web_search failed - {str(e)}")
        print(f"TOOL ERROR: serper_web_search traceback - {traceback.format_exc()}")
        return {"web_results": f"Error during web search: {str(e)}"}

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 results.
    Args:
        query: The search query."""
    print(f"TOOL: arxiv_search('{query[:50]}...')")
    try:
        search_docs = ArxivLoader(query=query, load_max_docs=3).load()
        print(f"TOOL: arxiv_search found {len(search_docs)} documents")
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
                for doc in search_docs
            ])
        result = {"arxiv_results": formatted_search_docs}
        print(f"TOOL RESULT: arxiv_search returned: {result}")
        return result
    except Exception as e:
        print(f"TOOL ERROR: arxiv_search failed - {str(e)}")
        print(f"TOOL ERROR: arxiv_search traceback - {traceback.format_exc()}")
        return {"arxiv_results": f"Error searching Arxiv: {str(e)}"}

# Load the system prompt from the file
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    print("System prompt loaded successfully")
except Exception as e:
    print(f"Failed to load system_prompt.txt: {e}")
    system_prompt = "You are a helpful AI assistant."

@tool
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    return f"File saved to {filepath}. You can read this file to process its contents."

@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    Args:
        url (str): the URL of the file to download.
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    try:
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"

        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"File downloaded to {filepath}. You can read this file to process its contents."
    except Exception as e:
        return f"Error downloading file: {str(e)}"

@tool
def analyze_youtube_video(url: str) -> str:
    """
    Analyze a YouTube video by extracting metadata and transcript.
    Args:
        url (str): YouTube video URL
    """
    try:
        import re
        # Extract video ID from URL
        video_id_match = re.search(r'(?:v=|\/embed\/|\/v\/|\.be\/)([a-zA-Z0-9_-]{11})', url)
        if not video_id_match:
            return "Error: Could not extract video ID from URL"
        
        video_id = video_id_match.group(1)
        
        # Try to get video metadata using web search
        search_query = f"site:youtube.com {video_id} video title description"
        metadata_result = web_search(search_query)
        
        result = f"YouTube Video Analysis for: {url}\n"
        result += f"Video ID: {video_id}\n\n"
        result += f"Metadata search results:\n{metadata_result}\n\n"
        
        # Try to search for transcripts
        transcript_query = f"youtube video {video_id} transcript subtitles"
        transcript_result = web_search(transcript_query)
        result += f"Transcript search results:\n{transcript_result}\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing YouTube video: {str(e)}"

@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using Gemini vision capabilities.
    Args:
        image_path (str): the path to the image file.
    """
    try:
        from langchain_core.messages import HumanMessage
        
        # Initialize Gemini model
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode()
        
        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Please extract all text from this image. Return only the text content without any additional commentary."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        )
        
        # Get response from Gemini
        response = llm.invoke([message])
        
        return f"Extracted text from image:\n\n{response.content}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

@tool
def read_file_content(file_path: str) -> str:
    """
    Read content from any file.
    Args:
        file_path (str): the path to the file to read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"File content ({len(content)} characters):\n{content}"
    except UnicodeDecodeError:
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return f"Binary file ({len(content)} bytes). Cannot display content as text."
        except Exception as e:
            return f"Error reading file: {str(e)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def analyze_csv_file(file_path: str, query: str = "") -> str:
    """
    Analyze a CSV file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the CSV file.
        query (str): Question about the data (optional)
    """
    try:
        df = pd.read_csv(file_path)
        
        result = f"CSV Analysis:\n"
        result += f"- Rows: {len(df)}, Columns: {len(df.columns)}\n"
        result += f"- Column names: {list(df.columns)}\n\n"
        
        # Show first few rows
        result += "First 5 rows:\n"
        result += str(df.head()) + "\n\n"
        
        # Show data types
        result += "Data types:\n"
        result += str(df.dtypes) + "\n\n"
        
        # Show summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            result += "Summary statistics (numeric columns):\n"
            result += str(df[numeric_cols].describe()) + "\n\n"
        
        # Calculate totals for numeric columns if requested
        if "total" in query.lower() or "sum" in query.lower():
            for col in numeric_cols:
                total = df[col].sum()
                result += f"Total {col}: {total}\n"

        return result

    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"

@tool
def analyze_excel_file(file_path: str, query: str = "") -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the Excel file.
        query (str): Question about the data (optional)
    """
    try:
        df = pd.read_excel(file_path)
        
        result = f"Excel Analysis:\n"
        result += f"- Rows: {len(df)}, Columns: {len(df.columns)}\n"
        result += f"- Column names: {list(df.columns)}\n\n"
        
        # Show first few rows
        result += "First 5 rows:\n"
        result += str(df.head()) + "\n\n"
        
        # Show data types
        result += "Data types:\n"
        result += str(df.dtypes) + "\n\n"
        
        # Show summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            result += "Summary statistics (numeric columns):\n"
            result += str(df[numeric_cols].describe()) + "\n\n"
        
        # Calculate totals for numeric columns if requested
        if "total" in query.lower() or "sum" in query.lower():
            for col in numeric_cols:
                total = df[col].sum()
                result += f"Total {col}: {total}\n"

        return result

    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"

# System message
sys_msg = SystemMessage(content=system_prompt)

# Build retriever
print("=== BUILDING RETRIEVER ===")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("HuggingFace embeddings initialized")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the environment.")
    supabase: Client = create_client(supabase_url, supabase_key)
    print("Supabase client created")
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )
    print("Vector store initialized")
    retriever_tool = create_retriever_tool(
        retriever=vector_store.as_retriever(),
        name="Question_Search",
        description="A tool to retrieve similar questions from a vector store.",
    )
    print("Retriever tool created successfully")
except Exception as e:
    print(f"Failed to initialize retriever: {e}")
    print(f"Retriever initialization traceback: {traceback.format_exc()}")
    vector_store = None
    retriever_tool = None

tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    web_search,
    wiki_search,
    arxiv_search,
    save_and_read_file,
    read_file_content,
    analyze_excel_file,
    analyze_csv_file,
    extract_text_from_image,
    download_file_from_url,
    analyze_youtube_video,
    
]
if retriever_tool:
    tools.append(retriever_tool)
print(f"Tools initialized: {[tool.name for tool in tools]}")



def get_llm(provider: str):
    """Initializes and returns the specified LLM."""
    if provider == "google":
        try:
            print("Initializing Google Gemini LLM")
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        except Exception as e:
            print(f"Failed to initialize Google Gemini LLM: {e}")
            return None
    elif provider == "groq":
        try:
            print("Initializing Groq LLM")
            return ChatGroq(model="qwen-qwq-32b", temperature=0)
        except Exception as e:
            print(f"Failed to initialize Groq LLM: {e}")
            return None
    elif provider == "huggingface":
        try:
            print("Initializing HuggingFace LLM")
            return ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                    temperature=0.1,
                ),
            )
        except Exception as e:
            print(f"Failed to initialize HuggingFace LLM: {e}")
            return None
    else:
        print(f"Invalid LLM provider specified: {provider}")
        return None

def build_graph(provider: str = "groq"):
    if provider == "groq":
        # Groq https://console.groq.com/docs/models
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
        # llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
    elif provider == "google":
        # Google Gemini for video and multimodal tasks
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                task="text-generation",  # for chat‐style use “text-generation”
                max_new_tokens=1024,
                do_sample=False,
                repetition_penalty=1.03,
                temperature=0,
            ),
            verbose=True,
        )
    else:
        raise ValueError("Invalid provider. Choose 'groq', 'google', or 'huggingface'.")

    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        """Assistant Node"""
        return {"messages": [llm_with_tools.invoke(state['messages'])]}

    # def retriever(state: MessagesState):
    #     """Retriever Node"""
    #     # Extract the latest message content
    #     query = state['messages'][-1].content
    #     similar_question = vector_store.similarity_search(query, k = 2)
    #     if similar_question:  
    #         example_msg = HumanMessage(
    #             content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
    #         )
    #         return {"messages": [sys_msg] + state["messages"] + [example_msg]}
    #     else:
    #         return {"messages": [sys_msg] + state["messages"]}


    builder = StateGraph(MessagesState)
    # builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    # builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()

if __name__ == "__main__":

    graph = build_graph(provider="groq")
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()