import os
import logging
import traceback
import requests
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google.genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Log environment variables (without revealing secrets)
logger.info("=== ENVIRONMENT SETUP ===")
logger.info(f"SUPABASE_URL exists: {'SUPABASE_URL' in os.environ}")
logger.info(f"SUPABASE_KEY exists: {'SUPABASE_KEY' in os.environ}")
logger.info(f"GROQ_API_KEY exists: {'GROQ_API_KEY' in os.environ}")
logger.info(f"GOOGLE_API_KEY exists: {'GOOGLE_API_KEY' in os.environ}")
logger.info(f"SERPER_API_KEY exists: {'SERPER_API_KEY' in os.environ}")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    logger.info(f"TOOL: multiply({a}, {b})")
    result = a * b
    logger.info(f"TOOL RESULT: multiply = {result}")
    return result

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    Args:
        a: first int
        b: second int
    """
    logger.info(f"TOOL: add({a}, {b})")
    result = a + b
    logger.info(f"TOOL RESULT: add = {result}")
    return result

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    Args:
        a: first int
        b: second int
    """
    logger.info(f"TOOL: subtract({a}, {b})")
    result = a - b
    logger.info(f"TOOL RESULT: subtract = {result}")
    return result

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    Args:
        a: first int
        b: second int
    """
    logger.info(f"TOOL: divide({a}, {b})")
    if b == 0:
        logger.error("TOOL ERROR: Division by zero attempted")
        raise ValueError("Cannot divide by zero.")
    result = a / b
    logger.info(f"TOOL RESULT: divide = {result}")
    return result

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    Args:
        a: first int
        b: second int
    """
    logger.info(f"TOOL: modulus({a}, {b})")
    result = a % b
    logger.info(f"TOOL RESULT: modulus = {result}")
    return result

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    Args:
        query: The search query."""
    logger.info(f"TOOL: wiki_search('{query[:50]}...')")
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        logger.info(f"TOOL: wiki_search found {len(search_docs)} documents")
        
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ])
        
        result = {"wiki_results": formatted_search_docs}
        logger.info(f"TOOL RESULT: wiki_search completed successfully")
        return result
    except Exception as e:
        logger.error(f"TOOL ERROR: wiki_search failed - {str(e)}")
        logger.error(f"TOOL ERROR: wiki_search traceback - {traceback.format_exc()}")
        return {"wiki_results": f"Error searching Wikipedia: {str(e)}"}

@tool
def serper_web_search(query: str) -> str:
    """Search the web using Serper API and return maximum 5 results.
    Args:
        query: The search query."""
    logger.info(f"TOOL: serper_web_search('{query[:50]}...')")
    
    api_key = os.getenv('SERPER_API_KEY')
    if not api_key:
        logger.error("TOOL ERROR: SERPER_API_KEY not found in environment variables")
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
        
        logger.info(f"TOOL: Making request to Serper API")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"TOOL: serper_web_search got response with {len(data.get('organic', []))} results")
        
        # Format the results
        formatted_results = []
        for result in data.get('organic', [])[:5]:
            formatted_result = f'<Document source="{result.get("link", "")}" title="{result.get("title", "")}"/>\n{result.get("snippet", "")}\n</Document>'
            formatted_results.append(formatted_result)
        
        formatted_search_docs = "\n\n---\n\n".join(formatted_results)
        result = {"web_results": formatted_search_docs}
        
        logger.info(f"TOOL RESULT: serper_web_search completed successfully")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"TOOL ERROR: serper_web_search network error - {str(e)}")
        logger.error(f"TOOL ERROR: serper_web_search traceback - {traceback.format_exc()}")
        return {"web_results": f"Network error during web search: {str(e)}"}
    except Exception as e:
        logger.error(f"TOOL ERROR: serper_web_search failed - {str(e)}")
        logger.error(f"TOOL ERROR: serper_web_search traceback - {traceback.format_exc()}")
        return {"web_results": f"Error during web search: {str(e)}"}

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 results.
    Args:
        query: The search query."""
    logger.info(f"TOOL: arxiv_search('{query[:50]}...')")
    try:
        search_docs = ArxivLoader(query=query, load_max_docs=3).load()
        logger.info(f"TOOL: arxiv_search found {len(search_docs)} documents")
        
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
                for doc in search_docs
            ])
        
        result = {"arxiv_results": formatted_search_docs}
        logger.info(f"TOOL RESULT: arxiv_search completed successfully")
        return result
    except Exception as e:
        logger.error(f"TOOL ERROR: arxiv_search failed - {str(e)}")
        logger.error(f"TOOL ERROR: arxiv_search traceback - {traceback.format_exc()}")
        return {"arxiv_results": f"Error searching Arxiv: {str(e)}"}

# Load the system prompt from the file
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    logger.info("System prompt loaded successfully")
except Exception as e:
    logger.error(f"Failed to load system_prompt.txt: {e}")
    system_prompt = "You are a helpful AI assistant."

# System message
sys_msg = SystemMessage(content=system_prompt)

# Build retriever
logger.info("=== BUILDING RETRIEVER ===")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    logger.info("HuggingFace embeddings initialized")
    
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the environment.")
        
    supabase: Client = create_client(supabase_url, supabase_key)
    logger.info("Supabase client created")
    
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents", # Corrected to a more generic name, adjust if needed
    )
    logger.info("Vector store initialized")
    
    retriever_tool = create_retriever_tool(
        retriever=vector_store.as_retriever(),
        name="Question_Search",
        description="A tool to retrieve similar questions from a vector store.",
    )
    logger.info("Retriever tool created successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize retriever: {e}")
    logger.error(f"Retriever initialization traceback: {traceback.format_exc()}")
    vector_store = None
    retriever_tool = None

tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    serper_web_search,
    arxiv_search,
]
if retriever_tool:
    tools.append(retriever_tool)


logger.info(f"Tools initialized: {[tool.name for tool in tools]}")

def get_llm(provider: str = "google"):
    """Initializes and returns the specified LLM, with a fallback to Groq."""
    if provider == "google":
        try:
            logger.info("Attempting to initialize Google Gemini LLM")
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            # A simple test to see if the API key is valid and the service is available
            llm.invoke("test")
            logger.info("Google Gemini LLM initialized successfully.")
            return llm
        except Exception as e:
            logger.warning(f"Failed to initialize Google Gemini LLM: {e}. Falling back to Groq.")
            provider = "groq"

    if provider == "groq":
        try:
            logger.info("Initializing Groq LLM")
            return ChatGroq(model="llama3-70b-8192", temperature=0)
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            raise

    if provider == "huggingface":
        try:
            logger.info("Initializing HuggingFace LLM")
            return ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                    temperature=0.1,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace LLM: {e}")
            raise
            
    raise ValueError(f"Invalid or failed to initialize provider: {provider}")


# Build graph function
def build_graph():
    """Build the graph"""
    
    try:
        llm = get_llm() # Automatically handles fallback
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)
        logger.info("Tools bound to LLM")

        # Node
        def assistant(state: MessagesState):
            """Assistant node"""
            logger.info("=== ASSISTANT NODE CALLED ===")
            logger.info(f"Assistant received {len(state['messages'])} messages")
            
            try:
                # Log the last message (user input)
                if state["messages"]:
                    last_msg = state["messages"][-1]
                    logger.info(f"Last message type: {type(last_msg).__name__}")
                    logger.info(f"Last message content (first 100 chars): {str(last_msg.content)[:100]}...")
                
                logger.info("Calling LLM...")
                response = llm_with_tools.invoke(state["messages"])
                logger.info(f"LLM response type: {type(response).__name__}")
                logger.info(f"LLM response content (first 200 chars): {str(response.content)[:200]}...")
                
                # Check if LLM wants to use tools
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    logger.info(f"LLM requested {len(response.tool_calls)} tool calls")
                    for i, tool_call in enumerate(response.tool_calls):
                        logger.info(f"Tool call {i+1}: {tool_call.get('name', 'unknown')} with args: {tool_call.get('args', {})}")
                else:
                    logger.info("LLM did not request any tool calls")
                
                logger.info("=== ASSISTANT NODE COMPLETED ===")
                return {"messages": [response]}
                
            except Exception as e:
                logger.error(f"ASSISTANT NODE ERROR: {str(e)}")
                logger.error(f"ASSISTANT NODE TRACEBACK: {traceback.format_exc()}")
                # Return an error message instead of crashing
                error_msg = HumanMessage(content=f"I encountered an error: {str(e)}")
                return {"messages": [error_msg]}

        def retriever(state: MessagesState):
            """Retriever node"""
            logger.info("=== RETRIEVER NODE CALLED ===")
            
            try:
                if vector_store is None:
                    logger.warning("Vector store not available, skipping retrieval")
                    return {"messages": [sys_msg] + state["messages"]}
                
                query = state["messages"][0].content
                logger.info(f"Searching for similar questions with query: {query[:100]}...")
                
                similar_question = vector_store.similarity_search(query)
                logger.info(f"Found {len(similar_question)} similar questions")
                
                if similar_question:
                    example_content = similar_question[0].page_content
                    logger.info(f"Using similar question (first 100 chars): {example_content[:100]}...")
                    
                    example_msg = HumanMessage(
                        content=f"Here I provide a similar question and answer for reference: \n\n{example_content}",
                    )
                    result = {"messages": [sys_msg] + state["messages"] + [example_msg]}
                else:
                    logger.info("No similar questions found")
                    result = {"messages": [sys_msg] + state["messages"]}
                
                logger.info("=== RETRIEVER NODE COMPLETED ===")
                return result
                
            except Exception as e:
                logger.error(f"RETRIEVER NODE ERROR: {str(e)}")
                logger.error(f"RETRIEVER NODE TRACEBACK: {traceback.format_exc()}")
                # Fallback to just system message + user messages
                return {"messages": [sys_msg] + state["messages"]}

        logger.info("Building state graph...")
        builder = StateGraph(MessagesState)
        builder.add_node("retriever", retriever)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(tools))
        
        builder.add_edge(START, "retriever")
        builder.add_edge("retriever", "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")

        logger.info("Compiling graph...")
        graph = builder.compile()
        logger.info("=== GRAPH BUILT SUCCESSFULLY ===")
        return graph
        
    except Exception as e:
        logger.error(f"GRAPH BUILD ERROR: {str(e)}")
        logger.error(f"GRAPH BUILD TRACEBACK: {traceback.format_exc()}")
        raise