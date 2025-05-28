import os
import traceback
import requests
import time
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
]
if retriever_tool:
    tools.append(retriever_tool)
print(f"Tools initialized: {[tool.name for tool in tools]}")

class LLMManager:
    """
    Manages LLM providers with a fallback and cooldown mechanism.
    """
    def __init__(self, llms_with_tools: dict, provider_order: list):
        """
        Args:
            llms_with_tools: A dictionary of provider names to their LLM instances with tools bound.
            provider_order: A list of provider names in the desired fallback order.
        """
        self.llms = llms_with_tools
        self.provider_order = provider_order
        self.cooldowns = {provider: 0 for provider in self.llms.keys()}

    def invoke(self, messages: MessagesState):
        """
        Invokes the next available LLM based on the fallback order and cooldowns.
        """
        for provider in self.provider_order:
            if provider not in self.llms:
                print(f"Provider '{provider}' in order list but not initialized.")
                continue
            if time.time() < self.cooldowns.get(provider, 0):
                print(f"Provider '{provider}' is on cooldown. Skipping.")
                continue
            print(f"Attempting to use LLM provider: {provider}")
            llm = self.llms[provider]
            try:
                # Successful invocation, return the result
                response = llm.invoke(messages)
                print(f"LLM provider '{provider}' succeeded.")
                return response
            except Exception as e:
                print(f"LLM provider '{provider}' failed: {e}")
                if provider == 'google':
                    # Specific logic for Google LLM failure
                    cooldown_duration = 60
                    self.cooldowns['google'] = time.time() + cooldown_duration
                    print(f"Google LLM failed. Placing on a {cooldown_duration}s cooldown.")
                    print("Immediately trying Groq as a specific fallback.")
                    try:
                        # Nested attempt to use Groq immediately
                        response = self.llms['groq'].invoke(messages)
                        print("Fallback to Groq succeeded.")
                        return response
                    except Exception as e_groq:
                        print(f"Fallback Groq LLM also failed: {e_groq}")
                        # Put Groq on its own cooldown if it fails here
                        groq_cooldown_duration_ms = 60
                        self.cooldowns['groq'] = time.time() + (groq_cooldown_duration_ms / 1000.0)
                        print(f"Groq LLM failed during fallback. Placing on a {groq_cooldown_duration_ms}ms cooldown.")
                        # Continue to the next provider in the main list
                elif provider == 'groq':
                    # Logic for Groq LLM failure
                    cooldown_duration_ms = 60
                    self.cooldowns['groq'] = time.time() + (cooldown_duration_ms / 1000.0)
                    print(f"Groq LLM failed. Placing on a {cooldown_duration_ms}ms cooldown.")
        # If all providers in the list have failed
        raise Exception("All available LLM providers failed.")

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

# Build graph function
def build_graph():
    """Build the graph"""
    try:
        # 1. Initialize all potential LLMs
        llm_providers = {
            "google": get_llm("google"),
            "groq": get_llm("groq"),
            "huggingface": get_llm("huggingface")
        }

        # Filter out any providers that failed to initialize
        initialized_llms = {name: llm for name, llm in llm_providers.items() if llm}
        if not initialized_llms:
            raise RuntimeError("No LLMs could be initialized. The application cannot start.")

        print(f"Successfully initialized LLMs: {list(initialized_llms.keys())}")

        # 2. Bind tools to each initialized LLM
        llms_with_tools = {
            name: llm.bind_tools(tools) for name, llm in initialized_llms.items()
        }
        print("Tools bound to all initialized LLMs")

        # 3. Create the LLMManager with a defined fallback order
        # The primary attempt will be Google, then Groq, then HuggingFace.
        fallback_order = ["google", "groq", "huggingface"]
        llm_manager = LLMManager(llms_with_tools, fallback_order)

        # Node
        def assistant(state: MessagesState):
            """Assistant node that uses the LLMManager for robust invocation"""
            print("=== ASSISTANT NODE CALLED ===")
            print(f"Assistant received {len(state['messages'])} messages")
            try:
                # Use the LLMManager to handle the call
                response = llm_manager.invoke(state["messages"])
                print(f"LLM response type: {type(response).__name__}")
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"LLM requested {len(response.tool_calls)} tool calls")
                else:
                    print("LLM did not request any tool calls")

                print("=== ASSISTANT NODE COMPLETED ===")
                return {"messages": [response]}
            except Exception as e:
                print(f"ASSISTANT NODE ERROR: {str(e)}")
                print(f"ASSISTANT NODE TRACEBACK: {traceback.format_exc()}")
                error_msg = HumanMessage(content=f"I encountered an error after trying all fallbacks: {str(e)}")
                return {"messages": [error_msg]}

        def retriever(state: MessagesState):
            """Retriever node"""
            print("=== RETRIEVER NODE CALLED ===")
            try:
                if vector_store is None:
                    print("Vector store not available, skipping retrieval")
                    return {"messages": [sys_msg] + state["messages"]}

                query = state["messages"][0].content
                print(f"Searching for similar questions with query: {query[:100]}...")
                similar_question = vector_store.similarity_search(query)
                print(f"Found {len(similar_question)} similar questions")

                if similar_question:
                    example_content = similar_question[0].page_content
                    print(f"Using similar question (first 100 chars): {example_content[:100]}...")
                    example_msg = HumanMessage(
                        content=f"Here I provide a similar question and answer for reference: \n\n{example_content}",
                    )
                    result = {"messages": [sys_msg] + state["messages"] + [example_msg]}
                else:
                    print("No similar questions found")
                    result = {"messages": [sys_msg] + state["messages"]}

                print("=== RETRIEVER NODE COMPLETED ===")
                return result
            except Exception as e:
                print(f"RETRIEVER NODE ERROR: {str(e)}")
                print(f"RETRIEVER NODE TRACEBACK: {traceback.format_exc()}")
                return {"messages": [sys_msg] + state["messages"]}

        print("Building state graph...")
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

        print("Compiling graph...")
        graph = builder.compile()
        print("=== GRAPH BUILT SUCCESSFULLY ===")
        return graph
    except Exception as e:
        print(f"GRAPH BUILD ERROR: {str(e)}")
        print(f"GRAPH BUILD TRACEBACK: {traceback.format_exc()}")
        raise