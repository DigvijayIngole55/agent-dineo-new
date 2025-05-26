import os
import time
import random
from typing import Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
import google.generativeai as genai

# Import all tools from tools.py
from tools import (
    WebSearchTool, WikiSearchTool, debug_print,
    download_from_url, extract_text_from_image, analyze_tabular_file,
    save_and_read_file, open_and_read_file, analyze_csv_file,
    analyze_excel_file, arxiv_search, analyze_video
)

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
        max_retries = 5
        base_sleep = 2  # Start with 2 second sleep
        
        # Check if the query contains a YouTube URL
        if "youtube.com" in query or "youtu.be" in query:
            try:
                # Use Gemini's native video analysis
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(query)
                return response.text
            except Exception as e:
                debug_print(f"Error in video analysis: {str(e)}")
                return f"Error analyzing video: {str(e)}"
        
        # Reset tool call counts for new query
        for tool in self.tools:
            if hasattr(tool.func, 'reset'):
                tool.func.reset()
        
        for attempt in range(max_retries):
            try:
                debug_print(f"Attempt {attempt + 1}/{max_retries}")
                response = self.agent.run(query)
                debug_print(f"Agent response received (length: {len(response)})")
                return response

            except Exception as e:
                debug_print(f"Error in attempt {attempt + 1}: {str(e)}")
                
                # Check if it's a rate limit error
                if "429" in str(e) or "quota" in str(e).lower():
                    # Exponential backoff for rate limits
                    sleep_time = base_sleep * (2 ** attempt) + random.uniform(0, 1)
                    if attempt < max_retries - 1:
                        debug_print(f"Rate limit hit. Waiting {sleep_time:.1f} seconds...")
                        print(f"Rate limit reached. Waiting {sleep_time:.1f} seconds before retry...")
                        time.sleep(sleep_time)
                        continue
                
                # For other errors, use shorter sleep
                sleep_time = base_sleep * (attempt + 1)
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
            # Configure the model for multimodal capabilities
            model = genai.GenerativeModel(self.model_name)
            
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0,
                max_output_tokens=2000,
                convert_system_message_to_human=True,
                system=system_prompt,
                model_kwargs={
                    "generation_config": {
                        "temperature": 0,
                        "top_p": 1,
                        "top_k": 32,
                        "max_output_tokens": 2000,
                    }
                }
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
            # Tool(
            #     name="analyze_video",
            #     func=analyze_video,
            #     description="Analyze YouTube video content"
            # ),
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
            # Initialize agent executor with improved settings
            agent = initialize_agent(
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                tools=self.tools,
                llm=self.llm,
                verbose=True,
                memory=self.memory,
                max_iterations=10,  # Increased from default
                max_execution_time=300,  # 5 minutes timeout
                early_stopping_method="generate",
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
        print("Please ensure GOOGLE_API_KEY is set in your environment variables.")