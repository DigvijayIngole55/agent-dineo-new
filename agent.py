import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from functools import wraps
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, tool, ChatMessage, LiteLLMModel, CodeAgent
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import json
import numpy as np
from datetime import datetime

load_dotenv()

# Path to store the local vector store
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "local_vectorstore")
# Path to store the local answers database
LOCAL_DB_PATH = os.getenv("LOCAL_DB_PATH", "local_answers.json")

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

class LocalVectorStore:
    """A simple class to manage a local vector store using FAISS and a JSON file for metadata."""
    
    def __init__(
        self, 
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        vector_store_path: str = VECTOR_STORE_PATH,
        local_db_path: str = LOCAL_DB_PATH
    ):
        self.vector_store_path = vector_store_path
        self.local_db_path = local_db_path
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name
        )
        
        # Initialize or load the vector store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize or load the local answers database
        self.answers_db = self._initialize_answers_db()
    
    def _initialize_vector_store(self) -> FAISS:
        """Initialize or load the FAISS vector store."""
        try:
            if os.path.exists(self.vector_store_path):
                print(f"Loading existing vector store from {self.vector_store_path}")
                return FAISS.load_local(self.vector_store_path, self.embedding_model,allow_dangerous_deserialization=True)
            else:
                print(f"Creating new vector store at {self.vector_store_path}")
                # Create an empty vector store
                empty_texts = ["placeholder"]
                vector_store = FAISS.from_texts(empty_texts, self.embedding_model)
                vector_store.save_local(self.vector_store_path)
                return vector_store
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            # Return empty vector store
            empty_texts = ["placeholder"]
            return FAISS.from_texts(empty_texts, self.embedding_model)
    
    def _initialize_answers_db(self) -> Dict:
        """Initialize or load the local answers database."""
        try:
            if os.path.exists(self.local_db_path):
                print(f"Loading existing answers database from {self.local_db_path}")
                with open(self.local_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"Creating new answers database at {self.local_db_path}")
                answers_db = {"qa_pairs": []}
                with open(self.local_db_path, 'w', encoding='utf-8') as f:
                    json.dump(answers_db, f, ensure_ascii=False, indent=2)
                return answers_db
        except Exception as e:
            print(f"Error initializing answers database: {str(e)}")
            return {"qa_pairs": []}
    
    def _save_answers_db(self):
        """Save the answers database to disk."""
        try:
            with open(self.local_db_path, 'w', encoding='utf-8') as f:
                json.dump(self.answers_db, f, ensure_ascii=False, indent=2)
            print(f"Saved answers database to {self.local_db_path}")
        except Exception as e:
            print(f"Error saving answers database: {str(e)}")
    
    def query(self, query: str, match_threshold: float = 0.7, match_count: int = 5) -> Dict[str, Any]:
        """Search for similar entries in the vector store."""
        try:
            if len(self.answers_db["qa_pairs"]) == 0:
                return {"source": "local_vector", "results": []}
            
            # Search the vector store
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query, k=match_count
            )
            
            # Print raw scores for debugging
            print("DEBUG - Raw similarity scores:")
            for doc, score in docs_and_scores:
                print(f"- Doc ID: {doc.metadata.get('id', 'unknown')}, Raw score: {score}")
            
            # Filter by threshold and format results
            results = []
            for doc, score in docs_and_scores:
                # Proper conversion depends on FAISS metric
                # For L2 distance: Convert to similarity in [0,1] range
                # Higher score = closer distance = better match
                distance = float(score)
                # Convert distance to similarity (1 means identical, 0 means completely different)
                similarity = 1.0 / (1.0 + distance)
                
                print(f"DEBUG - Doc ID: {doc.metadata.get('id', 'unknown')}, Raw distance: {distance}, Converted similarity: {similarity}")
                
                if similarity >= match_threshold:
                    # Get the metadata for this document
                    doc_id = doc.metadata.get("id")
                    if doc_id:
                        for qa_pair in self.answers_db["qa_pairs"]:
                            if qa_pair["id"] == doc_id:
                                results.append({
                                    "id": qa_pair["id"],
                                    "content": f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}",
                                    "similarity": similarity,
                                    "metadata": {
                                        "created_at": qa_pair["created_at"],
                                        "query": qa_pair["question"]
                                    }
                                })
            
            # Sort by similarity score in descending order
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {"source": "local_vector", "results": results}
        except Exception as e:
            print(f"Error querying vector store: {str(e)}")
            return {"source": "local_vector", "error": str(e), "results": []}

    
    def add(self, question: str, answer: str) -> Dict[str, Any]:
        """Add a new QA pair to the vector store and metadata database."""
        try:
            # Create a unique ID
            import uuid
            doc_id = str(uuid.uuid4())
            
            # Create a new QA pair
            qa_pair = {
                "id": doc_id,
                "question": question,
                "answer": answer,
                "created_at": datetime.now().isoformat()
            }
            
            # Add to local database
            self.answers_db["qa_pairs"].append(qa_pair)
            self._save_answers_db()
            
            # Add to vector store
            doc_text = f"Question: {question}\nAnswer: {answer}"
            self.vector_store.add_texts(
                [doc_text], 
                metadatas=[{"id": doc_id}]
            )
            
            # Save the updated vector store
            self.vector_store.save_local(self.vector_store_path)
            
            return {
                "source": "local_vector",
                "status": "success",
                "message": "Added query and answer to database",
                "id": doc_id
            }
        except Exception as e:
            print(f"Error adding to vector store: {str(e)}")
            return {
                "source": "local_vector",
                "status": "error",
                "message": f"Failed to add to database: {str(e)}"
            }

# Initialize the vector store
vector_store = LocalVectorStore()

@tool
def local_vector_retrieve(query: str, match_threshold: float = 0.2, match_count: int = 5) -> Dict[str, Any]:
    """
    Retrieve similar documents from a local vector store.

    Args:
        query: The text query to embed and use for similarity search.
        match_threshold: Minimum similarity score (0–1) required to return a match.
        match_count: Maximum number of similar documents to return.

    Returns:
        A dict with:
          - "source": always "local_vector"
          - "results": list of matching entries (empty if none pass the threshold)
    """
    return vector_store.query(query, match_threshold, match_count)

@tool
def add_to_local_vector(query: str, answer: str) -> Dict[str, Any]:
    """
    Add a new query-answer pair to the local vector database.
    
    Args:
        query: The question that was asked
        answer: The answer to store
        
    Returns:
        A dict with information about the operation's success
    """
    return vector_store.add(query, answer)

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

@tool
def web_search(query: str) -> dict:
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

def create_agent(provider: str = "google") -> ToolCallingAgent:
    model = get_llm(provider)
    tools = [
        local_vector_retrieve,
        add_to_local_vector,
        wiki_search,
        arxiv_search,
        web_search,
        multiply,
        add,
        subtract,
        divide,
        modulus,
    ]
    
    agent = ToolCallingAgent(tools=tools, model=model)
    agent.system_prompt = system_prompt
    print(agent.system_prompt)
    return agent

def check_search_results(response: str) -> bool:
    """Check if any search results were found in the response"""
    # This is a simple check - you might need to adjust based on your agent's response format
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

# Updated main logic for agent.py with dataset answer format consistency

if __name__ == "__main__":
    agent = create_agent("google")
    print("Agent created. Type 'exit' to quit.")
    
    # Threshold for high-quality matches
    HIGH_RELEVANCE_THRESHOLD = 0.6
    
    while True:
        query = input("\nYour question: ")
        if query.lower() in ("exit", "quit"):
            break
        
        try:
            # First check if the query exists in local vector store
            local_res = local_vector_retrieve(query, match_threshold=0.6, match_count=5)
            
            # Debug - Print all results with their scores
            if local_res.get("results"):
                for i, result in enumerate(local_res["results"]):
                    similarity = result.get('similarity', 0)
                    doc_id = result.get('id', 'unknown')
                    print(f"- {doc_id} (Similarity: {similarity:.4f}): {result.get('content', '')[:100]}…")
            
            # Check if we have a high-quality match
            high_quality_match = False
            if local_res.get("results"):
                best_match = local_res["results"][0]
                similarity = best_match.get('similarity', 0)
                
                # If similarity is above our threshold, use the stored answer
                if similarity >= HIGH_RELEVANCE_THRESHOLD:
                    high_quality_match = True
                    print(f"\n[From Local Vector Store - High Confidence Match]")
                    # Extract just the answer part from the content
                    content = best_match.get('content', '')
                    if "Answer: " in content:
                        answer = content.split("Answer: ", 1)[1]
                        print(f"Response: {answer}")
                    else:
                        print(f"Response: {content}")
            
            # If no high-quality match, use the agent
            if not high_quality_match:
                print("\n[Searching and generating answer...]")
                
                # Call the agent to get a new answer
                resp = agent.run(query)
                print(f"\nResponse: {resp}")
                
                # Check if the answer already exists in a similar form
                # Use a text-based similarity check to avoid duplicates
                from difflib import SequenceMatcher
                is_duplicate = False
                duplicate_id = None
                
                if local_res.get("results"):
                    for result in local_res["results"]:
                        existing_content = result.get('content', '')
                        if "Answer: " in existing_content:
                            existing_answer = existing_content.split("Answer: ", 1)[1]
                        else:
                            existing_answer = existing_content
                            
                        # Check text similarity between the new and existing answer
                        answer_similarity = SequenceMatcher(None, resp.lower(), existing_answer.lower()).ratio()
                        
                        # If answers are very similar, don't add a duplicate
                        if answer_similarity > 0.9:
                            is_duplicate = True
                            duplicate_id = result.get('id', 'unknown')
                            break
                
                # Only add to vector store if not a duplicate
                if not is_duplicate:
                    print("\n[Adding to local vector store for future reference]")
                    add_result = add_to_local_vector(query, resp)
                    if add_result.get("status") == "success":
                        print(f"- {add_result.get('message')} with ID: {add_result.get('id', 'unknown')}")
                    else:
                        print(f"- Error: {add_result.get('message')}")
                else:
                    print(f"\n[Not adding to vector store - similar answer already exists with ID: {duplicate_id}]")
                    
        except Exception as e:
            print(f"\n[Error]: An unexpected error occurred: {str(e)}")
            print("Please try again or check your configuration.")