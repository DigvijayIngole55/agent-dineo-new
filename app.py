import os
import time
import logging
import traceback
import gradio as gr
import requests
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from agent import build_graph

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------

class BasicAgent:
    """A langgraph agent."""
    def __init__(self):
        logger.info("=== INITIALIZING BASIC AGENT ===")
        try:
            logger.info("Building graph...")
            self.graph = build_graph()
            logger.info("Graph built successfully")
            
            logger.info("Generating graph visualization...")
            img_data = self.graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API,)

            with open('graph.png', "wb") as f:
                f.write(img_data)
            logger.info("Graph visualization saved as graph.png")
            
            logger.info("=== BASIC AGENT INITIALIZED SUCCESSFULLY ===")
            
        except Exception as e:
            logger.error(f"AGENT INITIALIZATION ERROR: {str(e)}")
            logger.error(f"AGENT INITIALIZATION TRACEBACK: {traceback.format_exc()}")
            raise

    def __call__(self, question: str) -> str:
        logger.info("=== AGENT CALL STARTED ===")
        logger.info(f"Question received (length: {len(question)}): {question[:100]}...")
        
        try:
            # time sleep to avoid recursion limit
            logger.info("Sleeping for 20 seconds to avoid recursion limit...")
            time.sleep(20)
            
            logger.info("Creating HumanMessage...")
            messages = [HumanMessage(content=question)]
            
            logger.info("Invoking graph...")
            result = self.graph.invoke({"messages": messages})
            
            logger.info(f"Graph invocation completed. Result type: {type(result)}")
            logger.info(f"Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
            
            if 'messages' in result and result['messages']:
                answer = result['messages'][-1].content
                logger.info(f"Answer extracted (length: {len(answer)}): {answer[:200]}...")
                logger.info("=== AGENT CALL COMPLETED SUCCESSFULLY ===")
                return answer
            else:
                logger.error("No messages found in result")
                logger.error(f"Full result: {result}")
                return "Error: No response generated"
                
        except Exception as e:
            logger.error(f"AGENT CALL ERROR: {str(e)}")
            logger.error(f"AGENT CALL TRACEBACK: {traceback.format_exc()}")
            return f"Agent Error: {str(e)}"

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    logger.info("=== RUN AND SUBMIT ALL STARTED ===")
    
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username = f"{profile.username}"
        logger.info(f"User logged in: {username}")
    else:
        logger.info("User not logged in")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    
    logger.info(f"Questions URL: {questions_url}")
    logger