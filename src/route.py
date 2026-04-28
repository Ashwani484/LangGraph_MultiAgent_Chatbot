from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from nodes import Node
from log import setup_logger
from paths import *

logger=setup_logger(LOG_DIR)

class Route:
    def __init__(self):
        print("Initializing Route class")

    
    def get_route(self, intent):
        try:
            logger.info("Route class initialized successfully")
            if intent == "question":
                return "questionAnswer"
            elif intent == "CodeError":
                return "codeError"
            elif intent == "laptop":
                return "laptop"
            elif intent == "stocks":
                return "stocks"
            else:
                raise ValueError(f"Unknown intent: {intent}")
            
        except Exception as e:
            logger.error(f"Error in get_route: {e}")
            return None
        
    
