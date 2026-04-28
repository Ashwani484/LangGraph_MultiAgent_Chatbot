from llm import init_llm, llm_explanation
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from pydantic import BaseModel,Field
from log import setup_logger
from paths import *
from typing import TypedDict, Literal

logger=setup_logger(LOG_DIR)

logger.info("QueryClassifierAgent initialized and ready to classify queries.")

# Define the structure for email classification
class QueryClassifier(TypedDict):
    intent: Literal["question", "CodeError", "laptop", "stocks"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str =Field(description="The topic of the user query")
    summary: str =Field(description="A brief summary of the user query")



# This agent will classify the user query into one of the predefined intents and also determine the urgency, topic, and summary of the query. The output will be structured according to the QueryClassifier TypedDict, which can be easily consumed by downstream nodes in the workflow.
class QueryClassifierAgent:
    def __init__(self, llm_type="openAI"):
        self.llm_instance = init_llm(llm_type)
        self.llm=self.llm_instance.with_structured_output(QueryClassifier)

    def classify_query(self, user_query, chat_history=None):
        #response = llm_explanation(self.llm, user_query, chat_history)
        response = self.llm.invoke(user_query)
        # Here you would parse the response to extract the classification details
        # For simplicity, let's assume the response is a JSON string that can be directly converted to QueryClassifier
        try:
            #print(f"LLM response for classification: {response}")
            return QueryClassifier(response)
        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            return None
    

if __name__ == "__main__":
    agent = QueryClassifierAgent()
    user_query = "what is price of HP laptop"
    classification = agent.classify_query(user_query)
    print(f"Classification Result: {classification}")

