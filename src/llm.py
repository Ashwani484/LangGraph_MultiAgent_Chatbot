from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as create_react_agent
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from log import setup_logger
from paths import *
logger=setup_logger(LOG_DIR)

load_dotenv()

chat_history=[]

# Initialize Ollama LLM instance
def init_llm(llm):
    try:
        if llm=="openAI":
            llm_instance=ChatOpenAI(
                model="gpt-5.1-2025-11-13", max_tokens=1000
                )
            
            return llm_instance
        if llm=="grok":
            llm_instance = ChatGroq(
                model="openai/gpt-oss-120b", 
                api_key=os.getenv("GROQ_API_KEY"),
                max_tokens=500, 
                temperature=0.5 # Lower temperature is better for factual RAG responses
            )
            return llm_instance
    except Exception as e:
        print("Error in loading llm")
        return None

# Function to generate explanation from RAG response
def llm_explanation(llm_instance, user_query, chat_history=None,tool=None):

    prompt = f"""
    User asked: {user_query}

    History of conversation is as follows:
    {chat_history}

    TASK:
    1. Check if the user query is related to any problem or any topic to discover.
    2. Stay 100% faithful to the technical facts 
    3. Generate output response in 500 tokens or short response
    """

    try:
        #agent=llm_instance.bind_tools(tool)
        agent=create_react_agent(llm_instance,tool)

        response = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        #logging.info(f"LLM response: {response}")
        return response["messages"][-1].content
    except Exception as e:
        return f"Error generating LLM response: {e}"
    

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # Step 3: Generate explanation
    llm = init_llm("grok")
    user_query="tell me about Agentic AI"
    if llm:
        llm_response = llm_explanation(llm, user_query)
        print(llm_response)



