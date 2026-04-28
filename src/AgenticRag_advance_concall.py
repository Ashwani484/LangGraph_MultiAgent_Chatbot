import os
import json
import pandas as pd
from typing import Optional
from dotenv import load_dotenv
from log import setup_logger
from paths import *

logger=setup_logger(LOG_DIR)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from langchain.agents import create_agent,AgentState
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage


from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.messages import RemoveMessage
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from typing import Any


from llm import init_llm


load_dotenv()

llm=init_llm("openAI")
# ==========================================

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver,MemorySaver
#from langgraph.store.memory import InMemoryStore
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever

load_dotenv()

# ==========================================
# 1. Configuration & Role Mapping
# ==========================================
# Map user-friendly roles to exact speaker names per company
ROLE_MAPPING = {
    "BEL": {
        "cfo": "Damodar Bhattad",
        "cmd": "Manoj Jain",
        "chairman": "Manoj Jain"
    },
    "HAL": {
        "cfo": "Barenya Senapati",
        "cmd": "D.K. Sunil",
        "chairman": "D.K. Sunil"
    },
    "TCS": {
        "cfo": "Samir Seksaria",
        "ceo": "K Krithivasan",
        "md": "K Krithivasan"
    }
}

# Global variables to hold our databases
df_structured = None
pdr_retriever = None  # Updated to hold the Parent Document Retriever

# ==========================================
# 2. Database Initialization (PDR + Pandas)
# ==========================================
def initialize_databases(json_filepath: str):
    """Loads the JSON data into a Pandas DataFrame and a Parent Document Retriever."""
    try:
        global df_structured, pdr_retriever
        
        with open(json_filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Flatten the nested JSON structure
        flat_data = []
        for company_array in raw_data:
            for entry in company_array:
                if "speaker" in entry and "content" in entry and "company" in entry:
                    flat_data.append(entry)
                    
        # --- 1. Create the Structured Database (Pandas) ---
        df_structured = pd.DataFrame(flat_data)
        logger.info("Structured database (DataFrame) initialized.")
        
        # --- 2. Create the Parent Document Retriever ---
        documents = []
        for item in flat_data:
            doc = Document(
                page_content=item["content"],
                metadata={
                    "speaker": item["speaker"],
                    "company": item["company"],
                    "page": item.get("page", 0)
                }
            )
            documents.append(doc)
            
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Setup Splitters
        # Child: Small for granular, precise matching
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        # Parent: Larger to keep the Q&A context together
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
        
        # Storage for Parent Documents
        # Note: Use LocalFileStore instead of InMemoryStore for persistent production storage
        store = InMemoryStore() 
        
        # Initialize FAISS for the Child Documents
        vectorstore = FAISS.from_texts(["initialization_placeholder"], embeddings)
        
        pdr_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        
        logger.info(f"Building Parent-Child relationships for {len(documents)} speaker turns...")
        pdr_retriever.add_documents(documents)
        logger.info("Parent Document Retriever initialized successfully.")

    except Exception as e:
        logger.error(f"Error initializing databases: {e}")
        
# ==========================================
# 3. Agent Tools
# ==========================================
@tool
def list_unique_speakers(company: Optional[str] = None) -> str:
    """
    Use this tool ONLY when the user asks to list, count, or name the speakers 
    or participants in the conference calls.
    """
    global df_structured
    
    if company:
        company = company.upper()
        filtered_df = df_structured[df_structured['company'] == company]
        if filtered_df.empty:
            return f"No data found for company: {company}"
        speakers = filtered_df['speaker'].unique().tolist()
        return f"Speakers for {company}: {', '.join(speakers)}"
    else:
        result = []
        for comp, group in df_structured.groupby('company'):
            speakers = group['speaker'].unique().tolist()
            result.append(f"{comp}: {', '.join(speakers)}")
        return "\n".join(result)


@tool
def search_concall_insights(query: str, company: Optional[str] = None, role: Optional[str] = None) -> str:
    """
    Use this tool when the user asks for specific information, guidance, plans, 
    or statements made during the conference calls.
    Args:
        query: The semantic search query (e.g., 'new guidance', 'EBITDA margins').
        company: The company ticker if mentioned (e.g., 'TCS', 'BEL', 'HAL').
        role: The role of the person if mentioned (e.g., 'cfo', 'ceo', 'cmd').
    """
    global pdr_retriever
    search_filter = {}
    
    if company:
        company = company.upper()
        search_filter["company"] = company
        
    # Map the role to the exact speaker name to ensure absolute precision
    if company and role:
        role = role.lower()
        company_roles = ROLE_MAPPING.get(company, {})
        exact_speaker_name = company_roles.get(role)
        
        if exact_speaker_name:
            search_filter["speaker"] = exact_speaker_name
            logger.info(f"[Tool Log] Applied Strict Filter -> Company: {company}, Speaker: {exact_speaker_name}")
    
    # CRITICAL: Apply the metadata filter to the underlying child vectorstore
    # We increase 'k' because the filter will remove non-matching chunks AFTER retrieval in FAISS
    pdr_retriever.vectorstore.search_kwargs = {
        "k": 15, 
        "filter": search_filter if search_filter else None
    }
    
    # Invoke the Parent Document Retriever
    # It searches child chunks, but returns the full Parent Document context
    results = pdr_retriever.invoke(query)
    
    if not results:
        return "No relevant insights found in the database for this query."
        
    # Format the retrieved Parent Documents
    formatted_results = "\n\n".join(
        [f"[{doc.metadata.get('company', 'Unknown')} - {doc.metadata.get('speaker', 'Unknown')}]: {doc.page_content}" 
         for doc in results]
    )
    return formatted_results




def llm_explanation(agent, user_query, chat_history=None,tool=None):
    logger.info("Generating LLM output")

    prompt = f"""
    User asked: {user_query}



    TASK:
    You are an expert financial analyst assistant. 
    Use the provided tools to answer questions about company conference calls.
    - If the user wants a list of names or participants, use the `list_unique_speakers` tool.
    - If the user wants to know what was said, guidance, or plans, use the `search_concall_insights` tool.
    """

    try:
        #using thread for handle user's process or session while Inmemory checkpoints
        response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config = {"configurable": {"thread_id": "user_123"}}  # ✅ Correct placement
        )

        #response = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        #logging.info(f"LLM response: {response}")
        return response["messages"][-1].content
    except Exception as e:
        return f"Error generating LLM response: {e}"

# ==========================================
# Memory Trimming Logic
# ==========================================
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 5:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-5:] if len(messages) % 2 == 0 else messages[-6:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    } 

#tools calling
tools = [list_unique_speakers, search_concall_insights]
    
system_prompt = """You are an expert financial analyst assistant. 
Use the provided tools to answer questions about company conference calls.
- If the user wants a list of names or participants, use the `list_unique_speakers` tool.
- If the user wants to know what was said, guidance, or plans, use the `search_concall_insights` tool."""
    
    
def agent_concall(query:str):
    try:    
        agent=create_agent(model=llm,tools=tools,system_prompt=system_prompt,
                        middleware=[trim_messages],
                        checkpointer=MemorySaver() )
        response=llm_explanation(agent,query)
        return response
    except Exception as e:
        logger.error(f"Error in agent_concall execution: {e}")
        return f"Error processing the query: {e}"



# ==========================================
# 4. Agent Setup & Execution
# ==========================================
if __name__ == "__main__":
    # Ensure you point this to the path where you saved the uploaded JSON
    json_path = "knowledgebase/concall_structured.json" 
    initialize_databases(json_path)
    
        
    # --- Test Queries ---
    queries = [
        "what Kavish Parekh told as a speaker",
        "From which company he is belongs"
        
        
    ]
    
    for q in queries:
        logger.info(f"\n==============================================")
        logger.info(f"User Query: {q}")
        logger.info(f"==============================================")
        response = agent_concall(q)
        #logger.info(f"\nFINAL ANSWER:\n{response['output']}\n")
        logger.info(response)

        