import os
import glob
import pandas as pd
import torch
from dotenv import load_dotenv
import yfinance as yf
from pprint import pprint        
import glob
import re,json
import sys

from langchain_core.documents import Document
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.chains.question_answering import load_qa_chain
from flashrank import Ranker # Must be imported first
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever

from langchain.agents import create_agent,AgentState
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain.tools import tool

from log import setup_logger
from paths import *
from llm import init_llm
logger=setup_logger(LOG_DIR)

FAISS_INDEX_PATH = r"D:\AI-Projects\Self_developed_AI\Stock_FAQ_Multiagentsystem\vectorstore\db_faiss_finance"
llm=init_llm("grok")

def embeding_model():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    return embeddings

def load_vectorDB():
    """Initializes or loads the FAISS vector store globally."""
    logger.info("Checking vector DB directory")

    if os.path.exists(FAISS_INDEX_PATH):
        logger.info("Loading existing FAISS index from disk...")
        global_vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeding_model(),
            allow_dangerous_deserialization=True
        )
    else:
        raise FileNotFoundError(f"No FAISS index found at {FAISS_INDEX_PATH}. Please add documents first.")
        
    return global_vectorstore

def llm_explanation(agent, user_query, chat_history=None,tool=None):
    logger.info("Generating LLM output")

    prompt = f"""
    User asked: {user_query}



    TASK:
    1. Check if the user query is related to financial info of the stock like PE ratio, debt to equity.
    3. Stay 100% faithful to the technical facts in the Artificial Intelligence, RAG, LLM, Agents, Agentic AI.
    4. Generate output response in 500 tokens or short response
    """

    try:
        #agent=llm_instance.bind_tools(tool)
        #agent=create_agent(llm_instance,tool)

        response = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        #logging.info(f"LLM response: {response}")
        return response["messages"][-1].content
    except Exception as e:
        return f"Error generating LLM response: {e}"


@tool
def RAG_context(query:str)->str:
    "query is related to financial info of the stock like PE ratio, debt to equity ratio, promotor holdings and other fincials info"
    logger.info("Loading vector DB for context")
    db=load_vectorDB()

    base_retriever = db.as_retriever(search_kwargs={"k": 15})
    compressor = FlashrankRerank(top_n=3)
    reranker = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    results = reranker.invoke(query)
    logger.info("Retrieved results for context")
    return "\n\n".join([d.page_content for d in results])
    #return results

@tool
def RAG_similarity(query:str)->str:
    "query is related to financial info of the stock like PE ratio, debt to equity ratio, promotor holdings and other fincials info"
    db=load_vectorDB()
    logger.info("Loading similarity search DB")
    results=db.similarity_search(query, k=3)
    logger.info("Retrieved results for similarity search")

    return "\n\n".join([d.page_content for d in results])


def agent_information(query:str):
    agent=create_agent(model=llm,tools=[RAG_context],
                       system_prompt="You are a helpful financial assistant. Be concise and accurate.")
    response=llm_explanation(agent,query)
    return response

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":

    query="what is current price of TCS"

    response=llm_explanation(llm,query)
    logger.info(f"Final response: {response}")
