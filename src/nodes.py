import pandas as pd
from langgraph.types import RetryPolicy
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage

from AgenticRag_advance_concall import initialize_databases,agent_concall
from agent_info import agent_information
from classify_query import QueryClassifierAgent
from llm import init_llm,llm_explanation

# --- New Imports for Document Ingestion ---
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader, CSVLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from log import setup_logger
from paths import *

logger=setup_logger(LOG_DIR)



logger.info("Nodes module initialized successfully.")

class Node:
    def __init__(self):
        self.llm = init_llm("openAI")

    def questionAnswer(self, query):
        try:
            json_path = "knowledgebase/concall_structured.json" 
            initialize_databases(json_path)
            response = agent_concall(query)
            #logger.info(f"\nFINAL ANSWER:\n{response['output']}\n")
            logger.info(response)
            return response
        except Exception as e:
            logger.error(f"Error in questionAnswer node: {e}")
            return f"Error in questionAnswer node: {e}"
    
    def codeError(self, query):
        try:
            logger.info("Code error node executed with query:", query)
            response = llm_explanation(self.llm, query)
            logger.info("LLM response for code error:", response)
            return response
        except Exception as e:
            logger.error(f"Error in codeError node: {e}")
            return f"Error in codeError node: {e}"


    def laptop(self, query):
        try:
            logger.info("Laptop node executed with query:", query)
            #response = llm_explanation(self.llm, query)
            #df=pd.read_csv("knowledgebase/laptop_price.csv")
            docs=CSVLoader(file_path="knowledgebase/laptop_price.csv").load()
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            #logger.info("Documents loaded from CSV:", chunks)  # logger.info first 2 documents to verify
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
            db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(chunks)
            results = db.similarity_search(query, k=3)
            response=llm_explanation(self.llm, results)

            logger.info("Laptop data loaded successfully.",results)
            logger.info("LLM response for laptop query:", response)
            return response
        except Exception as e:
            logger.error(f"Error in laptop node: {e}")
            return f"Error in laptop node: {e}"


    def stocks(self, query):
        try:
            response=agent_information(query)
            logger.info("LLM response for stock information:", response)
            return response

        except Exception as e:
            logger.error(f"Error in stocks node: {e}")
            return f"Error in stocks node: {e}"

if __name__ == "__main__":
    logger.info("This is the nodes module. It defines the structure for different nodes in the graph.")
    node = Node()
    node.laptop("HP laptop all models list name and price of 255 G6 Notebook")

