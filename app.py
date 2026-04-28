import streamlit as st
import requests
from src.log import setup_logger
from src.paths import *

logger = setup_logger(LOG_DIR)

logger.info("Streamlit app initialized successfully.")
# URL of your FastAPI backend
API_URL = "http://localhost:8000/process_query"

# Set up the page layout and title
st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖")
st.title("🤖 Multi-Agent Chatbot")
st.caption("Powered by LangGraph and FastAPI")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question, paste an error, or search for laptops/stocks..."):
    
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display an empty placeholder for the assistant's response while loading
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            try:
                # Send the POST request to FastAPI
                # user_query is passed as a query parameter based on your FastAPI setup
                response = requests.post(API_URL, params={"user_query": prompt})
                response.raise_for_status() 
                
                # Retrieve the state dictionary from LangGraph
                final_state = response.json()
                
                # Determine which key holds the final answer based on the intent
                intent = final_state.get("intent")
                assistant_response = ""
                
                if intent == "question":
                    assistant_response = final_state.get("questionAnswer")
                elif intent == "CodeError":
                    assistant_response = final_state.get("code_error")
                elif intent == "laptop":
                    assistant_response = final_state.get("laptop")
                elif intent == "stocks":
                    assistant_response = final_state.get("stocks")
                else:
                    # Fallback if intent is missing or unrecognized
                    assistant_response = f"I couldn't classify that query. State returned: {final_state}"

                # Ensure we don't print "None" if a node failed to populate the dictionary
                if not assistant_response:
                    assistant_response = "The agent processed the request, but returned an empty response."

                # Display the finalized response
                message_placeholder.markdown(assistant_response)
                
            except requests.exceptions.ConnectionError:
                assistant_response = "🚨 **Connection Error:** Could not connect to the FastAPI server. Is it running on port 8000?"
                message_placeholder.error(assistant_response)
            except Exception as e:
                assistant_response = f"🚨 **Error:** {str(e)}"
                message_placeholder.error(assistant_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})