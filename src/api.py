from fastapi import FastAPI
import uvicorn
from orchestrator import build_graph
from log import setup_logger
from paths import *
logger=setup_logger(LOG_DIR)

app = FastAPI()

logger.info("API server started and ready to process queries.")
@app.post("/process_query")
async def process_query(user_query: str):
    try:
        graph = build_graph()
        initial_state = {
            "user_query": user_query,
            "intent": None,
            "urgency": None,
            "topic": None,
            "summary": None,
            "questionAnswer": None,
            "code_error": None,
            "laptop": None,
            "stocks": None
        }
        final_state = graph.invoke(initial_state)
        return final_state
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


