from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from nodes import Node
from classify_query import QueryClassifierAgent
from typing import TypedDict, Optional
from nodes import Node
from route import Route
from log import setup_logger
from paths import *
logger=setup_logger(LOG_DIR)

logger.info("Orchestrator module initialized successfully.")
class OrchestratorState(TypedDict):
    user_query: str  # Added to pass the user's initial prompt to downstream nodes
    intent: Optional[str]
    urgency: Optional[str]
    topic: Optional[str]
    summary: Optional[str]
    questionAnswer: Optional[str]
    code_error: Optional[str]
    laptop: Optional[str]
    stocks: Optional[str]

def build_graph():
    workflow = StateGraph(OrchestratorState)
    
    # Instantiate your existing classes
    classifier = QueryClassifierAgent()
    node_funcs = Node()

    # 1. Node Wrappers (LangGraph nodes must take 'state' and return state updates)
    def classify_node(state: OrchestratorState):
        query = state.get("user_query")
        classification = classifier.classify_query(query)
        
        if classification:
            return {
                "intent": classification.get("intent"),
                "urgency": classification.get("urgency"),
                "topic": classification.get("topic"),
                "summary": classification.get("summary")
            }
        return END
    
    def questionAnswer(state: OrchestratorState):
        query= state.get("user_query")
        answer=node_funcs.questionAnswer(query)
        return {"questionAnswer": str(answer)}
    
    
    
    def code_error(state: OrchestratorState):
        query= state.get("user_query")
        return {"code_error": node_funcs.codeError(query)}
        
    
    def laptop(state: OrchestratorState):
        query= state.get("user_query")
        return {"laptop": node_funcs.laptop(query)}
        

    def stocks(state: OrchestratorState):
        query= state.get("user_query")
        return {"stocks": node_funcs.stocks(query)}
        
    
    # 2. Add Nodes to Graph
    workflow.add_node("classify_query", classify_node)
    workflow.add_node("questionAnswer", questionAnswer)
    workflow.add_node("code_error", code_error)
    workflow.add_node("laptop", laptop)
    workflow.add_node("stocks", stocks)

    # 3. Add Start Edge
    workflow.add_edge(START, "classify_query")

    # 4. Define Routing Logic 
    def route_query(state: OrchestratorState):
        intent = state.get("intent")
        
        route=Route()
        x_intent=route.get_route(intent)

        return x_intent if x_intent else END  # Fallback if the intent doesn't match
        
    
    logger.info("Graph nodes and edges defined successfully.")
    # 5. Add Conditional Edges
    workflow.add_conditional_edges(
        "classify_query",
        route_query
    )
    logger.info("Conditional edges added successfully.")
    # 6. Add End Edges
    workflow.add_edge("questionAnswer", END)
    workflow.add_edge("code_error", END)
    workflow.add_edge("laptop", END)
    workflow.add_edge("stocks", END)

    return workflow.compile()

if __name__ == "__main__":
    graph = build_graph()
    initial_state = {
        "user_query": "give me the list of concall speakers of TCS" ,
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
    print("Final State after execution:", final_state)  

