"""StateGraph for the FoodStack multi-agent orchestration system."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from foodstack.state import StackState
from foodstack.agents.orchestrator import orchestrator_node
from foodstack.agents.menu_agent import menu_agent_node
from foodstack.agents.order_agent import order_agent_node
from foodstack.agents.synthesizer import synthesizer_node


def route_to_agent(state: StackState) -> str:
    """Route to appropriate agent based on orchestrator decision using conditional edges.

    Args:
        state: Current graph state with routing decision.

    Returns:
        Next node name: menu_agent_node, order_agent_node, or synthesizer.
    """
    route = state.get("route", "")

    # Handle both string and list route values
    if isinstance(route, list):
        route = route[0] if route else ""

    if not route:
        return "synthesizer"

    # Route to the appropriate agent node
    return f"{route}_node"


def build_graph() -> StateGraph:
    """Build and return the FoodStack agent graph.

    Args:
        mode: "parallel" for Command + Send() dispatch, "sequential" for conditional edges.

    Returns:
        Compiled StateGraph with MemorySaver checkpointer for multi-turn conversation memory.
    """
    graph = StateGraph(StackState)

    # Add all nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("menu_agent_node", menu_agent_node)
    graph.add_node("order_agent_node", order_agent_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Add edges
    graph.add_edge(START, "orchestrator")

   
    # graph.add_conditional_edges(
    #     "orchestrator",
    #     route_to_agent,
    #     {
    #         "menu_agent_node": "menu_agent_node",
    #         "order_agent_node": "order_agent_node",
    #         "synthesizer": "synthesizer",
    #     },
    # )

    # Edges from agents to synthesizer
    graph.add_edge("menu_agent_node", "synthesizer")
    graph.add_edge("order_agent_node", "synthesizer")

    # Final node
    graph.add_edge("synthesizer", END)

    # Compile with MemorySaver checkpointer for multi-turn conversation memory
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
