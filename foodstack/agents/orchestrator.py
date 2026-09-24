"""Orchestrator for routing user queries to appropriate agents."""

from typing import Literal

from pydantic import BaseModel, Field

from foodstack.agents.prompts import ORCHESTRATOR_PROMPT
from foodstack.config import llm
from foodstack.state import StackState
from foodstack.logger import logger
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command, Send


class OrchestratorDecision(BaseModel):
    """Structured routing decision with agent selection and reasoning."""

    agents: list[Literal["menu_agent", "order_agent"]] = Field(
        ...,
        description="List of agents to route this query to. Can be ['menu_agent'], ['order_agent'], or ['menu_agent', 'order_agent'] for parallel dispatch",
        min_items=1,
        max_items=2,
    )
    intent: str = Field(
        ...,
        description="Classified intent category (e.g., 'browse_menu', 'track_order', 'mixed_query', 'greeting')",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this routing decision (0.0 to 1.0)",
    )
    reasoning: str = Field(
        ...,
        description="Detailed reasoning explaining why these agents were selected and how the query should be handled",
    )

routing_llm = llm.with_structured_output(OrchestratorDecision)

def orchestrator_node(state:StackState) -> Command[Literal["menu_agent_node, order_agent_node"]]:
    """Route a user query to appropriate agent(s).

    Args:
        user_query: The user's input query.

    Returns:
        OrchestratorDecision containing agents to route to and reasoning.
    """
    query = state["user_query"]
    logger.info("Routing query: %s", query[:60])

    # Conversation history is persisted via MemorySaver — just use it
    history = state.get("messages", [])[-6:]

    decision: OrchestratorDecision = routing_llm.invoke([
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        *history,
        HumanMessage(content=query),
    ])
    logger.info("Agents: %s | Reason: %s", decision.agents, decision.reasoning)

    # Reset per-turn buffers BEFORE sending to agents
    clean_state = {
        **state,
        "menu_messages": [],
        "order_messages": [],
        "menu_response": "",
        "order_response": "",
    }

    sends = [Send(f"{agent}_node", clean_state) for agent in decision.agents]
    return Command(goto=sends, update={"route": decision.agents})





