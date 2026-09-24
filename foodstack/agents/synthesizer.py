"""Synthesizer Agent for combining outputs from specialized agents."""

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from foodstack.config import llm
from foodstack.agents.prompts import SYNTHESIZER_PROMPT
from foodstack.logger import logger


def synthesizer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Combine outputs from menu and/or order agents into a coherent response.

    Args:
        state: The current state containing agent responses.

    Returns:
        Updated state with final_response.
    """
    menu_response = state.get("menu_response", "")
    order_response = state.get("order_response", "")
    user_query = state.get("user_query", "")

    logger.info("Synthesizer combining responses - Menu: %s, Order: %s",
                bool(menu_response), bool(order_response))

    # Determine which agents produced responses
    has_menu = bool(menu_response)
    has_order = bool(order_response)

    # Build input for synthesizer based on what responses we have
    if has_menu and has_order:
        # Both agents responded (mixed query)
        agent_input = f"""Menu Agent Response:
            {menu_response}

            Order Agent Response:
            {order_response}

            User's Original Query: {user_query}

            Please merge these responses into one coherent answer."""
    elif has_menu:
        # Only menu agent responded
        agent_input = f"""Menu Agent Response:
            {menu_response}

            Please present this response clearly and naturally."""
    elif has_order:
        # Only order agent responded
        agent_input = f"""Order Agent Response:
            {order_response}

            Please present this response clearly and naturally."""
    else:
        # No responses (shouldn't happen, but handle it)
        logger.warning("No agent responses found in state")
        agent_input = "I apologize, but I was unable to process your request. Please try again."

    # Prepare messages for synthesizer
    messages = [
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(content=agent_input),
    ]

    # Call LLM to synthesize responses
    response = llm.invoke(messages)
    final_response = response.content if hasattr(response, "content") else str(response)

    logger.info("Synthesizer produced final response")

    return {
        "final_response": final_response,
        "synthesizer_messages": messages + [response],
    }
