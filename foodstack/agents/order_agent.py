"""Order Agent for handling order-related queries using LangChain tools."""

import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.types import Command, interrupt

from foodstack.config import llm
from foodstack.agents.prompts import ORDER_AGENT_PROMPT
from foodstack.tools.order_tools import order_tools
from foodstack.logger import logger


# Bind tools to the LLM
order_llm = llm.bind_tools(order_tools)

# Create a map of tool functions for execution
tools_map = {tool.name: tool for tool in order_tools}

# Regex patterns for identifier extraction
PATTERNS = {
    "order_id": r"ORD-\d{3}",  # e.g., ORD-001
    "tracking_id": r"FS\d{3}TRK",  # e.g., FS202TRK
    "customer_email": r"[\w\.-]+@[\w\.-]+\.\w+",  # e.g., user@example.com
}


def _extract_identifiers(query: str) -> dict[str, str | None]:
    """Extract order identifiers from user query using regex.

    Args:
        query: The user's query string.

    Returns:
        Dictionary with extracted identifiers (order_id, tracking_id, email).
    """
    identifiers = {}

    for id_type, pattern in PATTERNS.items():
        match = re.search(pattern, query, re.IGNORECASE)
        identifiers[id_type] = match.group(0) if match else None
        if match:
            logger.info("Extracted %s: %s", id_type, identifiers[id_type])

    return identifiers


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name with the given input.

    Args:
        tool_name: Name of the tool to execute.
        tool_input: Input arguments for the tool.

    Returns:
        String result from the tool execution.
    """
    if tool_name not in tools_map:
        return f"Error: Tool '{tool_name}' not found"

    tool = tools_map[tool_name]
    try:
        result = tool.invoke(tool_input)
        logger.info("Tool %s executed successfully", tool_name)
        return str(result)
    except Exception as e:
        logger.error("Tool %s failed: %s", tool_name, str(e))
        return f"Error executing {tool_name}: {str(e)}"


def order_agent_node(state: dict[str, Any]) -> Command:
    """Process a user query using the order agent with tool-calling loop (max 5 iterations).

    Args:
        state: The current state containing user_query and messages.

    Returns:
        Command routing to synthesizer_node with updated state.
    """
    query = state.get("user_query", "")
    history = state.get("messages", [])[-6:]

    logger.info("Order Agent processing: %s", query[:60])

    # Extract identifiers from query
    identifiers = _extract_identifiers(query)
    has_identifier = any(identifiers.values())

    # If no identifier found, interrupt and ask for clarification
    if not has_identifier:
        logger.warning("No order identifier found in query, requesting user input")
        interrupt(
            value="Please provide an order ID (e.g., ORD-001), tracking ID (e.g., FS202TRK), or email address to look up your order."
        )

    # Prepare messages with system prompt
    messages = [
        SystemMessage(content=ORDER_AGENT_PROMPT),
        *history,
        HumanMessage(content=query),
    ]

    # Tool-calling loop (max 5 iterations)
    max_iterations = 5
    for iteration in range(max_iterations):
        logger.info("Order Agent iteration %d/%d", iteration + 1, max_iterations)

        # Call LLM with bound tools
        response = order_llm.invoke(messages)
        messages.append(response)

        # Check if there are tool calls
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            logger.info("No more tool calls, stopping loop")
            break

        # Execute all tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["type"] if isinstance(tool_call, dict) else tool_call.name
            tool_input = tool_call.get("args", {}) if isinstance(tool_call, dict) else tool_call.args

            logger.info("Executing tool: %s with input: %s", tool_name, tool_input)
            tool_result = _execute_tool(tool_name, tool_input)

            # Add tool result to messages
            tool_message = ToolMessage(
                content=tool_result,
                tool_call_id=tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id,
                name=tool_name,
            )
            messages.append(tool_message)

        logger.info("Iteration %d completed, continuing loop", iteration + 1)

    # Extract final response
    final_response = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            final_response = msg.content
            break

    return Command(
        goto="synthesizer_node",
        update={
            "order_response": final_response,
            "order_messages": messages,
        },
    )