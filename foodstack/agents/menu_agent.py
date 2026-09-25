"""Menu Agent for handling food/menu queries using LangChain tools."""

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from foodstack.config import llm
from foodstack.agents.prompts import MENU_AGENT_PROMPT
from foodstack.tools.menu_tools import menu_tools
from foodstack.logger import logger


# Bind tools to the LLM
menu_llm = llm.bind_tools(menu_tools)

# Create a map of tool functions for execution
tools_map = {tool.name: tool for tool in menu_tools}


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


def menu_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Process a user query using the menu agent with tool-calling loop (max 5 iterations).

    Args:
        state: The current state containing user_query and messages.

    Returns:
        Updated state with menu_response and menu_messages.
    """
    query = state.get("user_query", "")
    history = state.get("messages", [])[-6:]

    logger.info("Menu Agent processing: %s", query[:60])

    # Prepare messages with system prompt
    messages = [
        SystemMessage(content=MENU_AGENT_PROMPT),
        *history,
        HumanMessage(content=query),
    ]

    # Tool-calling loop (max 5 iterations)
    max_iterations = 5
    for iteration in range(max_iterations):
        logger.info("Menu Agent iteration %d/%d", iteration + 1, max_iterations)

        # Call LLM with bound tools
        response = menu_llm.invoke(messages)
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
            "menu_response": final_response,
            "menu_messages": messages,
        },
    )