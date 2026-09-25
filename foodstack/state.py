"""State definitions for FoodStack agent graph."""

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, AnyMessage
from langgraph.graph import add_messages


class StackState(TypedDict):
    """State passed through the FoodStack agent graph.

    Attributes:
        messages: Conversation history with LLM messages.
        user_query: The user's initial voice/text query.
        route: Routing decision (e.g., 'menu', 'order', 'general').
        menu_response: Response from the menu agent with available items.
        order_response: Response from the order processing agent.
        final_answer: Final response to return to the user.
    """

    messages: list[BaseMessage]
    user_query: str
    route: list[str]
    menu_response: str
    order_response: str
    final_answer: str

    # Agent-local message buffers (isolated via add_messages reducer)
    menu_messages: Annotated[list[AnyMessage], add_messages]
    order_messages: Annotated[list[AnyMessage], add_messages]
