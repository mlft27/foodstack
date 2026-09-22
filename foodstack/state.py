"""State definitions for FoodStack agent graph."""

from typing import TypedDict

from langchain_core.messages import BaseMessage


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
    route: str
    menu_response: str
    order_response: str
    final_answer: str
