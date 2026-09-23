"""Menu-related tools for the FoodStack agents."""

from langchain_core.tools import tool

from foodstack.tools.rag import get_menu_retriever


@tool
def search_menu_catalog(query: str, k: int = 3) -> str:
    """Search the menu catalog for items matching a query.

    Args:
        query: The search query (e.g., "vegetarian pasta", "spicy tacos").
        k: Number of results to return (default: 3).

    Returns:
        A formatted string with matching menu items and their details.
    """
    retriever = get_menu_retriever(k=k)
    results = retriever.invoke(query)

    if not results:
        return "No matching items found in the menu."

    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(f"{i}. {result.page_content}")

    return "\n\n".join(formatted_results)


# Export tools for agent use
menu_tools = [search_menu_catalog]
