"""Test the FoodStack graph execution."""

from foodstack.graph import build_graph
from foodstack.state import StackState


def test_graph_invocation_with_query():
    """Test graph invocation with a test query and verify final_answer in result."""
    graph = build_graph()

    # Initial state
    initial_state: StackState = {
        "user_query": "What vegetarian pizzas do you have?",
        "messages": [],
        "menu_messages": [],
        "order_messages": [],
        "menu_response": "",
        "order_response": "",
        "route": "",
        "final_answer": "",
    }

    # Invoke graph with a thread_id for conversation memory
    config = {"configurable": {"thread_id": "test-thread-1"}}
    result = graph.invoke(initial_state, config=config)

    print("\n" + "="*60)
    print("GRAPH INVOCATION RESULT")
    print("="*60)
    print(f"User Query: {initial_state['user_query']}")
    print(f"\nResult keys: {list(result.keys())}")

    # Check all possible response fields
    final_response = (
        result.get("final_response") or
        result.get("final_answer") or
        result.get("synthesizer_response")
    )

    print(f"\nfinal_response: {result.get('final_response', 'N/A')[:100]}")
    print(f"final_answer: {result.get('final_answer', 'N/A')[:100]}")
    print(f"synthesizer_response: {result.get('synthesizer_response', 'N/A')[:100]}")
    print(f"synthesizer_messages: {bool(result.get('synthesizer_messages', []))}")
    print(f"\nMenu Response: {result.get('menu_response', 'N/A')[:100]}...")
    print(f"Order Response: {result.get('order_response', 'N/A')[:100]}...")
    print(f"\nFinal Response: {final_response[:200] if final_response else 'N/A'}...")
    print("="*60)

    # Verify final response exists
    assert final_response is not None, "Result should contain 'final_response', 'final_answer', or 'synthesizer_response'"
    assert len(final_response) > 0, "Final response should not be empty"
    print("\n[PASS] Graph invocation successful!")
    print(f"[PASS] Final response length: {len(final_response)} characters")

    return result


if __name__ == "__main__":
    result = test_graph_invocation_with_query()
    print("\n[PASS] Test passed!")
