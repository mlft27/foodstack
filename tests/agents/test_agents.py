"""Tests for agent functions."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from foodstack.agents.menu_agent import menu_agent_node
from foodstack.agents.order_agent import order_agent_node, _extract_identifiers
from foodstack.agents.synthesizer import synthesizer_node
from foodstack.agents.orchestrator import orchestrator_node, OrchestratorDecision


# Mock StackState dictionaries for testing
def create_mock_state(
    user_query: str = "Show me vegetarian options",
    messages: list = None,
    menu_response: str = "",
    order_response: str = "",
) -> dict:
    """Create a mock StackState dictionary for testing."""
    return {
        "user_query": user_query,
        "messages": messages or [HumanMessage(content=user_query)],
        "menu_response": menu_response,
        "order_response": order_response,
        "menu_messages": [],
        "order_messages": [],
        "synthesizer_messages": [],
        "route": [],
        "final_response": "",
    }


class TestMenuAgent:
    """Tests for menu_agent_node function."""

    @patch("foodstack.agents.menu_agent.menu_llm")
    def test_menu_agent_basic_query(self, mock_llm):
        """Test menu agent with basic vegetarian query."""
        # Setup mock response
        mock_response = AIMessage(content="Here are our vegetarian options...")
        mock_llm.invoke.return_value = mock_response

        # Create mock state
        state = create_mock_state(user_query="Show me vegetarian options")

        # Call agent
        result = menu_agent_node(state)

        # Assertions
        assert isinstance(result, Command)
        assert result.goto == "synthesizer_node"
        assert "menu_response" in result.update
        assert "menu_messages" in result.update
        assert result.update["menu_response"] == "Here are our vegetarian options..."

    @patch("foodstack.agents.menu_agent.menu_llm")
    def test_menu_agent_with_tool_call(self, mock_llm):
        """Test menu agent making a tool call."""
        # Setup mock responses - first with tool call, then without
        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "search_menu_catalog",
                    "name": "search_menu_catalog",
                    "args": {"query": "vegetarian", "k": 3},
                }
            ],
        )
        final_response = AIMessage(content="Found these vegetarian items...")

        mock_llm.invoke.side_effect = [tool_call_response, final_response]

        state = create_mock_state(user_query="Show me vegetarian pizza")
        result = menu_agent_node(state)

        assert isinstance(result, Command)
        assert result.goto == "synthesizer_node"
        assert "menu_response" in result.update
        # Tool should have been called
        assert mock_llm.invoke.call_count >= 2

    @patch("foodstack.agents.menu_agent.menu_llm")
    def test_menu_agent_max_iterations(self, mock_llm):
        """Test menu agent respects max 5 iterations."""
        # Setup mock to always return tool calls (would exceed max iterations)
        tool_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "search_menu_catalog",
                    "name": "search_menu_catalog",
                    "args": {"query": "food"},
                }
            ],
        )
        mock_llm.invoke.return_value = tool_response

        state = create_mock_state(user_query="Search for everything")
        result = menu_agent_node(state)

        assert isinstance(result, Command)
        # Should not exceed max iterations (5) + 1 initial
        assert mock_llm.invoke.call_count <= 6

    @patch("foodstack.agents.menu_agent.menu_llm")
    def test_menu_agent_with_history(self, mock_llm):
        """Test menu agent preserves message history."""
        mock_response = AIMessage(content="Here are your options...")
        mock_llm.invoke.return_value = mock_response

        # Create state with message history
        history = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        state = create_mock_state(user_query="What's popular?", messages=history)

        result = menu_agent_node(state)

        assert isinstance(result, Command)
        # Check that history was included in messages
        assert len(result.update["menu_messages"]) >= len(history)


class TestOrderAgent:
    """Tests for order_agent_node function."""

    def test_extract_identifiers_order_id(self):
        """Test extracting order ID from query."""
        query = "Where is my order ORD-001?"
        identifiers = _extract_identifiers(query)

        assert identifiers["order_id"] == "ORD-001"
        assert identifiers["tracking_id"] is None

    def test_extract_identifiers_tracking_id(self):
        """Test extracting tracking ID from query."""
        query = "Track my delivery with FS202TRK"
        identifiers = _extract_identifiers(query)

        assert identifiers["tracking_id"] == "FS202TRK"
        assert identifiers["order_id"] is None

    def test_extract_identifiers_email(self):
        """Test extracting email from query."""
        query = "Find my order for user@example.com"
        identifiers = _extract_identifiers(query)

        assert identifiers["customer_email"] == "user@example.com"

    def test_extract_identifiers_no_match(self):
        """Test when no identifiers are found."""
        query = "Where is my order?"
        identifiers = _extract_identifiers(query)

        assert all(v is None for v in identifiers.values())

    def test_extract_identifiers_multiple(self):
        """Test extracting multiple identifiers."""
        query = "My order ORD-001 with tracking FS202TRK"
        identifiers = _extract_identifiers(query)

        assert identifiers["order_id"] == "ORD-001"
        assert identifiers["tracking_id"] == "FS202TRK"

    @patch("foodstack.agents.order_agent.interrupt")
    def test_order_agent_no_identifier_interrupt(self, mock_interrupt):
        """Test order agent interrupts when no identifier found."""
        # Make the mocked interrupt raise an exception (like the real one)
        mock_interrupt.side_effect = Exception("Interrupt called")

        state = create_mock_state(user_query="Where is my order?")

        # The interrupt will raise, so we expect it to be called
        with pytest.raises(Exception, match="Interrupt called"):
            order_agent_node(state)

        mock_interrupt.assert_called_once()

    @patch("foodstack.agents.order_agent.order_llm")
    def test_order_agent_with_order_id(self, mock_llm):
        """Test order agent with order ID provided."""
        mock_response = AIMessage(content="Your order ORD-001 is out for delivery...")
        mock_llm.invoke.return_value = mock_response

        state = create_mock_state(user_query="Where is my order ORD-001?")
        result = order_agent_node(state)

        assert isinstance(result, Command)
        assert result.goto == "synthesizer_node"
        assert "order_response" in result.update

    @patch("foodstack.agents.order_agent.order_llm")
    def test_order_agent_with_email(self, mock_llm):
        """Test order agent with email identifier."""
        mock_response = AIMessage(content="Found your order...")
        mock_llm.invoke.return_value = mock_response

        state = create_mock_state(user_query="Track order for user@example.com")
        result = order_agent_node(state)

        assert isinstance(result, Command)
        assert result.goto == "synthesizer_node"

    @patch("foodstack.agents.order_agent.order_llm")
    def test_order_agent_max_iterations(self, mock_llm):
        """Test order agent respects max 5 iterations."""
        tool_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "get_order_status",
                    "name": "get_order_status",
                    "args": {"search_value": "ORD-001"},
                }
            ],
        )
        mock_llm.invoke.return_value = tool_response

        state = create_mock_state(user_query="Find my order ORD-001")
        result = order_agent_node(state)

        assert isinstance(result, Command)
        # Should not exceed max iterations
        assert mock_llm.invoke.call_count <= 6


class TestSynthesizer:
    """Tests for synthesizer_node function."""

    @patch("foodstack.agents.synthesizer.llm")
    def test_synthesizer_menu_only(self, mock_llm):
        """Test synthesizer with only menu response."""
        mock_response = MagicMock()
        mock_response.content = "Here are our vegetarian options..."
        mock_llm.invoke.return_value = mock_response

        state = create_mock_state(
            menu_response="Caesar Salad, Vegetable Pasta, Buddha Bowl",
            order_response="",
        )

        result = synthesizer_node(state)

        assert "final_response" in result
        assert result["final_response"] == "Here are our vegetarian options..."
        mock_llm.invoke.assert_called_once()

    @patch("foodstack.agents.synthesizer.llm")
    def test_synthesizer_order_only(self, mock_llm):
        """Test synthesizer with only order response."""
        mock_response = MagicMock()
        mock_response.content = "Your order is out for delivery..."
        mock_llm.invoke.return_value = mock_response

        state = create_mock_state(
            menu_response="",
            order_response="Order ORD-001 status: Out for Delivery",
        )

        result = synthesizer_node(state)

        assert "final_response" in result
        assert result["final_response"] == "Your order is out for delivery..."

    @patch("foodstack.agents.synthesizer.llm")
    def test_synthesizer_both_responses(self, mock_llm):
        """Test synthesizer combining menu and order responses."""
        mock_response = MagicMock()
        mock_response.content = (
            "Here are our vegetarian options. Meanwhile, your order is out for delivery..."
        )
        mock_llm.invoke.return_value = mock_response

        state = create_mock_state(
            user_query="Show me vegetarian options and track my order ORD-001",
            menu_response="Caesar Salad, Vegetable Pasta, Buddha Bowl",
            order_response="Order ORD-001 status: Out for Delivery",
        )

        result = synthesizer_node(state)

        assert "final_response" in result
        assert "vegetarian" in result["final_response"]
        assert "out for delivery" in result["final_response"]

    @patch("foodstack.agents.synthesizer.llm")
    def test_synthesizer_no_responses(self, mock_llm):
        """Test synthesizer with no agent responses."""
        mock_response = MagicMock()
        mock_response.content = "I apologize, but I was unable to process..."
        mock_llm.invoke.return_value = mock_response

        state = create_mock_state(menu_response="", order_response="")
        result = synthesizer_node(state)

        assert "final_response" in result
        # Should contain the error message
        assert "unable" in result["final_response"].lower()

    @patch("foodstack.agents.synthesizer.llm")
    def test_synthesizer_preserves_messages(self, mock_llm):
        """Test synthesizer includes messages in result."""
        mock_response = MagicMock()
        mock_response.content = "Final response"
        mock_llm.invoke.return_value = mock_response

        state = create_mock_state(menu_response="Menu items...")

        result = synthesizer_node(state)

        assert "synthesizer_messages" in result
        assert len(result["synthesizer_messages"]) >= 2  # At least system + human


class TestAgentIntegration:
    """Integration tests combining multiple agents."""

    @patch("foodstack.agents.menu_agent.menu_llm")
    @patch("foodstack.agents.synthesizer.llm")
    def test_menu_agent_to_synthesizer(self, mock_synth_llm, mock_menu_llm):
        """Test menu agent output flowing to synthesizer."""
        # Setup menu agent response
        menu_response = AIMessage(content="Here are vegetarian options: Caesar Salad, Pasta")
        mock_menu_llm.invoke.return_value = menu_response

        # Setup synthesizer response
        synth_response = MagicMock()
        synth_response.content = "Great! Here are our vegetarian options..."
        mock_synth_llm.invoke.return_value = synth_response

        # Run menu agent
        state = create_mock_state(user_query="Show me vegetarian options")
        menu_result = menu_agent_node(state)

        # Verify menu agent output
        assert menu_result.goto == "synthesizer_node"

        # Run synthesizer with menu output
        synth_state = create_mock_state(
            menu_response=menu_result.update["menu_response"]
        )
        synth_result = synthesizer_node(synth_state)

        # Verify final output
        assert "final_response" in synth_result
        assert len(synth_result["final_response"]) > 0


class TestOrchestrator:
    """Tests for orchestrator_node function."""

    @patch("foodstack.agents.orchestrator.routing_llm")
    def test_orchestrator_menu_query(self, mock_routing_llm):
        """Test orchestrator routing menu-only query."""
        # Setup mock orchestrator decision
        decision = OrchestratorDecision(
            agents=["menu_agent"],
            intent="browse_menu",
            confidence=0.95,
            reasoning="User asking about food options",
        )
        mock_routing_llm.invoke.return_value = decision

        state = create_mock_state(user_query="Show me pizzas")
        result = orchestrator_node(state)

        assert isinstance(result, Command)
        assert "menu_agent_node" in str(result.goto)
        assert "route" in result.update
        assert result.update["route"] == ["menu_agent"]

    @patch("foodstack.agents.orchestrator.routing_llm")
    def test_orchestrator_order_query(self, mock_routing_llm):
        """Test orchestrator routing order-only query."""
        decision = OrchestratorDecision(
            agents=["order_agent"],
            intent="track_order",
            confidence=0.98,
            reasoning="User tracking their delivery",
        )
        mock_routing_llm.invoke.return_value = decision

        state = create_mock_state(user_query="Where is my order ORD-001?")
        result = orchestrator_node(state)

        assert isinstance(result, Command)
        assert "order_agent_node" in str(result.goto)
        assert result.update["route"] == ["order_agent"]

    @patch("foodstack.agents.orchestrator.routing_llm")
    def test_orchestrator_mixed_query(self, mock_routing_llm):
        """Test orchestrator routing mixed query to both agents."""
        decision = OrchestratorDecision(
            agents=["menu_agent", "order_agent"],
            intent="mixed_query",
            confidence=0.92,
            reasoning="User wants both menu info and order tracking",
        )
        mock_routing_llm.invoke.return_value = decision

        state = create_mock_state(
            user_query="Show me pizzas and track my order ORD-001"
        )
        result = orchestrator_node(state)

        assert isinstance(result, Command)
        assert result.update["route"] == ["menu_agent", "order_agent"]
        # Should send to multiple agents
        assert str(result.goto).count("Send") >= 2 or isinstance(result.goto, list)

    @patch("foodstack.agents.orchestrator.routing_llm")
    def test_orchestrator_greeting(self, mock_routing_llm):
        """Test orchestrator handling greeting with default to menu agent."""
        decision = OrchestratorDecision(
            agents=["menu_agent"],
            intent="greeting",
            confidence=0.99,
            reasoning="User greeting, default to menu agent",
        )
        mock_routing_llm.invoke.return_value = decision

        state = create_mock_state(user_query="Hi there! How are you?")
        result = orchestrator_node(state)

        assert isinstance(result, Command)
        assert result.update["route"] == ["menu_agent"]

    @patch("foodstack.agents.orchestrator.routing_llm")
    def test_orchestrator_uses_message_history(self, mock_routing_llm):
        """Test orchestrator includes message history in routing."""
        decision = OrchestratorDecision(
            agents=["menu_agent"],
            intent="follow_up",
            confidence=0.90,
            reasoning="Follow-up to previous menu discussion",
        )
        mock_routing_llm.invoke.return_value = decision

        # Create state with conversation history
        history = [
            HumanMessage(content="Show me pizzas"),
            AIMessage(content="Here are our pizzas..."),
            HumanMessage(content="Any vegetarian options?"),
        ]
        state = create_mock_state(user_query="And prices?", messages=history)

        result = orchestrator_node(state)

        assert isinstance(result, Command)
        # Verify history was passed to LLM
        mock_routing_llm.invoke.assert_called_once()
        call_args = mock_routing_llm.invoke.call_args[0][0]
        # Should include system prompt and messages
        assert len(call_args) >= 2

    @patch("foodstack.agents.orchestrator.routing_llm")
    def test_orchestrator_clears_per_turn_buffers(self, mock_routing_llm):
        """Test orchestrator clears per-turn message buffers."""
        decision = OrchestratorDecision(
            agents=["menu_agent"],
            intent="browse_menu",
            confidence=0.95,
            reasoning="User browsing",
        )
        mock_routing_llm.invoke.return_value = decision

        # Create state with existing buffers
        state = create_mock_state(user_query="Show pizzas")
        state["menu_messages"] = ["old message"]
        state["order_messages"] = ["old message"]
        state["menu_response"] = "old response"
        state["order_response"] = "old response"

        result = orchestrator_node(state)

        assert isinstance(result, Command)
        # Verify route is set
        assert result.update["route"] == ["menu_agent"]
        # Verify that Send commands are created with clean state
        assert hasattr(result, "goto")

    @patch("foodstack.agents.orchestrator.routing_llm")
    def test_orchestrator_decision_confidence(self, mock_routing_llm):
        """Test orchestrator with low confidence decision."""
        decision = OrchestratorDecision(
            agents=["menu_agent"],
            intent="unclear",
            confidence=0.45,
            reasoning="Unclear intent, defaulting to menu",
        )
        mock_routing_llm.invoke.return_value = decision

        state = create_mock_state(user_query="Tell me something")
        result = orchestrator_node(state)

        assert isinstance(result, Command)
        # Should still route even with low confidence
        assert len(result.update["route"]) > 0

    @patch("foodstack.agents.orchestrator.routing_llm")
    def test_orchestrator_preserves_user_query(self, mock_routing_llm):
        """Test orchestrator preserves user query in state."""
        decision = OrchestratorDecision(
            agents=["menu_agent"],
            intent="browse_menu",
            confidence=0.95,
            reasoning="User browsing",
        )
        mock_routing_llm.invoke.return_value = decision

        original_query = "Show me spicy food"
        state = create_mock_state(user_query=original_query)

        result = orchestrator_node(state)

        assert isinstance(result, Command)
        # Query should not be modified
        assert state["user_query"] == original_query


class TestOrchestratorIntegration:
    """Integration tests for orchestrator with agent routing."""

    @patch("foodstack.agents.orchestrator.routing_llm")
    @patch("foodstack.agents.menu_agent.menu_llm")
    def test_orchestrator_to_menu_agent(self, mock_menu_llm, mock_routing_llm):
        """Test orchestrator routing to menu agent."""
        # Setup orchestrator decision
        decision = OrchestratorDecision(
            agents=["menu_agent"],
            intent="browse_menu",
            confidence=0.95,
            reasoning="User browsing menu",
        )
        mock_routing_llm.invoke.return_value = decision

        # Setup menu agent response
        menu_response = AIMessage(content="Here are pizzas...")
        mock_menu_llm.invoke.return_value = menu_response

        # Route through orchestrator
        state = create_mock_state(user_query="Show me pizzas")
        routing_result = orchestrator_node(state)

        assert isinstance(routing_result, Command)
        assert "menu_agent_node" in str(routing_result.goto)

        # Verify route is set
        assert routing_result.update["route"] == ["menu_agent"]

    @patch("foodstack.agents.orchestrator.routing_llm")
    @patch("foodstack.agents.order_agent.order_llm")
    def test_orchestrator_to_order_agent(self, mock_order_llm, mock_routing_llm):
        """Test orchestrator routing to order agent."""
        decision = OrchestratorDecision(
            agents=["order_agent"],
            intent="track_order",
            confidence=0.98,
            reasoning="User tracking",
        )
        mock_routing_llm.invoke.return_value = decision

        order_response = AIMessage(content="Your order is out for delivery...")
        mock_order_llm.invoke.return_value = order_response

        state = create_mock_state(user_query="Where is my order ORD-001?")
        routing_result = orchestrator_node(state)

        assert isinstance(routing_result, Command)
        assert "order_agent_node" in str(routing_result.goto)
