"""FoodStackAssistant for orchestrating multi-agent food service queries."""

import argparse
import sys
from typing import Any
from uuid import uuid4

from foodstack.graph import build_graph
from foodstack.state import StackState
from foodstack.logger import logger
from langchain_core.messages import HumanMessage
from langgraph.types import Command


class FoodStackAssistant:
    """Multi-agent assistant for handling food ordering and menu queries.

    Wraps the compiled agent graph to orchestrate menu agent, order agent,
    and synthesizer for intelligent responses to food-related queries.
    """

    def __init__(self):
        """Initialize FoodStackAssistant with the compiled agent graph."""
        self.graph = build_graph()
        logger.info("FoodStackAssistant initialized with compiled graph")

    def run(
        self,
        user_query: str,
        thread_id: str | None = None,
        messages: list | None = None,
    ) -> dict[str, Any]:
        """Process a user query through the multi-agent system.

        Args:
            user_query: The user's input query or command.
            thread_id: Optional conversation thread ID for multi-turn conversations.
            messages: Optional conversation history (list of BaseMessage objects).

        Returns:
            Dictionary containing:
                - final_answer: The synthesized response to the user.
                - thread_id: The conversation thread ID for follow-ups.
                - state: Full final state including intermediate responses.
        """
        if thread_id is None:
            thread_id = str(uuid4())

        logger.info(
            "Processing query | Thread: %s | Query: %s",
            thread_id,
            user_query[:60],
        )

        config = {"configurable": {"thread_id": thread_id}}

        initial_state: StackState = {
            "user_query": user_query,
            "messages": messages or [HumanMessage(content=user_query)],
            "route": [],
            "menu_response": "",
            "order_response": "",
            "final_answer": "",
            "menu_messages": [],
            "order_messages": [],
        }

        final_state = self.graph.invoke(initial_state, config=config)

        logger.info(
            "Query processed | Thread: %s | Answer: %s",
            thread_id,
            final_state.get("final_answer", "")[:60],
        )

        return {
            "final_answer": final_state.get("final_answer", ""),
            "thread_id": thread_id,
            "state": final_state,
        }

    def ask(
        self,
        user_query: str,
        thread_id: str | None = None,
        messages: list | None = None,
    ) -> dict[str, Any]:
        """Invoke the graph and handle pending interrupts interactively.

        Invokes the graph, checks for pending interrupts, and handles them by:
        1. Displaying the interrupt question via input()
        2. Collecting the user's answer
        3. Resuming with Command(resume=answer)

        Args:
            user_query: The user's input query or command.
            thread_id: Optional conversation thread ID for multi-turn conversations.
            messages: Optional conversation history (list of BaseMessage objects).

        Returns:
            Dictionary containing:
                - final_answer: The synthesized response to the user.
                - thread_id: The conversation thread ID for follow-ups.
                - state: Full final state including intermediate responses.
                - pending_interrupts: List of pending interrupts (empty after handling).
                - has_interrupts: Boolean indicating if interrupts remain.
        """
        if thread_id is None:
            thread_id = str(uuid4())

        logger.info(
            "Asking graph | Thread: %s | Query: %s",
            thread_id,
            user_query[:60],
        )

        config = {"configurable": {"thread_id": thread_id}}

        initial_state: StackState = {
            "user_query": user_query,
            "messages": messages or [HumanMessage(content=user_query)],
            "route": [],
            "menu_response": "",
            "order_response": "",
            "final_answer": "",
            "menu_messages": [],
            "order_messages": [],
        }

        final_state = self.graph.invoke(initial_state, config=config)

        # Check for and handle pending interrupts
        while True:
            state_snapshot = self.graph.get_state(config)
            pending_interrupts = state_snapshot.next if state_snapshot else []

            if not pending_interrupts:
                logger.info("No pending interrupts, graph execution complete")
                break

            logger.info(
                "Found %d pending interrupt(s), handling interactively",
                len(pending_interrupts),
            )

            # Handle each pending interrupt
            for interrupt in pending_interrupts:
                logger.info("Processing interrupt: %s", interrupt)

                # Extract question from interrupt
                question = str(interrupt)
                if hasattr(interrupt, "question"):
                    question = interrupt.question
                elif isinstance(interrupt, dict) and "question" in interrupt:
                    question = interrupt["question"]

                # Display question and collect answer
                print(f"\n[Interrupt] {question}")
                answer = input("Your answer: ").strip()

                # Resume graph with answer
                logger.info("Resuming graph with answer: %s", answer[:60])
                final_state = self.graph.invoke(
                    Command(resume=answer),
                    config=config,
                )

        logger.info(
            "Graph invocation completed | Thread: %s",
            thread_id,
        )

        return {
            "final_answer": final_state.get("final_answer", ""),
            "thread_id": thread_id,
            "state": final_state,
            "pending_interrupts": [],
            "has_interrupts": False,
        }

    def run_text_loop(self) -> None:
        """Run an interactive REPL for multi-turn conversations.

        Reads user input in a loop, calls ask() for each query, and displays
        responses. Maintains conversation context across turns via thread_id.

        Commands:
            exit/quit: End the loop
            reset: Start a new conversation thread
        """
        thread_id = str(uuid4())
        logger.info("Starting text loop | Thread: %s", thread_id)
        print("\n=== FoodStack Assistant ===")
        print("Commands: 'reset' (new thread), 'quit'/'exit' (exit)\n")

        while True:
            try:
                # Read user input
                user_input = input("You: ").strip()

                # Check for quit commands
                if user_input.lower() in ("exit", "quit", "bye"):
                    logger.info("User initiated exit | Thread: %s", thread_id)
                    print("\nGoodbye!")
                    break

                # Check for reset command
                if user_input.lower() == "reset":
                    thread_id = str(uuid4())
                    logger.info("Conversation reset | New thread: %s", thread_id)
                    print("\n[Conversation reset]\n")
                    continue

                # Skip empty input
                if not user_input:
                    continue

                # Call ask() and display response
                logger.info("Processing user input in loop | Thread: %s", thread_id)
                result = self.ask(user_input, thread_id=thread_id)

                final_answer = result.get("final_answer", "")
                if final_answer:
                    print(f"\nAssistant: {final_answer}\n")
                    logger.info(
                        "Response logged | Thread: %s | Answer: %s",
                        thread_id,
                        final_answer[:60],
                    )
                else:
                    print("\n[No response generated]\n")
                    logger.warning("Empty response received | Thread: %s", thread_id)

            except KeyboardInterrupt:
                logger.info("User interrupted with Ctrl+C | Thread: %s", thread_id)
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(
                    "Error in text loop | Thread: %s | Error: %s",
                    thread_id,
                    str(e),
                )
                print(f"\n[Error] {str(e)}\n")


def main() -> None:
    """Main entry point for FoodStackAssistant CLI."""
    parser = argparse.ArgumentParser(
        description="FoodStack Assistant - Multi-agent food ordering and menu queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m foodstack.main                              # Interactive mode (REPL)
  python -m foodstack.main --query "What's on the menu?" # Single query mode
  python -m foodstack.main -i                           # Explicit interactive mode
        """,
    )

    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive REPL mode (default)",
    )

    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Run a single query and exit (non-interactive mode)",
    )

    args = parser.parse_args()

    # Initialize assistant
    assistant = FoodStackAssistant()
    logger.info("FoodStackAssistant initialized via CLI")

    try:
        if args.query:
            # Single query mode
            logger.info("Running single query mode: %s", args.query[:60])
            result = assistant.ask(args.query)
            final_answer = result.get("final_answer", "")

            if final_answer:
                print(f"\nAssistant: {final_answer}\n")
            else:
                print("\n[No response generated]\n")
                sys.exit(1)
        else:
            # Interactive mode (default)
            logger.info("Running interactive mode")
            assistant.run_text_loop()

    except KeyboardInterrupt:
        logger.info("Application terminated by user (Ctrl+C)")
        print("\n\nTerminated.")
        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error: %s", str(e))
        print(f"\n[Fatal Error] {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
