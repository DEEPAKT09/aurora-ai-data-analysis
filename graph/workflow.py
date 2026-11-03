
import os

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from graph.state import GraphState, GraphContext

from graph.nodes import (
    supervisor_node,
    profiling_node,
    analysis_node,
    advanced_analysis_node,
    visualization_node,
    prediction_node,
    unsupported_node,
)


def select_agent(state: GraphState):
    """Select the specialist agent based on the supervisor's route."""

    route = state.get("route", "unsupported")

    valid_routes = {
        "profiling",
        "analysis",
        "advanced_analysis",
        "visualization",
        "prediction",
        "unsupported",
    }

    return route if route in valid_routes else "unsupported"


def build_graph():
    """
    Build the multi-agent LangGraph workflow.

    Checkpoints are kept in application memory.
    No SQLite or PostgreSQL database is required.
    """

    # Retain the safer message serialization setting.
    os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

    # Temporary in-memory checkpoint storage.
    checkpointer = InMemorySaver()

    builder = StateGraph(
        GraphState,
        context_schema=GraphContext,
    )

    # Register specialist nodes.
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("profiling", profiling_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("advanced_analysis", advanced_analysis_node)
    builder.add_node("visualization", visualization_node)
    builder.add_node("prediction", prediction_node)
    builder.add_node("unsupported", unsupported_node)

    # Workflow entry point.
    builder.add_edge(START, "supervisor")

    # Route to the selected specialist.
    builder.add_conditional_edges(
        "supervisor",
        select_agent,
        {
            "profiling": "profiling",
            "analysis": "analysis",
            "advanced_analysis": "advanced_analysis",
            "visualization": "visualization",
            "prediction": "prediction",
            "unsupported": "unsupported",
        },
    )

    # All specialist agents finish the current workflow.
    for agent in [
        "profiling",
        "analysis",
        "advanced_analysis",
        "visualization",
        "prediction",
        "unsupported",
    ]:
        builder.add_edge(agent, END)

    return builder.compile(checkpointer=checkpointer)
