
import uuid
import streamlit as st


def start_new_chat():
    """
    Start a fresh conversation while preserving the uploaded dataset.

    A new LangGraph thread ID prevents the new conversation from
    inheriting the previous conversation's graph checkpoint.
    """

    # Clear conversation messages and recent questions.
    st.session_state["messages"] = []
    st.session_state["recent_queries"] = []

    # Generate a fresh LangGraph conversation thread.
    st.session_state["thread_id"] = str(uuid.uuid4())

    # Clear prediction and analytics results.
    result_keys = [
        "latest_prediction_result",
        "prediction_result",
        "chart_json",
        "chart_plan",
        "advanced_result",
        "analysis_result",
        "latest_analysis_result",
    ]

    for key in result_keys:
        st.session_state.pop(key, None)

    # Clear other temporary conversation-specific selections.
    temporary_keys = [
        "last_query",
        "last_response",
        "current_query",
    ]

    for key in temporary_keys:
        st.session_state.pop(key, None)


def clear_recent_history():
    """Clear only the recent questions displayed in this panel."""
    st.session_state["recent_queries"] = []


def render_history_panel():
    """Display recent questions and temporary conversation controls."""

    st.subheader("Conversation History")

    # Initialize required state safely.
    if "recent_queries" not in st.session_state:
        st.session_state["recent_queries"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    # Conversation controls.
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "＋ New Chat",
            key="history_new_chat",
            use_container_width=True,
        ):
            start_new_chat()
            st.rerun()

    with col2:
        if st.button(
            "Clear History",
            key="history_clear",
            use_container_width=True,
            disabled=not st.session_state["recent_queries"],
        ):
            clear_recent_history()
            st.rerun()

    st.divider()

    recent_queries = st.session_state.get("recent_queries", [])

    if not recent_queries:
        st.info("No questions yet.")
        st.caption(
            "Upload a dataset and start asking questions. "
            "Your recent questions will appear here."
        )
        return

    st.caption(
        f"Showing the latest {min(5, len(recent_queries))} questions"
    )

    # Display the latest five questions, newest first.
    for index, item in enumerate(reversed(recent_queries[-5:])):

        if not isinstance(item, dict):
            continue

        timestamp = item.get("timestamp", "Time unavailable")
        query = item.get("query", "Question unavailable")

        with st.container(border=True):
            st.caption(str(timestamp))
            st.write(str(query))

    st.divider()

    st.caption(f"Total questions: {len(recent_queries)}")
    st.caption("History is temporary and belongs to this active session.")
