
from langchain_core.messages import HumanMessage, SystemMessage

from agents.query_planner import (
    get_gemini_model,
    create_query_plan
)

from tools.analytics_tools import execute_analysis


def generate_natural_language_response(
    user_question: str,
    analysis_result: dict
) -> str:
    """
    Convert actual analytical results into a clear explanation.
    """

    model = get_gemini_model()

    system_prompt = """
You are an AI Data Analyst.

Explain the analytical results in clear, simple English.

Rules:

1. Use only the information in the supplied analytical results.
2. Never invent numbers, statistics or dataset facts.
3. Do not claim that an analysis was performed if it failed.
4. Clearly mention the relevant column names.
5. For grouped results, identify the highest or lowest
   group when supported by the supplied values.
6. Explain correlations carefully. Correlation does not
   establish causation.
7. Format numerical results clearly.
8. If the results are insufficient, say so honestly.
"""

    response = model.invoke([
        SystemMessage(content=system_prompt),

        HumanMessage(
            content=(
                f"User question: {user_question}\n\n"
                f"Computed analytical results: {analysis_result}\n\n"
                "Explain these results to the user."
            )
        )
    ])

    return response.content[0]["text"]


def ask_data_analyst(
    user_question: str,
    df
) -> dict:
    """
    Complete natural-language data analysis workflow.
    """

    if df is None or df.empty:
        return {
            "success": False,
            "answer": "Please upload a valid dataset first."
        }

    try:

        # Step 1: Understand the user's question
        plan = create_query_plan(
            user_question,
            df
        )

        # Step 2: Reject unsupported questions
        if plan.operation == "unsupported":

            return {
                "success": True,
                "answer": (
                    "I couldn't map that question to one of "
                    "the analytical operations currently supported. "
                    "Try asking about dataset information, "
                    "column statistics, aggregations, categories, "
                    "missing values or correlations."
                ),
                "operation": "unsupported"
            }

        # Step 3: Execute deterministic analysis
        analysis_result = execute_analysis(
            df,
            plan
        )

        # Step 4: Generate a natural-language explanation
        answer = generate_natural_language_response(
            user_question,
            analysis_result
        )

        return {
            "success": True,
            "answer": answer,
            "operation": plan.operation,
            "analysis_result": analysis_result,
            "plan": plan.model_dump()
        }

    except Exception as error:

        return {
            "success": False,
            "answer": (
                "I encountered an issue while analysing your "
                "question. Please check the selected dataset "
                "and try again."
            ),
            "error": str(error)
        }
