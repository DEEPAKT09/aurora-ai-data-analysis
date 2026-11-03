import pandas as pd

from graph.workflow import build_graph


def test_multi_agent_workflow():

    # Load the sample dataset
    df = pd.read_csv(
        "data/sample_datasets/sales_sample.csv"
    )

    # Build the LangGraph workflow
    graph = build_graph()

    questions = [
        "Give me an overview of this dataset.",
        "Which country generated the highest sales?",
        "Create a chart showing sales by country.",
        "Can you predict future sales?"
    ]

    for index, question in enumerate(questions):

        print("\n" + "=" * 60)
        print("QUESTION:", question)

        result = graph.invoke(
            {
                "user_question": question
            },
            config={
                "configurable": {
                    "thread_id": f"phase6-test-{index}"
                }
            },
            context={
                "dataframe": df
            }
        )

        print("\nSELECTED AGENT:", result.get("route"))
        print("\nANSWER:", result.get("answer"))
        print("\nRESULT:", result.get("analysis_result"))

        print("=" * 60)


if __name__ == "__main__":
    test_multi_agent_workflow()