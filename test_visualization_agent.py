import pandas as pd

from agents.visualization_agent import generate_visualization


def test_visualization_agent():

    df = pd.read_csv(
        "data/sample_datasets/sales_sample.csv"
    )

    questions = [
        "Create a bar chart showing total sales by country.",
        "Show the distribution of sales using a histogram.",
        "Compare sales and quantity using a scatter plot.",
        "Create a correlation heatmap."
    ]

    for question in questions:

        print("\n" + "=" * 60)

        print("USER QUESTION:")
        print(question)

        try:

            result = generate_visualization(
                question,
                df
            )

            print("\nCHART TYPE:")
            print(result["chart_type"])

            print("\nCHART PLAN:")
            print({
                key: value
                for key, value in result.items()
                if key != "chart_json"
            })

            print("\nCHART JSON GENERATED:")
            print(bool(result["chart_json"]))

        except Exception as error:

            print("\nERROR:")
            print(error)

        print("=" * 60)


if __name__ == "__main__":
    test_visualization_agent()