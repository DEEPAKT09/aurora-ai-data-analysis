
import pandas as pd

from agents.ai_analyst import ask_data_analyst


def test_data_analysis():

    df = pd.read_csv(
        "data/sample_datasets/sales_sample.csv"
    )

    questions = [
        "Which country generated the highest sales?",
        "What is the average sales value?",
        "How many records are in the dataset?",
        "Which product appears most frequently?",
        "Are sales and quantity correlated?",
        "Are there any missing values?"
    ]

    for question in questions:

        print("\n" + "=" * 60)

        print("USER QUESTION:")
        print(question)

        result = ask_data_analyst(
            question,
            df
        )

        print("\nAI RESPONSE:")
        print(result["answer"])

        print("\nOPERATION:")
        print(result.get("operation"))

        print("=" * 60)


if __name__ == "__main__":
    test_data_analysis()
