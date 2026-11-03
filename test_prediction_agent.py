
import numpy as np
import pandas as pd

from agents.prediction_agent import run_prediction_agent


def create_regression_dataset():
    np.random.seed(42)

    rows = 100

    experience = np.random.randint(1, 20, rows)
    education = np.random.choice(
        ["Bachelor", "Master", "PhD"],
        rows
    )

    education_bonus = {
        "Bachelor": 5000,
        "Master": 12000,
        "PhD": 20000
    }

    salary = (
        25000
        + experience * 3500
        + pd.Series(education).map(education_bonus).to_numpy()
        + np.random.normal(0, 3000, rows)
    )

    return pd.DataFrame({
        "Experience": experience,
        "Education": education,
        "Salary": salary.round(2)
    })


if __name__ == "__main__":

    df = create_regression_dataset()

    question = (
        "Compare regression models to predict Salary "
        "using the available dataset features."
    )

    print("\nRunning Prediction Agent...\n")

    result = run_prediction_agent(
        user_question=question,
        df=df
    )

    print("\nPREDICTION AGENT RESULT")

    print("Success:", result["success"])
    print("Task:", result.get("task_type"))
    print("Target:", result.get("target_column"))
    print("Engines:", result.get("engines"))
    print("Answer:", result["answer"])

    if result["success"]:
        sklearn_result = result["results"]["scikit-learn"]

        print("\nSCIKIT-LEARN RESULTS")
        print("Best model:", sklearn_result["best_model"])

        for model in sklearn_result["models"]:
            print("\nModel:", model["model"])

            if "error" in model:
                print("Error:", model["error"])
            else:
                print("Metrics:", model["metrics"])
    else:
        print("Error:", result.get("error"))
