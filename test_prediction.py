import pandas as pd

from ml_models.tabular_models import train_and_compare_models


def test_regression():

    df = pd.read_csv(
        "data/sample_datasets/regression_sample.csv"
    )

    result = train_and_compare_models(
        df=df,
        target_column="Salary",
        task_type="regression"
    )

    print("\nREGRESSION RESULTS")
    print("=" * 60)

    print("Best Model:", result["best_model"])

    for model in result["models"]:

        print("\nModel:", model["model"])

        if "error" in model:
            print("Error:", model["error"])
        else:
            print("Metrics:", model["metrics"])


def test_classification():

    df = pd.DataFrame({
        "Age": [
            22, 24, 25, 26, 28, 30, 32, 34, 35, 36,
            38, 40, 42, 44, 45, 47, 49, 51, 53, 55,
            57, 59, 61, 63, 65, 67, 69, 71, 73, 75
        ],
        "Income": [
            20000, 22000, 25000, 27000, 30000,
            32000, 35000, 37000, 40000, 42000,
            45000, 47000, 50000, 52000, 55000,
            57000, 60000, 62000, 65000, 67000,
            70000, 72000, 75000, 77000, 80000,
            82000, 85000, 87000, 90000, 92000
        ],
        "Purchased": [
            "No", "No", "No", "No", "No",
            "No", "No", "No", "No", "No",
            "No", "No", "No", "No", "No",
            "Yes", "Yes", "Yes", "Yes", "Yes",
            "Yes", "Yes", "Yes", "Yes", "Yes",
            "Yes", "Yes", "Yes", "Yes", "Yes"
        ]
    })

    result = train_and_compare_models(
        df=df,
        target_column="Purchased",
        task_type="classification"
    )

    print("\nCLASSIFICATION RESULTS")
    print("=" * 60)

    print("Best Model:", result["best_model"])

    for model in result["models"]:

        print("\nModel:", model["model"])
        print(
    "Cross-validation:",
    model.get("cross_validation")
)

        print(
            "Warnings:",
            model.get("error", "None")
        )

        if "error" in model:
            print("Error:", model["error"])
        else:
            print("Metrics:", model["metrics"])

# print("\nFEATURE WARNINGS:")
# print(result.get("feature_warnings"))

if __name__ == "__main__":

    test_regression()
    test_classification()