import pandas as pd

from ml_models.tabpfn_models import train_tabpfn


def test_tabpfn_regression():

    df = pd.read_csv(
        "data/sample_datasets/regression_sample.csv"
    )

    result = train_tabpfn(
        df=df,
        target_column="Salary",
        task_type="regression"
    )

    print("\nTABPFN REGRESSION")
    print(result)


def test_tabpfn_classification():

    df = pd.DataFrame({
        "Age": list(range(22, 52)),
        "Income": list(range(20000, 80000, 2000)),
        "Purchased": (
            ["No"] * 15 + ["Yes"] * 15
        )
    })

    result = train_tabpfn(
        df=df,
        target_column="Purchased",
        task_type="classification"
    )

    print("\nTABPFN CLASSIFICATION")
    print(result)


if __name__ == "__main__":

    test_tabpfn_regression()
    test_tabpfn_classification()