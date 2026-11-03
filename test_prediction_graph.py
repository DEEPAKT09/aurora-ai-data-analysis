import pandas as pd
import numpy as np

from graph.workflow import build_graph


# 1. Create sample dataset
np.random.seed(42)

rows = 100

experience = np.random.randint(1, 15, rows)

education = np.random.choice(
    ["Bachelor", "Master", "PhD"],
    rows
)

education_bonus = {
    "Bachelor": 0,
    "Master": 10000,
    "PhD": 20000
}

salary = (
    25000
    + experience * 5000
    + pd.Series(education).map(education_bonus).to_numpy()
    + np.random.normal(0, 2000, rows)
)

df = pd.DataFrame({
    "Experience": experience,
    "Education": education,
    "Salary": salary
})


# 2. Build LangGraph workflow
graph = build_graph()


# 3. Submit prediction request
question = (
    "Compare regression models to predict Salary "
    "using the available dataset features."
)

result = graph.invoke(
    {"user_question": question},
    config={
        "configurable": {
            "thread_id": "prediction-test-001"
        }
    },
    context={
        "dataframe": df
    }
)


# 4. Display results
print("\nLANGGRAPH PREDICTION TEST")
print("=" * 50)

print("Answer:", result.get("answer"))
print("Route:", result.get("route"))

prediction = result.get("prediction_result", {})

print("\nPrediction Success:", prediction.get("success"))
print("Task:", prediction.get("task_type"))
print("Target:", prediction.get("target_column"))
print("Engines:", prediction.get("engines"))

if prediction.get("results"):
    sklearn_results = prediction["results"].get(
        "scikit-learn", {}
    )

    print("\nBest Model:", sklearn_results.get("best_model"))

    for model in sklearn_results.get("models", []):
        print(
            model.get("model"),
            model.get("metrics", model.get("error"))
        )

if result.get("error"):
    print("\nERROR:", result["error"])