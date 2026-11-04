# AURORA AI DATA ANALYSIS

An AI-powered, multi-agent data analysis application built with **Streamlit**, **LangGraph**, **Google Gemini**, and Python's data-science ecosystem. AURORA helps users explore uploaded datasets, ask questions in natural language, generate visualizations, run statistical and advanced analytics, and build predictive models through a single web interface.

> **Project status:** Active development. The application is designed to run locally and can be deployed to Render from the GitHub `main` branch.

## Features

### Dataset upload and profiling
- Upload datasets in **CSV** and **Excel (`.xlsx`)** formats.
- Validate uploaded files and normalize column names.
- Review dataset dimensions, column information, missing values, data types, and data-quality summaries.

### Data analysis
- Explore numerical and categorical columns.
- Review descriptive statistics and categorical summaries.
- Inspect correlations between numerical variables.
- Explore individual columns through the interactive column explorer.

### Natural-language data analyst
- Ask questions about the uploaded dataset using natural language.
- Route requests to specialist agents through a LangGraph workflow.
- Generate responses grounded in the available dataset and analysis results.

### Visualizations
- Generate data visualizations from the uploaded dataset.
- Explore common chart types, including distributions, comparisons, relationships, and time-based patterns where the selected data supports them.

### Advanced analytics
- Outlier detection.
- Correlation analysis.
- Statistical hypothesis testing, including supported t-tests, ANOVA, chi-square, and normality testing.
- Trend and seasonality analysis.
- Time-series anomaly detection.
- Comprehensive analysis with evidence-backed findings and recommendations.

### Predictive analytics
- Classification and regression using scikit-learn models.
- Model comparison and evaluation using task-appropriate metrics.
- Time-series forecasting using **Holt-Winters exponential smoothing** (statsmodels).

### Observability
- Optional LangSmith tracing for inspecting and debugging LLM and LangGraph execution.

## Technology stack

| Area | Technologies |
|---|---|
| Web application | Streamlit |
| Agent orchestration | LangGraph, LangChain |
| LLM integration | Google Gemini through `langchain-google-genai` |
| LLM observability | LangSmith (optional) |
| Data manipulation | Pandas, NumPy |
| Visualization | Plotly, Streamlit charts |
| Machine learning | scikit-learn |
| Statistical analysis and forecasting | SciPy, statsmodels |
| Excel support | openpyxl |
| Configuration | python-dotenv |
| Testing | pytest |
| Deployment | Render |

## Application architecture

AURORA uses a supervisor-led LangGraph workflow. A user request is routed to a specialist node based on the task. The workflow includes routes for profiling, general analysis, advanced analytics, visualization, prediction, and unsupported requests.

```text
                         User
                           |
                           v
                    Streamlit interface
                           |
                           v
                    LangGraph workflow
                           |
                           v
                       Supervisor
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
      Profiling         Analysis       Advanced analytics
          |                |                |
          +----------------+----------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
       Visualization                 Prediction
             |                           |
             +-------------+-------------+
                           |
                           v
                    Results in UI
```

The graph uses an in-memory checkpointer in the current workflow configuration. This means conversational/workflow checkpoint state is not a persistent database and should not be assumed to survive application restarts.

## Repository structure

```text
ai-data-analyst/
├── agents/                 # Supervisor and specialist agent logic
├── assets/                 # Application logo and favicon
├── config/                 # Environment and application settings
├── data/
│   ├── processed/          # Processed-data directory
│   ├── sample_datasets/    # Example datasets
│   └── uploads/            # Upload-related directory
├── graph/                  # LangGraph state, nodes, and workflow
├── memory/                 # Memory-related application modules
├── ml_models/              # Tabular ML and forecasting implementations
├── monitoring/             # Monitoring-related modules
├── tests/                  # Advanced analytics and integration tests
├── tools/                  # Profiling, statistics, visualization, and analytics tools
├── ui/                     # Streamlit dashboard and UI panels
├── utils/                  # Shared utility modules
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Python dependencies
└── .env.example            # Environment-variable template
```

## Getting started

### 1. Prerequisites

- Python **3.12** is the development environment used for this project.
- Git.
- A Google AI Studio API key for Gemini-powered functionality.
- A LangSmith API key only if you want tracing enabled.

### 2. Clone the repository

```bash
git clone https://github.com/DEEPAKT09/aurora-ai-data-analysis.git
cd aurora-ai-data-analysis
```

### 3. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure environment variables

Copy `.env.example` to `.env` in the project root.

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Edit `.env` and add your own credentials:

```dotenv
# Google AI Studio
GOOGLE_API_KEY=your_google_ai_studio_api_key

# Optional LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ai-data-analysis

# Optional: override the default Gemini model
GEMINI_MODEL=gemini-3.5-flash-lite
```

Keep `.env` private. **Never commit API keys, tokens, or other secrets to Git.** The `.env.example` file is a template and should contain placeholders only.

### 6. Run the application

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, typically `http://localhost:8501`.

## Using AURORA

1. Start the application.
2. Upload a CSV or `.xlsx` dataset using the dataset uploader.
3. Review the dataset overview and profiling information.
4. Select the relevant section of the dashboard: Data Analysis, Visualizations, Predictive Analytics, or Advanced Analytics.
5. For conversational analysis, ask a question about the uploaded dataset in the chat interface.
6. For predictive or statistical tasks, select the relevant columns and settings, then run the analysis.
7. Review the generated metrics, visualizations, findings, supporting evidence, and recommendations.

**Data quality matters:** Check column types, missing values, date columns, and target columns before running analysis. Forecasting requires a usable date/time column and a numerical target.

## Testing

Run the test suite from the project root:

```bash
python -m pytest -v
```

Run a specific test module or test by name when debugging:

```bash
python -m pytest tests/test_advanced_analytics.py -v
python -m pytest test_prediction.py::test_classification -v
```

Some integration tests may require valid environment variables or access to the configured LLM provider. Run the relevant tests in an environment with the required credentials configured.

## Deployment on Render

AURORA can be deployed as a Render Web Service connected to the GitHub repository.

Recommended configuration:

| Render setting | Value |
|---|---|
| Runtime | Python |
| Branch | `main` |
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT` |

Add the required environment variables in **Render → Environment**. At minimum, configure `GOOGLE_API_KEY`. Add `LANGSMITH_API_KEY` and the LangSmith settings only if tracing is required.

For automatic deployments, enable **Auto-Deploy** for the connected `main` branch. A push to GitHub will then trigger a deployment on Render.

### Deployment considerations

- Render's filesystem on an ephemeral service should not be treated as permanent storage. Uploaded files and locally written application data may not survive restarts or redeployments.
- The current LangGraph workflow uses in-memory checkpointing. It does not provide durable conversation persistence across application restarts.
- Keep secrets in Render environment variables, not in source code or committed files.
- Check Render build and runtime logs after each deployment.
- If the service uses a free instance, it may spin down after inactivity and take time to start again.

## Security and data handling

- Do not commit `.env`, API keys, access tokens, private datasets, or user-specific credentials.
- Review `.gitignore` before staging files.
- Upload only datasets you are authorized to process.
- Be mindful that LLM-powered analysis may send relevant prompts and data context to the configured model provider. Do not upload confidential or regulated information unless your organization's policies and provider configuration permit it.

## Troubleshooting

| Issue | What to check |
|---|---|
| Gemini authentication error | Confirm `GOOGLE_API_KEY` is present and valid in `.env` or the Render environment. |
| Missing Python package | Activate the intended virtual environment and run `pip install -r requirements.txt`. |
| CSV upload error | Check file encoding, delimiters, headers, and whether the file contains records. |
| Excel upload error | Confirm the file is a valid `.xlsx` workbook. |
| Forecasting error | Confirm the selected date column parses correctly and the target column is numeric with sufficient observations. |
| Render deployment failure | Review the build and runtime logs; verify the build/start commands and environment variables. |
| Changes do not appear on Render | Confirm the commit was pushed to the connected branch and inspect the latest deployment status. |

## Contributing

1. Create a feature branch.
2. Make focused changes and run relevant tests.
3. Check the staged diff before committing.
4. Do not include secrets, local databases, caches, virtual environments, or private datasets.
5. Open a pull request describing the change and test results.

## License

No license information is currently specified in this repository. Unless a license is added, do not assume that the project is open source or that reuse, redistribution, or modification is permitted.

---

**AURORA AI DATA ANALYSIS**  
*Intelligent, evidence-backed data exploration and predictive analytics.*
