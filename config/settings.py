
import os

from dotenv import load_dotenv


# Load environment variables from the root .env file
load_dotenv()


# Google AI Studio configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

# LangSmith configuration
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

LANGSMITH_TRACING = os.getenv(
    "LANGSMITH_TRACING",
    "true"
).lower() == "true"

LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "ai-data-analysis"
)


# Application configuration
APP_NAME = "AI Data Analysis"

APP_VERSION = "1.0.0"
