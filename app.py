import streamlit as st

from ui.dashboard import render_dashboard


# ============================================================
# AURORA AI DATA ANALYSIS
# Application configuration
# ============================================================

APP_NAME = "AURORA AI DATA ANALYSIS"
APP_VERSION = "1.0.0"


# ============================================================
# PAGE CONFIGURATION
# Must be the first Streamlit command in this file.
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="assets/aurora_favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL AURORA LIGHT THEME
# Professional white dashboard with stronger contrast.
# ============================================================

AURORA_CSS = """
<style>
:root {
    --aurora-ink: #142033;
    --aurora-muted: #475569;
    --aurora-border: #AAB7C8;
    --aurora-border-strong: #64748B;
    --aurora-blue: #2563EB;
    --aurora-cyan: #0891B2;
    --aurora-purple: #7C3AED;
    --aurora-green: #15803D;
    --aurora-amber: #B45309;
    --aurora-red: #B91C1C;
}

.stApp, [data-testid="stAppViewContainer"] {
    background: #FFFFFF !important;
    color: var(--aurora-ink) !important;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.5rem;
    padding-bottom: 2.2rem;
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
    color: var(--aurora-ink);
}

h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
    font-weight: 750 !important;
    letter-spacing: -0.3px;
}

p, label, [data-testid="stMarkdownContainer"] {
    color: #263449;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #475569 !important;
}

.aurora-gradient-text {
    background: linear-gradient(
        90deg,
        #0891B2,
        #2563EB,
        #7C3AED
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 850;
}


/* ============================================================
   SIDEBAR
============================================================ */

[data-testid="stSidebar"] {
    background: #F8FAFC !important;
    border-right: 1px solid #CBD5E1 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #142033 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 8px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 9px 12px;
    color: #1E293B !important;
    transition: all .18s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #EFF6FF !important;
    border-color: #60A5FA !important;
    box-shadow:
        0 0 0 2px rgba(37,99,235,.10),
        0 4px 12px rgba(37,99,235,.12);
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #EAF2FF !important;
    border: 1px solid #3B82F6 !important;
    color: #1D4ED8 !important;
    font-weight: 700;
}


/* ============================================================
   BUTTONS AND INTERACTIVE CONTROLS
============================================================ */

.stButton > button,
[data-testid="stDownloadButton"] button {
    border: 1.5px solid #94A3B8 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
    color: #172033 !important;
    font-weight: 650 !important;
    transition: all .18s ease !important;
}

.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover {
    border-color: #2563EB !important;
    color: #1D4ED8 !important;
    box-shadow:
        0 0 0 2px rgba(37,99,235,.13),
        0 5px 16px rgba(37,99,235,.16) !important;
    transform: translateY(-1px);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(
        100deg,
        #0891B2,
        #2563EB,
        #6D5AE6
    ) !important;
    color: #FFFFFF !important;
    border: 1px solid #1D4ED8 !important;
}

.stButton > button[kind="primary"]:hover {
    color: #FFFFFF !important;
    box-shadow:
        0 0 0 3px rgba(37,99,235,.16),
        0 6px 18px rgba(37,99,235,.28) !important;
}


/* ============================================================
   METRIC CARDS
============================================================ */

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1.5px solid #94A3B8 !important;
    border-radius: 13px !important;
    padding: 15px 17px !important;
    box-shadow: 0 2px 7px rgba(15,23,42,.07) !important;
    transition: all .18s ease !important;
}

[data-testid="stMetric"]:hover {
    border-color: #3B82F6 !important;
    box-shadow:
        0 0 0 2px rgba(59,130,246,.12),
        0 7px 18px rgba(15,23,42,.11) !important;
    transform: translateY(-2px);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #334155 !important;
    font-weight: 650 !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 800 !important;
}


/* ============================================================
   CARDS, EXPANDERS, TABLES AND CHARTS
============================================================ */

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1.5px solid #94A3B8 !important;
    border-radius: 13px !important;
    box-shadow: 0 2px 8px rgba(15,23,42,.055) !important;
    transition:
        border-color .18s ease,
        box-shadow .18s ease,
        transform .18s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #60A5FA !important;
    box-shadow:
        0 0 0 2px rgba(59,130,246,.09),
        0 7px 18px rgba(15,23,42,.09) !important;
}

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1.5px solid #94A3B8 !important;
    border-radius: 11px !important;
    box-shadow: 0 2px 7px rgba(15,23,42,.045) !important;
    transition: all .18s ease;
}

[data-testid="stExpander"]:hover {
    border-color: #3B82F6 !important;
    box-shadow:
        0 0 0 2px rgba(59,130,246,.10),
        0 5px 14px rgba(15,23,42,.08) !important;
}

[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stPlotlyChart"] {
    border: 1.5px solid #A3B1C2 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(15,23,42,.06) !important;
}


/* ============================================================
   INPUTS AND SELECT BOXES
============================================================ */

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #172033 !important;
    border-radius: 9px !important;
}

[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #94A3B8 !important;
    border-radius: 9px !important;
}

input,
textarea {
    color: #172033 !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,.15) !important;
}


/* ============================================================
   FILE UPLOADER
============================================================ */

[data-testid="stFileUploader"] section {
    background: #FFFFFF !important;
    border: 1.5px dashed #64748B !important;
    border-radius: 11px !important;
    transition: all .18s ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color: #0891B2 !important;
    box-shadow:
        0 0 0 2px rgba(8,145,178,.11),
        0 5px 14px rgba(8,145,178,.10) !important;
}


/* ============================================================
   CHAT INTERFACE
============================================================ */

[data-testid="stChatMessage"] {
    background: #F8FAFC !important;
    border: 1.5px solid #A3B1C2 !important;
    border-radius: 12px !important;
    padding: 12px !important;
    margin-bottom: 10px;
    transition: all .18s ease;
}

[data-testid="stChatMessage"]:hover {
    border-color: #60A5FA !important;
    box-shadow:
        0 0 0 2px rgba(59,130,246,.09),
        0 5px 14px rgba(15,23,42,.08) !important;
}

[data-testid="stChatInput"] {
    border: 1.8px solid #64748B !important;
    border-radius: 14px !important;
    background: #FFFFFF !important;
    box-shadow: 0 2px 9px rgba(15,23,42,.08) !important;
    transition: all .18s ease;
}

[data-testid="stChatInput"]:hover,
[data-testid="stChatInput"]:focus-within {
    border-color: #2563EB !important;
    box-shadow:
        0 0 0 3px rgba(37,99,235,.14),
        0 5px 16px rgba(37,99,235,.12) !important;
}


/* ============================================================
   TABS AND DIVIDERS
============================================================ */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #CBD5E1;
}

.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
    font-weight: 650;
}

.stTabs [aria-selected="true"] {
    color: #1D4ED8 !important;
    border-bottom-color: #2563EB !important;
}

hr {
    border-color: #CBD5E1 !important;
}


/* ============================================================
   FOOTER AND RESPONSIVE SPACING
============================================================ */

.aurora-footer {
    text-align: center;
    color: #64748B;
    font-size: 12px;
    padding: 15px 0 5px;
    border-top: 1px solid #CBD5E1;
    margin-top: 30px;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
"""


# ============================================================
# APPLY THE GLOBAL THEME
# ============================================================

st.markdown(
    AURORA_CSS,
    unsafe_allow_html=True,
)


# ============================================================
# RENDER APPLICATION
# ============================================================

render_dashboard()


# ============================================================
# APPLICATION FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="aurora-footer">
        <span class="aurora-gradient-text">
            AURORA AI DATA ANALYSIS
        </span>
        &nbsp; · &nbsp; Version {APP_VERSION}
        &nbsp; · &nbsp; Intelligent Multi-Agent Analytics Platform
    </div>
    """,
    unsafe_allow_html=True,
)