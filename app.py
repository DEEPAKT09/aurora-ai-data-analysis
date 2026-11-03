
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
# Professional white dashboard with subtle Aurora accents.
# ============================================================

AURORA_CSS = """
<style>

/* ---------------------------------------------------------
   GLOBAL APPLICATION
--------------------------------------------------------- */

.stApp {
    background: #FFFFFF;
    color: #172033;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

/* ---------------------------------------------------------
   TYPOGRAPHY
--------------------------------------------------------- */

html,
body,
[class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

h1, h2, h3, h4, h5 {
    color: #172033;
    font-weight: 700;
    letter-spacing: -0.35px;
}

p {
    color: #475569;
}

[data-testid="stCaptionContainer"] {
    color: #64748B;
}

/* ---------------------------------------------------------
   AURORA GRADIENT TEXT
--------------------------------------------------------- */

.aurora-gradient-text {
    background: linear-gradient(
        90deg,
        #06B6D4,
        #3B82F6,
        #8B5CF6
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
}

/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

[data-testid="stSidebar"] {
    background: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #172033 !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #64748B;
}

/* Sidebar navigation radio controls */

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 7px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 9px;
    padding: 9px 12px;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #EFF6FF;
}

/* ---------------------------------------------------------
   BUTTONS
--------------------------------------------------------- */

.stButton > button,
[data-testid="stDownloadButton"] button {
    border: 1px solid #CBD5E1;
    border-radius: 9px;
    background: #FFFFFF;
    color: #172033;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover {
    border-color: #8B5CF6;
    color: #6D28D9;
    box-shadow:
        0 3px 12px rgba(59, 130, 246, 0.10),
        0 0 8px rgba(139, 92, 246, 0.08);
}

/* Primary buttons */

.stButton > button[kind="primary"] {
    background: linear-gradient(
        100deg,
        #06B6D4,
        #6366F1,
        #8B5CF6
    );
    color: #FFFFFF;
    border: none;
}

.stButton > button[kind="primary"]:hover {
    color: #FFFFFF;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
}

/* ---------------------------------------------------------
   METRIC CARDS
--------------------------------------------------------- */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 13px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    border-color: #A5B4FC;
    box-shadow:
        0 4px 14px rgba(59, 130, 246, 0.08);
    transform: translateY(-1px);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    color: #172033 !important;
    font-weight: 750;
}

/* ---------------------------------------------------------
   CONTAINERS AND EXPANDERS
--------------------------------------------------------- */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #E2E8F0;
    border-radius: 12px;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 11px;
}

/* ---------------------------------------------------------
   INPUTS AND SELECTBOXES
--------------------------------------------------------- */

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextArea"] textarea {
    background: #FFFFFF;
    color: #172033;
    border: 1px solid #CBD5E1;
    border-radius: 9px;
}

[data-baseweb="select"] > div {
    background: #FFFFFF;
    border-color: #CBD5E1;
    border-radius: 9px;
}

/* Focus state */

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #8B5CF6;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.10);
}

/* ---------------------------------------------------------
   FILE UPLOADER
--------------------------------------------------------- */

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #94A3B8;
    border-radius: 11px;
    padding: 8px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #06B6D4;
}

/* ---------------------------------------------------------
   CHAT INTERFACE
--------------------------------------------------------- */

[data-testid="stChatMessage"] {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
}

[data-testid="stChatInput"] {
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
}

[data-testid="stChatInput"]:focus-within {
    border-color: #8B5CF6;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.12);
}

/* ---------------------------------------------------------
   TABLES
--------------------------------------------------------- */

[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
}

/* ---------------------------------------------------------
   TABS
--------------------------------------------------------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    color: #64748B;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #6D28D9 !important;
    border-bottom-color: #8B5CF6 !important;
}

/* ---------------------------------------------------------
   DIVIDERS
--------------------------------------------------------- */

hr {
    border-color: #E2E8F0;
}

/* ---------------------------------------------------------
   RESPONSIVE SPACING
--------------------------------------------------------- */

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.aurora-footer {
    text-align: center;
    color: #94A3B8;
    font-size: 12px;
    padding: 15px 0 5px 0;
    border-top: 1px solid #E2E8F0;
    margin-top: 30px;
}

/* =====================================================
   AURORA MAIN DASHBOARD – BORDER AND SHADOW POLISH
   Scoped to main content. Sidebar remains untouched.
   ===================================================== */

/* Main dashboard bordered containers */
section.main [data-testid="stVerticalBlockBorderWrapper"] {
    border: 2px solid #334155 !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.10) !important;
}

/* KPI cards */
section.main [data-testid="stMetric"] {
    border: 2px solid #334155 !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.10) !important;
}

/* Expanders */
section.main [data-testid="stExpander"] {
    border: 2px solid #334155 !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.09) !important;
}

/* Tables */
section.main [data-testid="stDataFrame"],
section.main [data-testid="stTable"] {
    border: 2px solid #334155 !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.10) !important;
    overflow: hidden !important;
}

/* Charts */
section.main [data-testid="stPlotlyChart"] {
    border: 2px solid #334155 !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.10) !important;
    padding: 5px !important;
}

/* Chat message boxes */
section.main [data-testid="stChatMessage"] {
    border: 2px solid #334155 !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.09) !important;
    padding: 12px !important;
}

/* =====================================================
   CHAT INPUT – COMPLETE OUTLINE AND AURORA GLOW
   ===================================================== */

/* Outer chat input */
section.main [data-testid="stChatInput"] {
    border: 2px solid #2563EB !important;
    border-radius: 16px !important;
    background: #FFFFFF !important;
    box-shadow:
        0 0 0 2px rgba(6, 182, 212, 0.16),
        0 0 12px rgba(59, 130, 246, 0.16),
        0 4px 12px rgba(15, 23, 42, 0.10) !important;
    box-sizing: border-box !important;
    overflow: visible !important;
}

/* Actual textarea surface */
section.main [data-testid="stChatInput"] [data-baseweb="textarea"] {
    border: 2px solid #334155 !important;
    border-radius: 13px !important;
    background: #FFFFFF !important;
    box-shadow: none !important;
    box-sizing: border-box !important;
}

/* Remove conflicting inner focus outlines */
section.main [data-testid="stChatInput"] textarea {
    outline: none !important;
    box-shadow: none !important;
}

/* Stronger Aurora glow while typing */
section.main [data-testid="stChatInput"]:focus-within {
    border-color: #0891B2 !important;
    box-shadow:
        0 0 0 2px rgba(6, 182, 212, 0.24),
        0 0 16px rgba(59, 130, 246, 0.28),
        0 5px 15px rgba(15, 23, 42, 0.12) !important;
}

/* File uploader – main content only */
section.main [data-testid="stFileUploader"] section {
    border: 2px dashed #334155 !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.08) !important;
}
/* =====================================================
   AURORA – PERMANENT CHAT INPUT OUTLINE
   Keeps all four borders visible without hovering
   ===================================================== */

section.main [data-testid="stChatInput"] {
    position: relative !important;
    border: 2px solid #2563EB !important;
    border-radius: 16px !important;
    outline: none !important;
    box-shadow:
        0 0 0 2px rgba(6, 182, 212, 0.16),
        0 0 12px rgba(59, 130, 246, 0.16),
        0 4px 12px rgba(15, 23, 42, 0.10) !important;
}

/* Draw a permanent four-sided outline above the input */
section.main [data-testid="stChatInput"]::after {
    content: "" !important;
    position: absolute !important;
    inset: 0 !important;
    border: 2px solid #2563EB !important;
    border-radius: 16px !important;
    pointer-events: none !important;
    z-index: 10 !important;
    box-sizing: border-box !important;
}

/* Maintain the Aurora glow on focus */
section.main [data-testid="stChatInput"]:focus-within::after {
    border-color: #0891B2 !important;
    box-shadow: 0 0 10px rgba(6, 182, 212, 0.25) !important;
}

</style>
"""


# Apply the global theme.
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
