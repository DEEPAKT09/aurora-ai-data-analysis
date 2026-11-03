
import base64
import html
from pathlib import Path

import streamlit as st


# ============================================================
# AURORA SIDEBAR CONFIGURATION
# ============================================================

PAGES = [
    "Dashboard",
    "Data Analysis",
    "Visualizations",
    "Predictive Analytics",
    "Advanced Analytics",
]

PAGE_ICONS = {
    "Dashboard": ":material/space_dashboard:",
    "Data Analysis": ":material/query_stats:",
    "Visualizations": ":material/monitoring:",
    "Predictive Analytics": ":material/model_training:",
    "Advanced Analytics": ":material/science:",
}


# ============================================================
# LOGO HELPER
# ============================================================

def get_aurora_logo():
    """Load the Aurora SVG logo as a base64 image."""

    logo_path = (
        Path(__file__).resolve().parent.parent
        / "assets"
        / "aurora_logo.svg"
    )

    if not logo_path.exists():
        return None

    try:
        svg_content = logo_path.read_bytes()
        return base64.b64encode(svg_content).decode("utf-8")
    except OSError:
        return None


# ============================================================
# SIDEBAR STYLING
# ============================================================

def inject_sidebar_styles():
    """Apply professional Aurora styling to the sidebar."""

    st.markdown(
        """
        <style>

        /* ------------------------------------------------
           SIDEBAR BACKGROUND
        ------------------------------------------------ */

        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #FFFFFF 0%,
                #F8FAFC 100%
            );
            border-right: 1px solid #E2E8F0;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.2rem;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-left: 1rem;
            padding-right: 1rem;
        }


        /* ------------------------------------------------
           BRAND AREA
        ------------------------------------------------ */

        .aurora-sidebar-brand {
            text-align: center;
            padding: 8px 5px 14px;
            margin-bottom: 8px;
        }

        .aurora-sidebar-brand img {
            width: 100%;
            max-width: 225px;
            height: auto;
            display: block;
            margin: 0 auto;
        }

        .aurora-sidebar-subtitle {
            color: #64748B;
            font-size: 11px;
            text-align: center;
            letter-spacing: 0.3px;
            margin-top: 5px;
        }


        /* ------------------------------------------------
           SECTION HEADINGS
        ------------------------------------------------ */

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #334155 !important;
            font-size: 14px !important;
            font-weight: 700 !important;
        }


        /* ------------------------------------------------
           NAVIGATION - ONE FULL-WIDTH ROW PER PAGE
        ------------------------------------------------ */

        [data-testid="stSidebar"] [role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            gap: 7px !important;
            width: 100% !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] > label {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-start !important;

            width: 100% !important;
            min-height: 46px !important;
            box-sizing: border-box !important;

            padding: 10px 13px !important;
            margin: 0 !important;

            border: 1px solid transparent !important;
            border-radius: 10px !important;

            background: transparent;
            cursor: pointer !important;

            transition: all 0.2s ease !important;
        }

        /* Keep the icon and text consistently aligned */

        [data-testid="stSidebar"] [role="radiogroup"] label p {
            display: flex !important;
            align-items: center !important;
            gap: 11px !important;

            margin: 0 !important;
            color: #334155 !important;

            font-size: 13px !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
        }

                /* ------------------------------------------------
        ENHANCED MATERIAL ICONS
        ------------------------------------------------ */

        /* Icon sizing and filled appearance */

        [data-testid="stSidebar"] [role="radiogroup"]
        label span[data-testid="stIconMaterial"],
        [data-testid="stSidebar"] [role="radiogroup"]
        label .material-symbols-rounded {

            font-size: 23px !important;

            font-variation-settings:
                'FILL' 1,
                'wght' 600,
                'GRAD' 0,
                'opsz' 24 !important;

            color: #64748B !important;

            line-height: 1 !important;

            min-width: 25px !important;
            width: 25px !important;

            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;

            transition: all 0.2s ease !important;
        }

        /* Hover icon */

        [data-testid="stSidebar"] [role="radiogroup"]
        label:hover span[data-testid="stIconMaterial"],
        [data-testid="stSidebar"] [role="radiogroup"]
        label:hover .material-symbols-rounded {

            color: #0891B2 !important;

            transform: scale(1.08);
        }

        /* Active page icon */

                /* ------------------------------------------------
        AURORA ACTIVE NAVIGATION HIGHLIGHT
        ------------------------------------------------ */

        [data-testid="stSidebar"] [role="radiogroup"]
        label:has(input:checked) {

            background: linear-gradient(
                100deg,
                #E0F7FF 0%,
                #EFF6FF 55%,
                #F3E8FF 100%
            ) !important;

            border: 1px solid #93C5FD !important;

            border-radius: 11px !important;

            box-shadow:
                0 3px 12px rgba(59, 130, 246, 0.12),
                0 0 8px rgba(6, 182, 212, 0.08),
                inset 3px 0 0 #06B6D4 !important;

            transform: translateX(2px);
        }

        /* Selected navigation label */

        [data-testid="stSidebar"] [role="radiogroup"]
        label:has(input:checked) p {
            color: #1E40AF !important;
            font-weight: 700 !important;
        }

        /* Selected navigation icon */

        [data-testid="stSidebar"] [role="radiogroup"]
        label:has(input:checked)
        span[data-testid="stIconMaterial"],
        [data-testid="stSidebar"] [role="radiogroup"]
        label:has(input:checked) .material-symbols-rounded {

            color: #2563EB !important;

            font-variation-settings:
                'FILL' 1,
                'wght' 700,
                'GRAD' 0,
                'opsz' 24 !important;
        }

        /* Smooth hover effect */

        [data-testid="stSidebar"] [role="radiogroup"]
        label:hover:not(:has(input:checked)) {

            background: linear-gradient(
                100deg,
                #F0FDFF,
                #EFF6FF
            ) !important;

            border-color: #BAE6FD !important;
        }   

        /* ------------------------------------------------
           DATASET UPLOADER
        ------------------------------------------------ */

        [data-testid="stSidebar"] [data-testid="stFileUploader"] {
            background: #FFFFFF;
            border: 1px dashed #94A3B8;
            border-radius: 10px;
            padding: 7px;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
            border-color: #06B6D4;
        }


        /* ------------------------------------------------
           DATASET STATUS CARD
        ------------------------------------------------ */

        .aurora-dataset-card {
            background: linear-gradient(
                120deg,
                #F0FDFA,
                #EFF6FF
            );

            border: 1px solid #BAE6FD;
            border-radius: 10px;

            padding: 11px;
            margin-top: 8px;

            overflow-wrap: anywhere;
        }

        .aurora-dataset-name {
            color: #172033;
            font-size: 13px;
            font-weight: 700;
        }

        .aurora-dataset-detail {
            color: #64748B;
            font-size: 11px;
            margin-top: 4px;
        }

        .aurora-status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10B981;
            margin-right: 6px;
        }


        /* ------------------------------------------------
           SIDEBAR FOOTER
        ------------------------------------------------ */

        .aurora-sidebar-footer {
            text-align: center;
            color: #94A3B8;
            font-size: 10px;
            padding: 12px 2px 5px;
        }


        /* ------------------------------------------------
           DIVIDERS
        ------------------------------------------------ */

        [data-testid="stSidebar"] hr {
            border-color: #E2E8F0;
            margin-top: 13px;
            margin-bottom: 13px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR RENDERER
# ============================================================

def render_sidebar():
    """Render Aurora branding, navigation and dataset uploader."""

    inject_sidebar_styles()

    with st.sidebar:

        # ------------------------------------------
        # BRANDING
        # ------------------------------------------

        logo_base64 = get_aurora_logo()

        if logo_base64:
            st.markdown(
                f"""
                <div class="aurora-sidebar-brand">
                    <img
                        src="data:image/svg+xml;base64,{logo_base64}"
                        alt="Aurora AI Data Analysis"
                    />
                    <div class="aurora-sidebar-subtitle">
                        Intelligent Multi-Agent Analytics Platform
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                """
                <div class="aurora-sidebar-brand">
                    <h2 style="
                        background:linear-gradient(
                            90deg,#06B6D4,#3B82F6,#8B5CF6
                        );
                        -webkit-background-clip:text;
                        -webkit-text-fill-color:transparent;
                        font-size:22px;
                        font-weight:800;
                    ">
                        AURORA
                    </h2>
                    <div class="aurora-sidebar-subtitle">
                        AI DATA ANALYSIS
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ------------------------------------------
        # NAVIGATION
        # ------------------------------------------

        st.markdown("### Navigation")

        selected_page = st.radio(
            "Go to",
            options=PAGES,
            format_func=lambda page: (
                f"{PAGE_ICONS[page]} {page}"
            ),
            label_visibility="collapsed",
            key="selected_page",
            horizontal=False,
        )

        st.divider()

        # ------------------------------------------
        # DATASET UPLOAD
        # ------------------------------------------

        st.markdown("### Dataset")

        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=["csv", "xlsx"],
            help=(
                "Upload a CSV or Excel file to begin "
                "your data analysis."
            ),
            key="dataset_uploader",
            label_visibility="visible",
        )

        st.session_state["uploaded_file"] = uploaded_file

        if uploaded_file is not None:

            file_size_mb = uploaded_file.size / (1024 * 1024)
            safe_filename = html.escape(uploaded_file.name)

            st.markdown(
                f"""
                <div class="aurora-dataset-card">
                    <div class="aurora-dataset-name">
                        <span class="aurora-status-dot"></span>
                        Dataset selected
                    </div>
                    <div class="aurora-dataset-detail">
                        {safe_filename}
                    </div>
                    <div class="aurora-dataset-detail">
                        {file_size_mb:.2f} MB
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.info(
                "Upload a dataset to start your analysis."
            )

        # ------------------------------------------
        # FOOTER
        # ------------------------------------------

        st.divider()

        st.markdown(
            """
            <div class="aurora-sidebar-footer">
                <span style="
                    color:#06B6D4;
                    font-weight:700;
                ">
                    AURORA
                </span>
                AI DATA ANALYSIS
                <br/>
                Version 1.0.0
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page, uploaded_file
