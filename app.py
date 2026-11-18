"""
app.py
Streamlit Web Interface for the AI Contract Intelligence & Risk Detection System.
Designed with a clean, editorial legal-tech layout suitable for academic demonstrations.
"""

import os
import html
import hashlib
import re
import streamlit as st

from document_utils import extract_text_from_pdf, save_report_files
from analyzer import (
    load_standard_clauses,
    analyze_contract,
    generate_markdown_report,
    generate_json_report,
)
from knowledge_graph import build_nda_knowledge_graph, to_graphviz_dot
from evaluation import run_sample_evaluation
from review_store import (
    load_review_store, build_review_candidates, apply_approved_baseline_matches,
    approve_permanently, semi_approve, promote_pending_to_baseline, remove_pending, reject_remove,
)
from state_space import STATE_LABELS

# Resolve paths relative to this file so the app can be launched from any directory.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

st.set_page_config(
    page_title="Contract Intelligence & Risk Detection System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a refined, editorial legal-tech interface
CUSTOM_CSS = """
<style>
    /* Global layout refinements */
    .stApp {
        background-color: #F7F8FA;
        color: #1E293B;
    }
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 1220px;
    }

    /* Editorial Masthead */
    .masthead {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #1B365D;
        border-radius: 6px;
        padding: 1.35rem 1.6rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .masthead-kicker {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.3rem;
    }
    .masthead-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 0.35rem 0;
        line-height: 1.2;
    }
    .masthead-subtitle {
        font-size: 0.94rem;
        color: #475569;
        margin: 0;
        line-height: 1.5;
    }

    /* Single offline-engine indicator */
    .mode-pill-kw {
        display: inline-block;
        background-color: #F8FAFC;
        color: #334155;
        border: 1px solid #CBD5E1;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        margin-top: 0.65rem;
    }

    /* Metric KPI Cards */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 1rem 1.15rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        height: 100%;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.3rem;
    }

    /* Semantic Risk & Status Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 4px;
        letter-spacing: 0.02em;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
    }

    /* Editorial Summary & Evidence Cards */
    .section-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .section-card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.45rem;
    }
    .evidence-quote {
        background-color: #F8FAFC;
        border-left: 3px solid #1B365D;
        padding: 0.75rem 1rem;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 0.92rem;
        color: #1E293B;
        border-radius: 0 4px 4px 0;
        margin: 0.45rem 0 0.85rem 0;
        line-height: 1.55;
    }
    .disclaimer-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        font-size: 0.82rem;
        color: #475569;
        margin-top: 1.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_risk_badge(risk_level: str) -> str:
    if risk_level == "Low Risk":
        return '<span class="badge badge-low">LOW RISK</span>'
    if risk_level == "Medium Risk":
        return '<span class="badge badge-medium">MEDIUM RISK</span>'
    return '<span class="badge badge-high">HIGH RISK</span>'


def render_status_badge(status: str) -> str:
    if status == "Present":
        return '<span class="badge badge-low">Present</span>'
    if status == "Needs Review":
        return '<span class="badge badge-medium">Needs Review</span>'
    return '<span class="badge badge-high">Missing / Not Detected</span>'


def main():
    # Initialize session state variables
    if "active_pdf_bytes" not in st.session_state:
        st.session_state.active_pdf_bytes = None
    if "active_pdf_name" not in st.session_state:
        st.session_state.active_pdf_name = None
    if "extraction_result" not in st.session_state:
        st.session_state.extraction_result = None
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "dismissed_review_items" not in st.session_state:
        st.session_state.dismissed_review_items = set()
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None

    # Load the core baseline plus the persistent human-approved supplement.
    standards_path = os.path.join(PROJECT_DIR, "data", "nda_standards.json")
    review_store_path = os.getenv(
        "CIRD_APPROVAL_STORE_PATH",
        os.path.join(PROJECT_DIR, "data", "human_approval_store.json"),
    )
    try:
        standards_data = load_standard_clauses(standards_path)
        standards_list = standards_data.get("clauses", [])
        approval_store = load_review_store(review_store_path)
        standards_data["approved_patterns"] = approval_store.get("approved_baseline", [])
    except Exception as exc:
        st.error(f"Critical Configuration Error: Could not load the baseline or human-approval store. {exc}")
        st.stop()

    # -------------------------------------------------------------------------
    # SIDEBAR: Settings, Demo Contract Loader & Reference Library Inspector
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Analysis Engine")
        st.caption("Offline-only contextual screening. No API key, cloud model, or internet connection is required for contract analysis.")
        st.info("All uploaded contract text is processed locally by the built-in screening engine.")
        st.markdown("---")

        st.markdown("### Sample NDA Documents")
        st.caption("Load a built-in sample PDF to test or demonstrate the pipeline immediately:")

        sample_std_path = os.path.join(PROJECT_DIR, "sample_documents", "standard_mutual_nda.pdf")
        sample_risk_path = os.path.join(PROJECT_DIR, "sample_documents", "high_risk_flawed_nda.pdf")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("Standard NDA", use_container_width=True, help="Load a complete, balanced mutual NDA"):
                if os.path.exists(sample_std_path):
                    with open(sample_std_path, "rb") as f:
                        st.session_state.active_pdf_bytes = f.read()
                    st.session_state.active_pdf_name = "standard_mutual_nda.pdf"
                    st.session_state.extraction_result = extract_text_from_pdf(
                        st.session_state.active_pdf_bytes, st.session_state.active_pdf_name
                    )
                    st.session_state.analysis_result = None
                    st.session_state.dismissed_review_items = set()
                else:
                    st.warning("Sample file not found. Run generate_samples.py first.")

        with col_s2:
            if st.button("Flawed NDA", use_container_width=True, help="Load an NDA with missing clauses and perpetual terms"):
                if os.path.exists(sample_risk_path):
                    with open(sample_risk_path, "rb") as f:
                        st.session_state.active_pdf_bytes = f.read()
                    st.session_state.active_pdf_name = "high_risk_flawed_nda.pdf"
                    st.session_state.extraction_result = extract_text_from_pdf(
                        st.session_state.active_pdf_bytes, st.session_state.active_pdf_name
                    )
                    st.session_state.analysis_result = None
                    st.session_state.dismissed_review_items = set()
                else:
                    st.warning("Sample file not found. Run generate_samples.py first.")

        st.markdown("---")
        st.markdown("### Standard NDA Reference Library")
        st.caption("Core baseline: 10 NDA clause categories. Human-approved examples are stored separately and reused for future matching:")
        st.caption(f"Permanent company examples: {len(approval_store.get('approved_baseline', []))} · Semi-approved queue: {len(approval_store.get('pending_approvals', []))}")
        clause_labels = [f"{c['id']}. {c['name']}" for c in standards_list]
        selected_ref_label = st.selectbox("Select Reference Clause", clause_labels, label_visibility="collapsed")
        selected_ref = standards_list[clause_labels.index(selected_ref_label)]

        with st.expander("View Baseline Criteria", expanded=False):
            st.markdown(f"**Weight:** `{selected_ref['weight']}x`")
            st.markdown(f"**Standard Description:** {selected_ref['standard_description']}")
            st.markdown(f"**Expected Purpose:** {selected_ref['expected_purpose']}")
            st.markdown("**Key Elements Checked:**")
            for elem in selected_ref.get("important_elements", []):
                st.markdown(f"- {elem}")

    # -------------------------------------------------------------------------
    # MAIN HEADER MASTHEAD
    # -------------------------------------------------------------------------
    engine_badge = '<span class="mode-pill-kw">Analysis Engine: Offline-Only Contextual Screening</span>'

    st.markdown(
        f"""
        <div class="masthead">
            <div class="masthead-kicker">Preliminary Legal Document Screening &bull; Mutual NDA Module</div>
            <h1 class="masthead-title">AI Contract Intelligence &amp; Risk Detection System</h1>
            <p class="masthead-subtitle">
                Upload a Non-Disclosure Agreement (PDF) to extract clauses, benchmark wording against a 10-point
                standard NDA baseline, detect omissions or unusual deviations, and generate plain-English risk explanations.
            </p>
            {engine_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # STEP 1: CONTRACT UPLOAD & PDF EXTRACTION
    # -------------------------------------------------------------------------
    st.markdown("#### 1. Contract Upload & Document Inspection")
    upload_col, action_col = st.columns([2.2, 1.3], gap="large")

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a text-based NDA PDF document",
            type=["pdf"],
            help="Supported format: text-based PDF (.pdf). Scanned image-only PDFs are detected and flagged.",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            if (
                st.session_state.active_pdf_name != uploaded_file.name
                or st.session_state.active_pdf_bytes != file_bytes
            ):
                st.session_state.active_pdf_bytes = file_bytes
                st.session_state.active_pdf_name = uploaded_file.name
                st.session_state.extraction_result = extract_text_from_pdf(file_bytes, uploaded_file.name)
                st.session_state.analysis_result = None
                st.session_state.dismissed_review_items = set()

    with action_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if st.session_state.active_pdf_name:
            st.markdown(f"**Selected File:** `{st.session_state.active_pdf_name}`")
            ext = st.session_state.extraction_result
            if ext and ext.get("success"):
                st.caption(
                    f"Pages: {ext['page_count']} | Words: {ext['word_count']:,} | "
                    f"Empty Pages: {ext['empty_pages']}"
                )
            elif ext and not ext.get("success"):
                st.caption("Document validation failed — see notice below.")
        else:
            st.markdown("**No PDF Selected**")
            st.caption("Upload a PDF on the left or choose a sample NDA from the sidebar.")

        analyse_clicked = st.button(
            "Analyse Contract",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.active_pdf_bytes is None,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Validate extraction status
    ext_res = st.session_state.extraction_result
    if ext_res and not ext_res.get("success"):
        st.error(f"PDF Extraction Error: {ext_res.get('error')}")
        st.stop()

    # Run analysis when user clicks "Analyse Contract"
    if analyse_clicked and ext_res and ext_res.get("success"):
        with st.spinner("Extracting clauses, comparing against baseline standards, and evaluating risk..."):
            analysis = analyze_contract(
                contract_text=ext_res["full_text"],
                standards_data=standards_data,
            )
            st.session_state.analysis_result = analysis
            st.session_state.dismissed_review_items = set()

    # -------------------------------------------------------------------------
    # STEP 2: RESULTS DASHBOARD
    # -------------------------------------------------------------------------
    raw_analysis = st.session_state.analysis_result
    # Re-read persisted decisions on each rerun so approvals survive browser refreshes/restarts.
    approval_store = load_review_store(review_store_path)
    standards_data["approved_patterns"] = approval_store.get("approved_baseline", [])
    analysis = apply_approved_baseline_matches(raw_analysis, approval_store, standards_list) if raw_analysis and raw_analysis.get("success") else raw_analysis
    if analysis is not None:
        if not analysis.get("success"):
            st.error(f"Analysis Failed: {analysis.get('error')}")
            st.stop()

        st.caption(f"Analysis actually performed: {analysis.get('analysis_mode', 'Unknown engine')}")
        score = analysis["score_summary"]
        clauses = analysis["clauses"]

        st.markdown("---")
        st.markdown("#### 2. Contract Risk Assessment Dashboard")

        # KPI Row
        k1, k2, k3, k4, k5 = st.columns(5)
        score_color = (
            "#166534" if score["overall_category"] == "Low Risk"
            else "#92400E" if score["overall_category"] == "Medium Risk"
            else "#991B1B"
        )

        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Overall Risk Score</div>
                    <div class="kpi-value" style="color:{score_color};">{score['overall_score']} <span style="font-size:1rem;color:#64748B;">/ 100</span></div>
                    <div class="kpi-sub">Screening Index (0–100)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Risk Category</div>
                    <div style="margin-top:0.35rem;">{render_risk_badge(score['overall_category'])}</div>
                    <div class="kpi-sub" style="margin-top:0.55rem;">{score['present_count']} Present &bull; {score['needs_review_count']} Needs Review &bull; {score['missing_count']} Missing</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Clauses Reviewed</div>
                    <div class="kpi-value">{score['clauses_reviewed']}</div>
                    <div class="kpi-sub">Standard NDA Categories</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">High-Risk Findings</div>
                    <div class="kpi-value" style="color:#991B1B;">{score['high_risk_count']}</div>
                    <div class="kpi-sub">Missing or severe deviation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Medium / Low Risk</div>
                    <div class="kpi-value">{score['medium_risk_count']} <span style="font-size:1rem;color:#64748B;">Med</span> / {score['low_risk_count']} <span style="font-size:1rem;color:#64748B;">Low</span></div>
                    <div class="kpi-sub">{score['needs_review_count']} need closer review</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

        # Executive Summary & Top Review Priorities
        sum_col, prio_col = st.columns([1.35, 1.0], gap="large")
        with sum_col:
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="section-card-title">Executive Summary</div>
                    <p style="margin:0; line-height:1.6; font-size:0.95rem; color:#1E293B;">
                        {html.escape(str(analysis['executive_summary']))}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with prio_col:
            priorities_html = "".join(
                f"<li style='margin-bottom:0.4rem; line-height:1.45;'>{html.escape(str(item))}</li>"
                for item in analysis.get("top_review_priorities", [])
            )
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="section-card-title">Recommended Review Priorities</div>
                    <ol style="margin:0; padding-left:1.2rem; font-size:0.92rem; color:#1E293B;">
                        {priorities_html}
                    </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View Transparent Risk Scoring Formula & Methodology"):
            st.write(score["formula_explanation"])
            st.markdown(
                """
                - **Primary score:** Importance-weighted mean of final clause risk labels: Low = 10, Medium = 50, High = 90.
                - **Fuzzy diagnostic:** Uses concept coverage and red-flag strength; Mamdani rules are aggregated with min/max and centroid defuzzification. It does not override the primary score.
                - **CSP:** A finite-domain constraint solver checks status/risk consistency.
                - Neither score is a legal probability or a substitute for human legal review.
                """
            )
            st.caption(f"Score basis: {score.get('score_basis', 'Importance-weighted final clause risk labels')} · Fuzzy diagnostic: {score.get('fuzzy_overall_score', 'N/A')}/100")

        # Structured Analysis Tabs
        tab_clauses, tab_missing, tab_methods, tab_review, tab_text, tab_report = st.tabs([
            "Clause-by-Clause Comparison",
            f"Missing & High-Risk Items ({score['high_risk_count'] + score['missing_count']})",
            "AI Methods & Knowledge Graph",
            "Human Review",
            "Extracted PDF Text",
            "Downloadable Reports",
        ])

        # TAB 1: Clause-by-Clause Comparison
        with tab_clauses:
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                status_filter = st.selectbox(
                    "Filter by Comparison Status",
                    options=["All Statuses", "Missing / Not Detected", "Needs Review", "Present"],
                )
            with f_col2:
                risk_filter = st.selectbox(
                    "Filter by Risk Level",
                    options=["All Risk Levels", "High Risk", "Medium Risk", "Low Risk"],
                )

            filtered_clauses = [
                c for c in clauses
                if (status_filter == "All Statuses" or c["status"] == status_filter)
                and (risk_filter == "All Risk Levels" or c["risk_level"] == risk_filter)
            ]

            # Compact overview table
            summary_rows = [
                {
                    "ID": c["clause_id"],
                    "Clause Category": c["clause_name"],
                    "Comparison Result": c["status"],
                    "Risk Level": c["risk_level"],
                }
                for c in filtered_clauses
            ]
            if summary_rows:
                st.dataframe(summary_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No clauses match the selected filter criteria.")

            st.markdown("##### Detailed Clause Findings")
            for c in filtered_clauses:
                header_label = f"{c['clause_id']}. {c['clause_name']}  —  {c['status']}  ({c['risk_level']})"
                expand_by_default = c["risk_level"] in {"High Risk", "Medium Risk"}
                with st.expander(header_label, expanded=expand_by_default):
                    st.markdown(
                        f"{render_status_badge(c['status'])} &nbsp; {render_risk_badge(c['risk_level'])}",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Extracted Contract Wording / Evidence:**")
                    st.markdown(
                        f'<div class="evidence-quote">{html.escape(str(c["extracted_evidence"]))}</div>',
                        unsafe_allow_html=True,
                    )
                    if c.get("evidence_verified") is False:
                        st.warning("Evidence wording could not be verified as an exact excerpt. Inspect the original PDF before relying on this finding.")
                    elif c.get("evidence_verified") is True:
                        st.caption(c.get("verification_note", "Evidence was traced to extracted PDF text."))
                    if c.get("fuzzy_risk_score") is not None:
                        input_label = "Local concept-match coverage"
                        st.caption(f"Engine fuzzy estimate: {c['fuzzy_risk_score']}/100 ({c.get('fuzzy_risk_level', c['risk_level'])}) · {input_label}: {c.get('coverage_ratio', 0):.0%}")
                    if c.get("human_reviewed"):
                        st.info(f"Human-reviewed category override: {c['risk_level']} ({c.get('human_review_score', 'N/A')}/100). Original engine fuzzy estimate: {c.get('engine_fuzzy_risk_score', c.get('fuzzy_risk_score', 'N/A'))}/100.")
                    if c.get("matched_red_flags"):
                        st.warning("Configured red-flag phrase(s): " + ", ".join(c["matched_red_flags"]))
                    if c.get("csp_solved") is False:
                        st.error(c.get("csp_note", "Constraint check failed; manual review is required."))
                    elif c.get("csp_note"):
                        st.caption(c["csp_note"])
                    if c.get("human_reviewed"):
                        st.info(f"Human-reviewed: {c.get('pre_review_status')} / {c.get('pre_review_risk_level')} → {c['status']} / {c['risk_level']}")
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("**Standard Reference Baseline:**")
                        st.write(c["standard_baseline"])
                        st.markdown("**Comparison & Difference Explanation:**")
                        st.write(c["difference_explanation"])
                    with d2:
                        st.markdown("**Potential Implication:**")
                        st.write(c["potential_implication"])
                        st.markdown("**Suggested Action for Review:**")
                        st.write(c["suggested_action"])

        # TAB 2: Missing & High-Risk Items
        with tab_missing:
            missing_clauses = [c for c in clauses if c["status"] == "Missing / Not Detected"]
            elevated_clauses = [
                c for c in clauses
                if c["risk_level"] in {"High Risk", "Medium Risk"} and c["status"] != "Missing / Not Detected"
            ]

            st.markdown("##### Missing Standard Clauses")
            if missing_clauses:
                for mc in missing_clauses:
                    st.markdown(
                        f"""
                        <div class="section-card" style="border-left:4px solid #991B1B;">
                            <strong>{html.escape(str(mc['clause_id']))}. {html.escape(str(mc['clause_name']))}</strong> &nbsp; {render_risk_badge(mc['risk_level'])}
                            <p style="margin:0.45rem 0 0.3rem 0; font-size:0.92rem;"><strong>Baseline Requirement:</strong> {html.escape(str(mc['standard_baseline']))}</p>
                            <p style="margin:0; font-size:0.92rem; color:#991B1B;"><strong>Recommended Action:</strong> {html.escape(str(mc['suggested_action']))}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success("All 10 standard NDA clause categories were detected in the uploaded contract.")

            st.markdown("##### Present Clauses Flagged for Closer Review")
            if elevated_clauses:
                for ec in elevated_clauses:
                    border_hex = "#991B1B" if ec["risk_level"] == "High Risk" else "#B45309"
                    st.markdown(
                        f"""
                        <div class="section-card" style="border-left:4px solid {border_hex};">
                            <strong>{html.escape(str(ec['clause_id']))}. {html.escape(str(ec['clause_name']))}</strong> &nbsp; {render_status_badge(ec['status'])} &nbsp; {render_risk_badge(ec['risk_level'])}
                            <p style="margin:0.45rem 0 0.3rem 0; font-size:0.92rem;"><strong>Finding:</strong> {html.escape(str(ec['difference_explanation']))}</p>
                            <p style="margin:0; font-size:0.92rem;"><strong>Suggested Action:</strong> {html.escape(str(ec['suggested_action']))}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No present clauses were flagged with elevated deviations.")

        # METHODS TAB: syllabus-aligned, implemented AI techniques
        with tab_methods:
            st.markdown("##### Locally Trained Clause Classifier")
            from learned_clause_model import model_status
            learned_status = model_status()
            if learned_status.get("available"):
                st.success("Local trained model loaded — no API key or network required.")
                st.write(f"Training examples: {learned_status.get('training_examples', 'N/A')} · Held-out examples: {learned_status.get('held_out_examples', 'N/A')} · Held-out accuracy: {learned_status.get('held_out_accuracy', 'N/A')}")
                st.caption("The reported hold-out result is from a small synthetic educational dataset; it is not a reliable estimate of real-world legal accuracy. Low-confidence model matches are ignored; ML-only matches go to human review.")
                st.write("Model: TF-IDF word and character n-grams with Logistic Regression. Training data: ten NDA clause families plus unrelated text.")
                st.code("python train_offline_model.py", language="bash")
            else:
                st.warning(learned_status.get("message", "Local learned model is unavailable; contextual rules remain active."))

            st.markdown("##### Knowledge Representation")
            st.write("The JSON reference library stores each NDA clause as structured knowledge: identifier, description, purpose, important elements, keywords, red-flag indicators and an importance weight.")
            graph = build_nda_knowledge_graph(standards_data)
            st.caption(f"Knowledge graph generated from the reference library: {len(graph['nodes'])} nodes and {len(graph['edges'])} relationships.")
            st.graphviz_chart(to_graphviz_dot(graph), width="stretch")

            st.markdown("##### Fuzzy Logic Risk Assessment")
            st.write("The Mamdani fuzzy inference layer accepts configured keyword-coverage ratio and red-flag strength in the range 0–1 (these are lexical indicators, not proof of legal completeness). Triangular/trapezoidal membership functions represent Low, Medium and High input/output sets. Min is used for rule firing, max for aggregation and centroid defuzzification for a crisp 0–100 clause score.")
            st.caption("Fuzzy inputs are derived locally from clause-specific concept matches, contextual patterns and configured red-flag phrases.")
            st.markdown("**Rule examples**")
            st.markdown("""- Low coverage → High risk.
- High red-flag strength → High risk.
- Medium coverage → Medium risk.
- High coverage AND low red-flag strength → Low risk.""")

            st.markdown("##### Constraint Satisfaction")
            st.write("For each clause i, the variables are Status_i and Risk_i. Status domains are {Present, Needs Review, Missing}; risk domains are {Low, Medium, High}. Constraints require Missing → High Risk, prohibit Present + High Risk, and prohibit Needs Review + Low Risk. A detected red flag also prevents a clause from being labelled Present. The CSP uses AC-3, minimum-remaining-values (MRV), degree tie-breaking and backtracking.")
            csp_solved = sum(1 for c in clauses if c.get("csp_solved", True))
            st.metric("Clause assignments satisfying CSP constraints", f"{csp_solved}/{len(clauses)}")

            st.markdown("##### State-Space Search")
            path = analysis.get("method_details", {}).get("state_space_path", [])
            if not path:
                path = ["START", "PDF_VALIDATED", "TEXT_EXTRACTED", "ANALYSIS_READY", "REPORT_READY"]
            st.write("Breadth-First Search (BFS) traverses the workflow-state graph to find a shortest valid processing route from initial upload to report-ready state. This models workflow planning; it is not the clause-detection algorithm.")
            st.code(" → ".join(STATE_LABELS.get(state, state) for state in path), language="text")
            st.caption("Initial state: START (no document processed) · Goal state: REPORT_READY (report generated) · Actions: validate PDF, extract text, run the local offline analysis engine, aggregate assessment, generate report.")

            st.markdown("##### Intelligent Agent: PEAS Framework")
            st.dataframe([
                {"PEAS element": "Performance measure", "CIRD description": "Relevant clause coverage, correctly flagged red flags, traceable evidence, useful review priorities"},
                {"PEAS element": "Environment", "CIRD description": "Uploaded Mutual NDA PDF and the educational 10-clause reference baseline"},
                {"PEAS element": "Actuators", "CIRD description": "Dashboard findings, risk alerts, report exports and human-review workflow"},
                {"PEAS element": "Sensors", "CIRD description": "PDF upload, extracted page text, local concept patterns and clause-specific context"},
            ], use_container_width=True, hide_index=True)

            st.markdown("##### Implemented Analysis Components")
            st.dataframe([
                {"Component": "Locally trained clause classifier", "Type": "Supervised ML – TF-IDF word/character n-grams and Logistic Regression; uncertain matches abstain"},
                {"Component": "Context-aware offline NLP pattern engine", "Type": "Local deterministic processing – clause concepts, phrases, sentence context and negation"},
                {"Component": "Red-flag detection", "Type": "Local rule-based pattern matching with contextual safeguards"},
                {"Component": "Mamdani fuzzy inference", "Type": "AI reasoning technique – local risk aggregation and defuzzification"},
                {"Component": "CSP label consistency solver", "Type": "AI search/constraint technique – AC-3 and backtracking"},
                {"Component": "Knowledge graph", "Type": "Structured representation of NDA clause concepts and relationships"},
                {"Component": "PDF extraction / Streamlit / exports", "Type": "Local software components"},
                {"Component": "GAN / VAE / image generation models", "Type": "Not applicable to NDA clause classification; excluded to avoid irrelevant complexity"},
            ], use_container_width=True, hide_index=True)

            st.markdown("##### Responsible Use and Limitations")
            st.markdown("""- **Privacy:** Contract text is processed locally by the application. No external AI API is called and no API key is required.
- **Baseline limitations:** The NDA baseline is a simplified reference and may not fit every jurisdiction, industry or negotiation context.
- **Detection limitations:** The local pattern engine may miss unusual paraphrases or flag wording that needs human context; verify every important finding against the original PDF.
- **Human oversight:** Standard low-risk baseline matches may be auto-approved; novel or higher-risk findings enter the permanent-approval / semi-approval / reject workflow. The tool does not replace a qualified lawyer.
- **Scope:** The current knowledge library is designed for Mutual NDAs, not arbitrary legal documents.""")

            st.markdown("##### Built-in Continuous Evaluation")
            st.write("The regression suite checks the two bundled, intentionally authored sample NDAs. This is a functional regression check, not a statistically representative accuracy benchmark.")
            if st.button("Run sample NDA regression evaluation", key="run_eval"):
                st.session_state.evaluation_result = run_sample_evaluation(PROJECT_DIR)
            evaluation = st.session_state.evaluation_result
            if evaluation:
                e1, e2 = st.columns(2)
                e1.metric("Predefined checks passed", f"{evaluation['checks_passed']}/{evaluation['checks_total']}")
                e2.metric("Check pass rate", f"{evaluation['check_pass_rate']}%")
                for case in evaluation["cases"]:
                    with st.expander(f"{case['name']} — {'PASS' if case.get('success') else 'CHECK'}"):
                        if case.get("error"):
                            st.error(case["error"])
                        else:
                            st.caption(f"Observed score: {case['overall_score']}/100 · {case['overall_category']}")
                            for item in case["checks"]:
                                st.write(("✅ " if item["passed"] else "❌ ") + item["check"])
                st.caption(evaluation["interpretation"])

        # HUMAN REVIEW TAB: baseline deviations and new provisions follow a three-outcome workflow.
        with tab_review:
            st.markdown("### Human Approval Workflow")
            st.write(
                "Clauses detected as Present + Low Risk against the current baseline are automatically approved. "
                "New provisions and baseline deviations with Medium/High risk are sent here for a human decision."
            )
            doc_bytes = st.session_state.active_pdf_bytes or b""
            document_hash = hashlib.sha256(doc_bytes).hexdigest()
            review_workflow = build_review_candidates(
                analysis, ext_res.get("full_text", ""), document_hash,
                ext_res.get("filename", "Unknown.pdf"), approval_store, standards_list,
            )
            dismissed = st.session_state.dismissed_review_items or set()
            active_candidates = [item for item in review_workflow["candidates"] if item["id"] not in dismissed]
            analysis["approval_workflow_summary"] = {
                "auto_approved_count": len(review_workflow["auto_approved"]),
                "awaiting_human_approval_count": len(active_candidates),
                "semi_approved_queue_count": len(approval_store.get("pending_approvals", [])),
                "permanent_supplemental_baseline_count": len(approval_store.get("approved_baseline", [])),
                "store_file": "data/human_approval_store.json",
            }

            auto_approved_rows = [
                {"Clause / Provision": item["title"], "Clause ID": item.get("clause_id") or "Additional", "Why auto-approved": item["reason"]}
                for item in review_workflow["auto_approved"]
            ]
            auto_approved_rows.extend([
                {"Clause / Provision": item["title"], "Clause ID": "Additional", "Why auto-approved": item["reason"]}
                for item in review_workflow.get("recognized_additional", [])
            ])
            with st.expander(f"Baseline-matched / auto-approved ({len(auto_approved_rows)})", expanded=False):
                if auto_approved_rows:
                    st.dataframe(auto_approved_rows, use_container_width=True, hide_index=True)
                else:
                    st.caption("No clauses have been auto-approved yet.")

            st.markdown("#### Items requiring human approval")
            if not active_candidates:
                st.success("There are no outstanding findings for this document. Baseline-matched items are auto-approved; semi-approved items are shown in the separate queue below.")
            for item in active_candidates:
                item_id = item["id"]
                label = f"{item['title']} · {item.get('status', '')} · {item.get('risk_level', '')}"
                with st.expander(label, expanded=True):
                    st.caption(f"Type: {'Additional clause not in baseline' if item['source_type'] == 'additional_clause' else 'Existing baseline clause requiring review'} · Page: {item.get('page_number') or 'Not available'}")
                    if item.get("evidence"):
                        st.markdown("**Extracted contract wording**")
                        st.code(item["evidence"], language=None)
                    else:
                        st.warning("No clause text was extracted. This item can be semi-approved for later review or removed, but it cannot be added to the permanent baseline without actual wording.")
                    if item.get("explanation"):
                        st.write(item["explanation"])
                    if item.get("suggested_action"):
                        st.caption(f"Suggested review: {item['suggested_action']}")

                    reviewer_note = st.text_input("Reviewer note (optional)", key=f"approval_note_{item_id}")
                    approved_level = st.selectbox(
                        "Risk level to store if permanently approved",
                        options=["Low Risk", "Medium Risk"],
                        index=1 if item.get("risk_level") in {"High Risk", "Needs Human Assessment"} else 0,
                        key=f"approval_risk_{item_id}",
                        help="This is the accepted risk rating CIRD will apply when materially similar wording is found in a future document.",
                    )
                    a_col, s_col, r_col = st.columns(3)
                    with a_col:
                        if st.button("Approve Permanently", key=f"approve_perm_{item_id}", use_container_width=True, disabled=not item.get("can_add_to_baseline")):
                            item["reviewer_note"] = reviewer_note.strip()
                            try:
                                approve_permanently(item, approved_level, approval_store, review_store_path)
                                st.success("Permanently added to the supplemental baseline. Future materially similar wording can be recognised.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not add to baseline: {exc}")
                    with s_col:
                        if st.button("Semi-Approve (Queue)", key=f"semi_{item_id}", use_container_width=True):
                            item["reviewer_note"] = reviewer_note.strip()
                            semi_approve(item, approval_store, review_store_path)
                            st.success("Saved in the separate approval queue. It does not affect future matching yet.")
                            st.rerun()
                    with r_col:
                        if st.button("Reject / Remove", key=f"reject_{item_id}", use_container_width=True):
                            reject_remove(item, approval_store, review_store_path)
                            st.session_state.dismissed_review_items = set(dismissed) | {item_id}
                            st.rerun()

            st.markdown("---")
            st.markdown(f"### Semi-approved queue ({len(approval_store.get('pending_approvals', []))})")
            st.caption("Semi-approved findings are saved here for a later decision. They are not used for future contract matching until promoted to the permanent baseline.")
            for queued in list(approval_store.get("pending_approvals", [])):
                qid = queued["id"]
                with st.expander(f"{queued.get('title', 'Unnamed item')} · {queued.get('filename', 'Unknown document')}", expanded=False):
                    st.caption(queued.get("queue_status", "Pending approval"))
                    if queued.get("evidence"):
                        st.code(queued["evidence"], language=None)
                    if queued.get("reviewer_note"):
                        st.write(f"Reviewer note: {queued['reviewer_note']}")
                    if queued.get("can_add_to_baseline"):
                        q_risk = st.selectbox("Accepted risk if promoted", ["Low Risk", "Medium Risk"], index=1, key=f"queue_risk_{qid}")
                        q1, q2 = st.columns(2)
                        with q1:
                            if st.button("Promote to Permanent Baseline", key=f"promote_{qid}", use_container_width=True):
                                promote_pending_to_baseline(qid, q_risk, approval_store, review_store_path)
                                st.success("Promoted to the permanent baseline.")
                                st.rerun()
                        with q2:
                            if st.button("Remove from Queue", key=f"remove_queue_{qid}", use_container_width=True):
                                remove_pending(qid, approval_store, review_store_path)
                                st.success("Removed from the approval queue. The permanent baseline was not changed.")
                                st.rerun()
                    else:
                        st.warning("This item has no extracted clause wording, so it cannot be promoted to the permanent baseline.")
                        if st.button("Remove from Queue", key=f"remove_queue_{qid}"):
                            remove_pending(qid, approval_store, review_store_path)
                            st.rerun()

            st.markdown("---")
            st.markdown(f"### Permanent supplemental baseline ({len(approval_store.get('approved_baseline', []))})")
            st.caption("These are human-approved examples that are used in future matching. The original ten-clause educational baseline remains unchanged.")
            with st.expander("View permanently approved examples", expanded=False):
                if approval_store.get("approved_baseline"):
                    st.dataframe([
                        {"Provision": x.get("title"), "Baseline category": x.get("clause_id") or "Additional clause", "Accepted risk": x.get("approved_risk_level"), "Approved at": x.get("approved_at"), "Source": x.get("source_filename")}
                        for x in approval_store["approved_baseline"]
                    ], use_container_width=True, hide_index=True)
                else:
                    st.caption("No additional examples have been permanently approved yet.")
            st.markdown("---")
            st.markdown(f"### Recent human-approval activity ({len(approval_store.get('audit_log', []))})")
            audit_rows = [
                {
                    "Action": str(entry.get("action", "")).replace("_", " ").title(),
                    "Provision": entry.get("title", ""),
                    "Date (UTC)": entry.get("at", ""),
                    "Item ID": entry.get("item_id", ""),
                }
                for entry in reversed(approval_store.get("audit_log", [])[-50:])
            ]
            if audit_rows:
                st.dataframe(audit_rows, use_container_width=True, hide_index=True)
            else:
                st.caption("No human-approval actions have been recorded yet.")

            st.caption(f"Saved locally in: data/human_approval_store.json · {review_store_path}")
            st.info("Reject / Remove only dismisses the item from this document's active approval list. Semi-Approve saves it to the separate queue. Approve Permanently updates the supplemental baseline used for future matching. The original core baseline file is not overwritten.")

        # TAB 3: Extracted PDF Text
        with tab_text:
            st.caption(
                f"Extracted using PyMuPDF (fitz) from '{ext_res['filename']}' — "
                f"{ext_res['page_count']} page(s), {ext_res['word_count']:,} words, {ext_res['char_count']:,} characters."
            )
            view_mode = st.radio(
                "Text Display Format",
                options=["Combined Document Text", "Page-by-Page Breakdown"],
                horizontal=True,
            )
            if view_mode == "Combined Document Text":
                st.text_area(
                    "Extracted Contract Text",
                    value=ext_res["full_text"],
                    height=420,
                    label_visibility="collapsed",
                )
            else:
                for page in ext_res["pages"]:
                    title = f"Page {page['page_number']} ({page['word_count']} words)"
                    with st.expander(title, expanded=(page["page_number"] == 1)):
                        if page["is_empty"]:
                            st.warning("This page contained no extractable text.")
                        else:
                            st.text(page["text"])

        # TAB 4: Downloadable Reports
        with tab_report:
            md_report = generate_markdown_report(analysis, ext_res)
            json_report = generate_json_report(analysis, ext_res)
            base_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(ext_res["filename"]))[0]).strip("_") or "contract"

            st.markdown("##### Export Analysis Results")
            st.write(
                "Download the structured contract analysis report in Markdown (`.md`) for readable documentation "
                "or JSON (`.json`) for structured data inspection."
            )

            dl1, dl2, dl3 = st.columns([1, 1, 1.2])
            with dl1:
                st.download_button(
                    label="Download Markdown Report (.md)",
                    data=md_report,
                    file_name=f"{base_stem}_risk_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    label="Download JSON Report (.json)",
                    data=json_report,
                    file_name=f"{base_stem}_risk_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with dl3:
                if st.button("Save Copy to Local reports/ Folder", use_container_width=True):
                    saved_md, saved_json = save_report_files(md_report, json_report, ext_res["filename"], os.path.join(PROJECT_DIR, "reports"))
                    st.success(f"Saved to `{saved_md}` and `{saved_json}`")

            with st.expander("Preview Markdown Report", expanded=True):
                st.markdown(md_report)

    # Persistent Footer Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Academic &amp; Preliminary Screening Disclaimer:</strong>
            This application is developed as a college academic project to assist preliminary NDA review.
            It uses an educational reference baseline and does not provide formal legal advice or replace
            consultation with a qualified legal professional. AI findings may be incomplete or incorrect; verify quoted evidence and use human legal review before making decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
