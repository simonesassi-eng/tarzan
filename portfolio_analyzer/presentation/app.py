"""Tarzan — Portfolio Analyzer Streamlit entry point.

Run with: streamlit run portfolio_analyzer/presentation/app.py
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is in Python path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="Tarzan · Portfolio Analyzer",
    page_icon="🦍",
    layout="wide",
    initial_sidebar_state="expanded",
)


PAGES = [
    ("📊", "Dashboard"),
    ("💼", "Holdings"),
    ("⚖️", "Optimizer"),
    ("📈", "Performance"),
    ("🌊", "Return Contribution"),
    ("📖", "Documentation"),
]


def main():
    _inject_css()
    _sidebar()

    if "metrics" not in st.session_state:
        _show_welcome()
        return

    page = st.session_state.get("page", "Dashboard")
    metrics = st.session_state["metrics"]
    config = st.session_state.get("config")

    if page == "Dashboard":
        from portfolio_analyzer.presentation.views.dashboard import render
        render(metrics, config)
    elif page == "Holdings":
        from portfolio_analyzer.presentation.views.holdings import render
        render(metrics)
    elif page == "Optimizer":
        from portfolio_analyzer.presentation.views.optimizer import render
        render(metrics, config)
    elif page == "Performance":
        from portfolio_analyzer.presentation.views.performance import render
        render(metrics)
    elif page == "Return Contribution":
        from portfolio_analyzer.presentation.views.contribution import render
        render(metrics)
    elif page == "Documentation":
        from portfolio_analyzer.presentation.views.documentation import render
        render(metrics)


def _inject_css():
    """Global CSS tweaks for a cleaner dark look."""
    st.markdown(
        """
        <style>
        /* Hide Streamlit default chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Custom metric cards */
        .metric-card {
            background: linear-gradient(135deg, #161b22 0%, #1e2530 100%);
            border: 1px solid #21262d;
            border-radius: 12px;
            padding: 18px;
            text-align: center;
        }
        .metric-label {
            font-size: 0.7rem;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            margin: 6px 0 2px;
        }
        .metric-rating {
            font-size: 0.7rem;
        }

        /* Nav buttons */
        .stButton button {
            justify-content: flex-start !important;
            text-align: left !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def _sidebar():
    """Sidebar: upload files, navigation, export."""
    with st.sidebar:
        st.markdown("### 🦍 Tarzan")
        st.caption("Portfolio Analyzer")

        # Navigation (only if data loaded)
        if "metrics" in st.session_state:
            st.markdown("---")
            current = st.session_state.get("page", "Dashboard")
            for icon, name in PAGES:
                if st.button(f"{icon}  {name}", key=f"nav_{name}",
                             use_container_width=True,
                             type="primary" if name == current else "secondary"):
                    st.session_state["page"] = name
                    st.rerun()

        st.markdown("---")
        st.markdown("##### 📁 Data Input")

        holdings_file = st.file_uploader(
            "Holdings (CSV/XLSX)", type=["csv", "xlsx"],
            key="holdings_upload",
        )
        targets_file = st.file_uploader(
            "Targets (CSV, optional)", type=["csv"],
            key="targets_upload",
        )

        if st.button("🔄 Analyze Portfolio", use_container_width=True, type="primary"):
            if holdings_file is not None:
                _run_analysis(holdings_file, targets_file)
            else:
                st.warning("Upload a holdings file first.")

        if st.button("📂 Load sample data", use_container_width=True):
            _run_analysis("input/holdings.csv", "input/targets.csv")

        if "metrics" in st.session_state:
            st.markdown("---")
            if st.button("📥 Export Excel", use_container_width=True):
                _export_excel()


def _run_analysis(holdings_source, targets_source):
    """Run the orchestrator and store results in session_state."""
    for key in ["metrics", "config"]:
        st.session_state.pop(key, None)

    # Clear lru_cache on config loaders
    from portfolio_analyzer.config import _load_raw, _load_static, _load_indexes_csv
    _load_raw.cache_clear()
    _load_static.cache_clear()
    _load_indexes_csv.cache_clear()

    with st.spinner("Analyzing portfolio... (fetching market data may take 1-2 min)"):
        try:
            from portfolio_analyzer.orchestrator import run

            h_filename = ""
            if hasattr(holdings_source, "name"):
                h_filename = holdings_source.name

            metrics, config = run(
                holdings_source=holdings_source,
                config_source=targets_source,
                holdings_filename=h_filename,
            )

            st.session_state["metrics"] = metrics
            st.session_state["config"] = config
            st.session_state["page"] = "Dashboard"
            st.success(f"✅ Analysis complete. Portfolio value: €{metrics.total_value:,.2f}")
            st.rerun()

        except Exception as e:
            st.error(f"Analysis failed: {e}")


def _export_excel():
    """Generate Excel and offer download."""
    try:
        import tempfile
        import os
        from portfolio_analyzer.export.excel import generate_excel
        metrics = st.session_state["metrics"]
        config = st.session_state.get("config")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_excel(metrics, [], config, tmpdir)
            with open(path, "rb") as f:
                data = f.read()
            st.download_button(
                "⬇️ Download Excel",
                data=data,
                file_name=os.path.basename(path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except Exception as e:
        st.error(f"Export failed: {e}")


def _show_welcome():
    """Welcome screen when no data is loaded."""
    st.markdown(
        """
        <div style='text-align:center; padding: 60px 20px;'>
            <h1 style='font-size: 3rem; margin-bottom: 0;'>🦍 Tarzan</h1>
            <p style='color: #8b949e; font-size: 1.1rem;'>Portfolio analysis for investors who swing smart.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📈 **Performance**\n\nCAGR, period returns, Sharpe, Sortino, Alpha, Beta vs benchmark")
    with col2:
        st.info("⚡ **Risk Analytics**\n\nVaR, CVaR, Max Drawdown, Volatility on 5y horizon")
    with col3:
        st.info("⚖️ **Optimizer**\n\nMILP-based rebalancing with lump sum, min transaction, freeze rules")

    st.markdown("---")
    st.markdown("#### Get started")
    st.markdown("Upload your holdings in the sidebar, or click **Load sample data** to try with demo data.")


if __name__ == "__main__":
    main()