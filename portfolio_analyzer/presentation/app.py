"""Portfolio Analyzer — Streamlit entry point.

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
    page_title="Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    _sidebar()

    if "metrics" not in st.session_state:
        _show_welcome()
        return

    page = st.session_state.get("page", "Dashboard")
    metrics = st.session_state["metrics"]
    config = st.session_state.get("config")
    holdings = st.session_state.get("holdings", [])

    if page == "Dashboard":
        from portfolio_analyzer.presentation.views.dashboard import render
        render(metrics, config)
    elif page == "Holdings":
        from portfolio_analyzer.presentation.views.holdings import render
        render(metrics)
    elif page == "Allocation":
        from portfolio_analyzer.presentation.views.allocations import render
        render(metrics, config)
    elif page == "Performance":
        from portfolio_analyzer.presentation.views.performance import render
        render(metrics)
    elif page == "Risk":
        from portfolio_analyzer.presentation.views.risk import render
        render(metrics)
    elif page == "Backtest":
        from portfolio_analyzer.presentation.views.backtest import render
        render(metrics)
    elif page == "Rebalancing":
        from portfolio_analyzer.presentation.views.rebalancing import render
        render(metrics, config)
    elif page == "Benchmark":
        from portfolio_analyzer.presentation.views.benchmark import render
        render(metrics)


def _sidebar():
    """Render sidebar: file upload, navigation, settings."""
    with st.sidebar:
        st.markdown("### 📊 Portfolio Analyzer")

        # Navigation (only if data loaded)
        if "metrics" in st.session_state:
            st.markdown("---")
            pages = ["Dashboard", "Holdings", "Allocation", "Performance",
                     "Risk", "Backtest", "Rebalancing", "Benchmark"]
            icons = ["📊", "💼", "🎯", "📈", "⚡", "📉", "⚖️", "🏆"]
            current = st.session_state.get("page", "Dashboard")
            for icon, page in zip(icons, pages):
                if st.button(f"{icon} {page}", key=f"nav_{page}",
                             use_container_width=True,
                             type="primary" if page == current else "secondary"):
                    st.session_state["page"] = page
                    st.rerun()

        st.markdown("---")
        st.markdown("##### 📁 Data Input")

        holdings_file = st.file_uploader(
            "Holdings CSV/XLSX", type=["csv", "xlsx"],
            key="holdings_upload",
        )
        targets_file = st.file_uploader(
            "Targets CSV (optional)", type=["csv"],
            key="targets_upload",
        )

        st.markdown("##### ⚙️ Parameters")
        backtest = st.selectbox("Backtest period", ["1y", "2y", "3y", "5y", "10y", "max"], index=3)

        if st.button("🔄 Analyze Portfolio", use_container_width=True, type="primary"):
            if holdings_file is not None:
                _run_analysis(holdings_file, targets_file, backtest)
            else:
                st.warning("Upload a holdings file first.")

        # Sample data option
        if st.button("📂 Load sample data", use_container_width=True):
            _run_analysis("input/holdings.csv", "input/targets.csv", backtest)

        # Excel export
        if "metrics" in st.session_state:
            st.markdown("---")
            if st.button("📥 Export Excel", use_container_width=True):
                _export_excel()


def _run_analysis(holdings_source, targets_source, backtest_period: str):
    """Run the orchestrator and store results in session_state."""
    # Clear old results and caches
    for key in ["metrics", "config"]:
        st.session_state.pop(key, None)

    # Clear lru_cache on config loaders to pick up fresh data
    from portfolio_analyzer.config import _load_raw, _load_static, _load_indexes_csv
    _load_raw.cache_clear()
    _load_static.cache_clear()
    _load_indexes_csv.cache_clear()

    with st.spinner("Analyzing portfolio... This may take a minute."):
        try:
            from portfolio_analyzer.orchestrator import run

            h_source = holdings_source
            h_filename = ""
            if hasattr(holdings_source, "name"):
                h_filename = holdings_source.name

            metrics, config = run(
                holdings_source=h_source,
                config_source=targets_source,
                holdings_filename=h_filename,
            )

            st.session_state["metrics"] = metrics
            st.session_state["config"] = config
            st.session_state["page"] = "Dashboard"
            st.success(f"Analysis complete. Portfolio value: €{metrics.total_value:,.2f}")
            st.rerun()

        except Exception as e:
            st.error(f"Analysis failed: {e}")


def _export_excel():
    """Generate Excel and offer download."""
    try:
        import io
        from portfolio_analyzer.export.excel import generate_excel
        metrics = st.session_state["metrics"]

        # Generate to BytesIO
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_excel(metrics, [], None, tmpdir)
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
    st.markdown("# 📊 Portfolio Analyzer")
    st.markdown("Upload your holdings CSV in the sidebar to get started, or click **Load sample data** to try with demo data.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📈 **Performance**\n\nCAGR, period returns, YTD, per-holding comparison")
    with col2:
        st.info("⚡ **Risk Analytics**\n\nSharpe, Sortino, VaR, CVaR, Beta/Alpha, drawdown")
    with col3:
        st.info("⚖️ **Rebalancing**\n\nLP-optimized buy/sell actions to reach your targets")


if __name__ == "__main__":
    main()