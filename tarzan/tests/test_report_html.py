"""Tests for the unified human-readable HTML run report.

report.html = a run summary (headline figures) + the data-quality report.
(The rebalancing audit is intentionally NOT part of this report.)
"""

from __future__ import annotations

from tarzan import data_quality, report_html
from tarzan.models.portfolio import PortfolioMetrics


class TestUnifiedReport:
    def teardown_method(self):
        data_quality.reset()

    def test_valid_html_with_all_sections(self):
        data_quality.reset()
        html = report_html.render(generated_at="2026-07-12 10:00", metrics=None,
                                  log_text="hello log")
        assert html.startswith("<!doctype html>")
        assert "<html" in html and "</html>" in html
        assert "Run summary" in html and "Data quality" in html
        assert "Full run log" in html          # the log is now IN the file
        assert "<script" not in html           # no JS / injection surface
        assert "Rebalancing audit" not in html  # audit dropped

    def test_full_log_embedded_and_escaped(self):
        data_quality.reset()
        # A log line with HTML metacharacters must be escaped inside <pre>.
        log = "2026-07-12 [INFO] tarzan: fetched <AAPL> & priced 100%\nline2"
        html = report_html.render(generated_at="t", metrics=None, log_text=log)
        assert "<pre class='log'>" in html
        assert "fetched &lt;AAPL&gt; &amp; priced 100%" in html  # escaped
        assert "<AAPL>" not in html                              # not raw
        assert "2 lines" in html                                 # line count

    def test_missing_log_shows_placeholder(self):
        data_quality.reset()
        html = report_html.render(generated_at="t", metrics=None, log_text=None)
        assert "No log trace captured" in html

    def test_clean_run_reports_no_issues(self):
        data_quality.reset()
        html = report_html.render(generated_at="t", metrics=None)
        assert "every input parsed and priced cleanly" in html

    def test_run_summary_tiles_from_metrics(self):
        data_quality.reset()
        m = PortfolioMetrics()
        m.total_value = 233049.18
        m.invested_value = 223380.66
        m.cash_value = 9668.51
        m.xirr_pct = 18.81
        m.twror_pct = 11.55
        m.returns_coverage_pct = 100.0
        html = report_html.render(generated_at="t", metrics=m)
        assert "€233,049.18" in html      # total value tile
        assert "18.81%" in html           # XIRR
        assert "11.55%" in html           # TWROR
        assert "100.00%" in html          # coverage

    def test_none_order_fields_show_dash_not_crash(self):
        data_quality.reset()
        m = PortfolioMetrics()  # holdings-only defaults: xirr/twror None
        m.total_value = 1000.0
        html = report_html.render(generated_at="t", metrics=m)
        assert "€1,000.00" in html
        assert "—" in html  # XIRR/TWROR/coverage render as em-dash, no crash

    def test_degraded_computers_surfaced(self):
        data_quality.reset()
        m = PortfolioMetrics()
        m.total_value = 1000.0
        m.degraded_computers = ["_risk", "_returns"]
        html = report_html.render(generated_at="t", metrics=m)
        assert "fell back to defaults" in html
        assert "_risk" in html and "_returns" in html

    def test_issues_are_escaped_and_grouped(self):
        data_quality.reset()
        data_quality.warning("order_load", "row 3: <b>ISIN</b> & bad", context="row 3")
        data_quality.error("metrics", "computer failed")
        html = report_html.render(generated_at="t", metrics=None)
        assert "&lt;b&gt;ISIN&lt;/b&gt; &amp; bad" in html   # escaped
        assert "<b>ISIN</b>" not in html                     # not raw
        assert "order_load" in html and "metrics" in html
        assert "1 WARNING" in html and "1 ERROR" in html

    def test_write_report_creates_html_file(self, tmp_path):
        data_quality.reset()
        path = report_html.write_report(str(tmp_path), generated_at="2026-07-12 10:00")
        assert path is not None
        content = (tmp_path / "report.html").read_text()
        assert content.startswith("<!doctype html>")
        assert "Tarzan — Run Report" in content

    def test_write_never_raises_on_bad_dir(self):
        assert report_html.write_report("/nonexistent/\0/bad", generated_at="t") is None
