"""Tests for the unified human-readable HTML run report."""

from __future__ import annotations

from tarzan import audit, data_quality, report_html


class TestUnifiedReport:
    def teardown_method(self):
        data_quality.reset()
        audit.reset()

    def test_clean_run_renders_ok_and_valid_html(self):
        data_quality.reset()
        audit.reset()
        html = report_html.render(generated_at="2026-07-12 10:00")
        assert html.startswith("<!doctype html>")
        assert "<html" in html and "</html>" in html
        assert "Data quality" in html and "Rebalancing audit" in html
        assert "every input parsed and priced cleanly" in html
        assert "No rebalancing plans" in html
        assert "<script" not in html  # no JS / injection surface

    def test_issues_are_escaped_and_grouped(self):
        data_quality.reset()
        # A message with HTML metacharacters must be escaped, not injected.
        data_quality.warning("order_load", "row 3: <b>ISIN</b> & bad", context="row 3")
        data_quality.error("metrics", "computer failed")
        html = report_html.render(generated_at="t")
        assert "&lt;b&gt;ISIN&lt;/b&gt; &amp; bad" in html   # escaped
        assert "<b>ISIN</b>" not in html                     # not raw
        assert "order_load" in html and "metrics" in html    # both sources
        assert "1 WARNING" in html and "1 ERROR" in html      # chips

    def test_audit_plan_rendered_as_prose(self):
        data_quality.reset()
        audit.reset()
        from tarzan.models.investor_config import InvestorConfig
        audit.record_rebalancing_plan(
            "Buy only", no_sell=True, total_value=10000.0, lump_sum=500.0,
            config=InvestorConfig(), holdings=[],
            suggestions=[{"direction": "buy", "name": "VWCE", "ticker": "VWCE",
                          "amount_eur": 500.0, "reason": "toward target"}],
            verifications=[{"check": "Invested Allocation", "status": "OK",
                            "detail": "Equities 60%"}])
        html = report_html.render(generated_at="t")
        assert "Buy only" in html
        assert "VWCE" in html and "toward target" in html   # action prose
        assert "€500.00" in html                             # formatted amount
        assert "lump sum €500.00" in html                    # config prose
        assert '"amount_eur"' not in html                    # NOT raw JSON

    def test_write_report_creates_html_file(self, tmp_path):
        data_quality.reset()
        audit.reset()
        path = report_html.write_report(str(tmp_path), generated_at="2026-07-12 10:00")
        assert path is not None
        content = (tmp_path / "report.html").read_text()
        assert content.startswith("<!doctype html>")
        assert "Tarzan — Run Report" in content

    def test_write_never_raises_on_bad_dir(self):
        # Best-effort: an unwritable path returns None, does not raise.
        assert report_html.write_report("/nonexistent/\0/bad", generated_at="t") is None
