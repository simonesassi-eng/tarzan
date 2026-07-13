"""Tests for the single HTML run report (report.html).

report.html = a top summary of real issues (Tarzan data-quality events + how
handled, plus an explained note for yfinance probe chatter) followed by a
lean log table (Tarzan records + any warning/error; third-party debug hidden).
"""

from __future__ import annotations

from tarzan.runtime import data_quality, report_html


def _rec(level, msg, time="10:00:00", origin="tarzan"):
    return {"level": level, "time": time, "origin": origin, "message": msg}


class TestRunReport:
    def teardown_method(self):
        data_quality.reset()

    def test_valid_html_with_sections(self):
        data_quality.reset()
        html = report_html.render(generated_at="2026-07-12 10:00",
                                  log_records=[_rec("INFO", "hello")])
        assert html.startswith("<!doctype html>")
        assert "Tarzan — Run Report" in html
        assert "Issues &amp; how they were handled" in html
        assert "Run log" in html
        assert "<script" not in html

    def test_clean_run_says_no_issues(self):
        data_quality.reset()
        html = report_html.render(generated_at="t", log_records=[])
        assert "No data-quality issues" in html

    def test_data_quality_issues_with_handling_shown(self):
        data_quality.reset()
        data_quality.warning(
            "enricher",
            "FX for ZAR unavailable — valued from the CSV/order EUR anchor",
            context="XS2105803527")
        html = report_html.render(generated_at="t", log_records=[])
        assert "FX for ZAR unavailable" in html          # what + how handled
        assert "valued from the CSV/order EUR anchor" in html
        assert "XS2105803527" in html                    # context column
        assert "enricher" in html

    def test_third_party_probe_note_explains_yfinance(self):
        data_quality.reset()
        recs = [
            _rec("ERROR", "$NTSG.MI: possibly delisted", origin="yfinance"),
            _rec("ERROR", "$NTSG.DE: possibly delisted", origin="yfinance"),
        ]
        html = report_html.render(generated_at="t", log_records=recs)
        assert "yfinance emitted 2 expected ticker-probe" in html
        assert "not</b> Tarzan errors" in html or "not" in html

    def test_log_filters_third_party_debug(self):
        data_quality.reset()
        recs = [
            _rec("DEBUG", "tarzan internal detail", origin="tarzan.engine"),
            _rec("DEBUG", "yfinance chatter", origin="yfinance"),
            _rec("DEBUG", "peewee sql", origin="peewee"),
            _rec("WARNING", "a real warning", origin="yfinance"),
            _rec("INFO", "tarzan info", origin="tarzan"),
        ]
        html = report_html.render(generated_at="t", log_records=recs)
        # Kept: tarzan.* (any level) + WARNING/ERROR from anywhere.
        assert "tarzan internal detail" in html          # tarzan DEBUG kept
        assert "a real warning" in html                  # yfinance WARNING kept
        assert "tarzan info" in html
        # Dropped: third-party DEBUG/INFO.
        assert "yfinance chatter" not in html
        assert "peewee sql" not in html
        assert "3 of 5 log entries shown" in html

    def test_message_escaped(self):
        data_quality.reset()
        recs = [_rec("INFO", "fetched <AAPL> & priced 100%", origin="tarzan.data")]
        html = report_html.render(generated_at="t", log_records=recs)
        assert "fetched &lt;AAPL&gt; &amp; priced 100%" in html
        assert "<AAPL>" not in html

    def test_write_report_creates_file(self, tmp_path):
        data_quality.reset()
        path = report_html.write_report(str(tmp_path), generated_at="t",
                                        log_records=[_rec("INFO", "x")])
        assert path is not None
        assert (tmp_path / "report.html").read_text().startswith("<!doctype html>")

    def test_write_never_raises_on_bad_dir(self):
        assert report_html.write_report("/nonexistent/\0/bad", generated_at="t") is None
