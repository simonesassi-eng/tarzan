"""Tests for the single color-coded HTML run log (report.html).

report.html = one color-coded table of the whole run's log (level · time ·
origin · message), one row per entry, colored by log level — no separate
analyzer.log, no summary/audit sections.
"""

from __future__ import annotations

from tarzan import report_html


def _rec(level, msg, time="10:00:00", origin="tarzan"):
    return {"level": level, "time": time, "origin": origin, "message": msg}


class TestRunLogReport:
    def test_valid_html_table_structure(self):
        html = report_html.render(generated_at="2026-07-12 10:00",
                                  log_records=[_rec("INFO", "hello")])
        assert html.startswith("<!doctype html>")
        assert "<html" in html and "</html>" in html
        assert "Tarzan — Run Log" in html
        assert "<table class='log'>" in html
        # Column headers present.
        for col in ("Level", "Time", "Origin", "Message"):
            assert f">{col}</th>" in html
        assert "<script" not in html  # no JS / injection surface

    def test_rows_colored_by_level(self):
        recs = [_rec("INFO", "ok"), _rec("WARNING", "careful"),
                _rec("ERROR", "boom"), _rec("DEBUG", "detail")]
        html = report_html.render(generated_at="t", log_records=recs)
        assert "color:#D28004" in html   # WARNING amber
        assert "color:#DC2626" in html   # ERROR red
        assert "color:#579FA8" in html   # DEBUG teal
        assert "color:#1E293B" in html   # INFO ink

    def test_message_and_fields_escaped(self):
        recs = [_rec("INFO", "fetched <AAPL> & priced 100%", origin="tarzan.data")]
        html = report_html.render(generated_at="t", log_records=recs)
        assert "fetched &lt;AAPL&gt; &amp; priced 100%" in html  # escaped
        assert "<AAPL>" not in html                              # not raw
        assert "tarzan.data" in html                             # origin column

    def test_summary_chips_count_by_level(self):
        recs = [_rec("INFO", "a"), _rec("INFO", "b"), _rec("WARNING", "c")]
        html = report_html.render(generated_at="t", log_records=recs)
        assert "2 INFO" in html and "1 WARNING" in html
        assert "3 entries" in html

    def test_empty_log_renders_cleanly(self):
        html = report_html.render(generated_at="t", log_records=[])
        assert "0 entries" in html
        assert "no log entries" in html
        assert "<table class='log'>" in html  # table still present, no rows

    def test_write_report_creates_file(self, tmp_path):
        path = report_html.write_report(str(tmp_path), generated_at="2026-07-12 10:00",
                                        log_records=[_rec("INFO", "x")])
        assert path is not None
        content = (tmp_path / "report.html").read_text()
        assert content.startswith("<!doctype html>")
        assert "Tarzan — Run Log" in content

    def test_write_never_raises_on_bad_dir(self):
        assert report_html.write_report("/nonexistent/\0/bad", generated_at="t") is None
