"""Idempotent console setup and context/run-owned private log capture."""

from contextvars import ContextVar
import logging
import sys

from tarzan.runtime.session import current_or_matching_session, current_session


_bootstrap_records: ContextVar[list | None] = ContextVar("tarzan_bootstrap_logs", default=None)


def _bootstrap():
    records = _bootstrap_records.get()
    if records is None:
        records = []
        _bootstrap_records.set(records)
    return records


def bind_capture_to_session(session):
    session.log_records.extend(_bootstrap())
    _bootstrap_records.set([])


class RecordCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self._tarzan_private_capture = True
        self._fmt = logging.Formatter(datefmt="%H:%M:%S")

    @property
    def records(self):
        session = current_or_matching_session()
        return list(session.log_records if session is not None else _bootstrap())

    def emit(self, record):
        try:
            session = current_session()
            records = session.log_records if session is not None else _bootstrap()
            message = record.getMessage()
            if record.exc_info:
                message += "\n" + self._fmt.formatException(record.exc_info)
            if record.stack_info:
                message += "\n" + record.stack_info
            records.append({
                "level": record.levelname,
                "time": self._fmt.formatTime(record, "%H:%M:%S"),
                "origin": record.name,
                "message": message,
            })
        except Exception:
            pass


LOG_CAPTURE = RecordCaptureHandler()


def setup_logging():
    _bootstrap_records.set([])
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = next(
        (handler for handler in root.handlers if getattr(handler, "_tarzan_console", False)),
        None,
    )
    if console is None:
        console = logging.StreamHandler(sys.stdout)
        console._tarzan_console = True
        root.addHandler(console)
    console.setLevel(logging.INFO)
    console.stream = sys.stdout
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    if LOG_CAPTURE not in root.handlers:
        root.addHandler(LOG_CAPTURE)
    from tarzan import log_redaction
    log_redaction.install()
