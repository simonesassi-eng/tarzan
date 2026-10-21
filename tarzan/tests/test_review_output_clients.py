"""Clients created by completed-run output share its complete owned lifetime."""

from datetime import date

from curl_cffi import requests
import pytest

from tarzan import main, runtime
from tarzan.runtime.artifacts import LocalOnlyWorkbook
from tarzan.runtime.session import activate_session, completed_run_scope, current_session, last_run_result, record_last_run_result
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings
from tarzan.tests.test_review_output_lifecycle import fake_cli, result


@pytest.fixture
def clients(monkeypatch):
    created = []

    class Session:
        def __init__(self, **kwargs):
            self.close_calls = 0
            created.append(self)

        def close(self):
            self.close_calls += 1

        def request(self, *args, **kwargs):
            pytest.fail("synthetic output opened a network request")

    monkeypatch.setattr(requests, "Session", Session)
    return created


def attach_budget(completed):
    completed.session.transport = ProviderBudget(
        ProviderTransportSettings.configured(), completed.ledger,
    )
    return completed.session.transport


@pytest.mark.parametrize("primary", [None, ValueError("synthetic output error"), KeyboardInterrupt()])
def test_output_created_client_closes_only_after_the_outer_owned_scope(clients, primary):
    completed = result()
    budget = attach_budget(completed)

    def output():
        with completed_run_scope(completed):
            client = budget.yahoo_client()
            with completed_run_scope(completed):
                assert budget.yahoo_client() is client
            assert client.close_calls == 0
            if primary is not None:
                raise primary

    if primary is None:
        output()
    else:
        with pytest.raises(type(primary)) as caught:
            output()
        assert caught.value is primary
    assert len(clients) == 1 and clients[0].close_calls == 1
    assert budget._yahoo_client is None
    assert current_session() is None


@pytest.mark.parametrize("primary", [None, RuntimeError("synthetic primary output failure")])
def test_cleanup_failure_retains_primary_outcome_and_restores_the_caller(clients, primary, caplog):
    first, caller = result("first"), result("caller", date(2025, 2, 3))
    budget = attach_budget(first)
    record_last_run_result(caller)
    closed = []

    def close():
        assert current_session() is first.session
        assert last_run_result() is first
        assert runtime.today() == date(2025, 1, 2)
        closed.append(True)
        raise OSError("synthetic private close failure")

    with runtime.bind_context(caller.session.context):
        def output():
            with completed_run_scope(first):
                client = budget.yahoo_client()
                client.close = close
                if primary is not None:
                    raise primary

        if primary is None:
            output()
        else:
            with pytest.raises(RuntimeError) as caught:
                output()
            assert caught.value is primary
        assert last_run_result() is caller
        assert runtime.today() == date(2025, 2, 3)
    assert closed == [True]
    assert budget._yahoo_client is None and current_session() is None
    assert any(getattr(record, "public_code", None) == "PROVIDER_CLIENT_CLEANUP_FAILED" for record in caplog.records)


def test_reentering_an_old_result_owns_and_closes_its_new_client(clients):
    completed = result()
    budget = attach_budget(completed)
    for _ in range(2):
        with completed_run_scope(completed):
            budget.yahoo_client()
    assert len(clients) == 2
    assert [client.close_calls for client in clients] == [1, 1]


def test_scope_does_not_allocate_transport_just_to_close_it(clients):
    completed = result()
    assert completed.session.transport is None
    with completed_run_scope(completed):
        pass
    assert completed.session.transport is None and clients == []


def test_an_active_analysis_nested_render_does_not_prematurely_close_its_transport(clients):
    completed = result()
    budget = attach_budget(completed)
    with activate_session(completed.session):
        client = budget.yahoo_client()
        with completed_run_scope(completed):
            pass
        assert client.close_calls == 0
    # activate_session is the low-level binding, not the orchestrator owner.
    budget.close()
    assert client.close_calls == 1


@pytest.mark.parametrize("optional_failure", [False, True])
def test_real_optional_cli_path_closes_clients_without_invalidating_core(
    clients, monkeypatch, tmp_path, optional_failure,
):
    completed = fake_cli(monkeypatch)
    budget = attach_budget(completed)
    monkeypatch.setattr("tarzan.backtest.newsletter_portfolios", lambda **kwargs: [object()])

    def workbook(*args):
        budget.yahoo_client()
        if optional_failure:
            raise ValueError("synthetic optional workbook failure")
        return LocalOnlyWorkbook(b"synthetic-only-workbook")

    monkeypatch.setattr(main, "_export_whatif", workbook)
    assert main.main(["--output", str(tmp_path), "--what-if"]) == 0
    assert len(list(tmp_path.glob("*/synthetic/manifest.json"))) == 1
    assert len(list(tmp_path.glob("*/what-if/synthetic/manifest.json"))) == 1
    assert len(clients) == 1 and clients[0].close_calls == 1
    assert budget._yahoo_client is None
