"""Interrupted demand ownership and unknown history ports stay isolated."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Event

import pytest

from tarzan.data import yahoo_history_groups, yahoo_info_groups, yahoo_info, yahoo_quote_batch
from tarzan.tests.test_native_quote_group_recovery import acquisition_owner


@pytest.mark.parametrize("kind", ["metadata", "daily-history"])
def test_gather_interruption_releases_driver_and_unstarted_own_demands(monkeypatch, kind):
    with acquisition_owner() as owner:
        if kind == "metadata":
            cohorts = yahoo_info_groups._DemandGroups(owner.transport)
            invoke = lambda: cohorts.get("SYNGATHERCANCEL.MI")
        else:
            cohorts = yahoo_history_groups._HistoryGroups(owner.transport)
            invoke = lambda: cohorts.get(
                lambda: pytest.fail("interrupted gather must not load history"),
                lambda: pytest.fail("interrupted gather must not pace history"),
            )

        def cancelled_wait(timeout=None):
            assert cohorts.driving is True
            raise KeyboardInterrupt("synthetic gather interruption")

        monkeypatch.setattr(cohorts.condition, "wait", cancelled_wait)
        with pytest.raises(KeyboardInterrupt, match="synthetic gather interruption"):
            invoke()
        assert cohorts.driving is False and cohorts.pending == []
        cohorts.close()
        state = owner.transport.coordinator.snapshot()
        assert state["requested"] == state["acquired"] == state["released"] == 0


@pytest.mark.parametrize("kind", ["metadata", "daily-history"])
def test_cancelled_queued_follower_cannot_unlock_or_fail_an_active_driver(monkeypatch, kind):
    started, release = Event(), Event()

    def wait_for_release():
        started.set()
        assert release.wait(timeout=5)

    with acquisition_owner() as owner:
        if kind == "metadata":
            cohorts = yahoo_info_groups._DemandGroups(owner.transport)

            def load(keys):
                assert keys == ["SYNLEADER.MI"]
                wait_for_release()
                return {keys[0]: "completed-leader"}

            monkeypatch.setattr(yahoo_quote_batch, "info_many", load)
            leader_call = lambda: cohorts.get("SYNLEADER.MI")
            follower_call = lambda: cohorts.get("SYNFOLLOWER.MI")
        else:
            cohorts = yahoo_history_groups._HistoryGroups(owner.transport)
            leader_call = lambda: cohorts.get(lambda: "completed-leader", wait_for_release)
            follower_call = lambda: cohorts.get(
                lambda: pytest.fail("cancelled queued follower must not load"),
                lambda: pytest.fail("cancelled queued follower must not pace"),
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            leader = pool.submit(copy_context().run, leader_call)
            try:
                assert started.wait(timeout=5)

                def cancelled_wait(timeout=None):
                    raise KeyboardInterrupt("synthetic queued follower cancellation")

                monkeypatch.setattr(cohorts.condition, "wait", cancelled_wait)
                with pytest.raises(KeyboardInterrupt, match="queued follower cancellation"):
                    follower_call()
                assert cohorts.driving is True and cohorts.pending == []
            finally:
                release.set()
            assert leader.result(timeout=5) == "completed-leader"
        assert cohorts.driving is False
        cohorts.close()


@pytest.mark.parametrize("port_index", range(6))
def test_unknown_history_port_retains_scalar_path_even_after_scope_entry(monkeypatch, port_index):
    method_owner, name, original = yahoo_info._NATIVE_HISTORY_PORTS[port_index]

    def unknown_wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    with acquisition_owner(), yahoo_history_groups.history_group_scope():
        monkeypatch.setattr(method_owner, name, unknown_wrapper)
        handled, value = yahoo_history_groups.demanded_history(
            "SYNCONTRACT.MI",
            {"interval": "1d", "auto_adjust": False, "period": "5y"},
            lambda: pytest.fail("unknown history implementation must not be grouped"),
            lambda: pytest.fail("unknown history implementation must not borrow pacing"),
        )
        assert handled is False and value is None
