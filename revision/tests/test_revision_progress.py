"""Fake-child streaming/deadline checks; never launch a process or initialize CUDA."""
from __future__ import annotations

import io
import queue
import signal
import subprocess
import threading
from types import SimpleNamespace

import pytest

import revision_runner as controller

SIGKILL = getattr(signal, "SIGKILL", 9)


class Pipe:
    def __init__(self, lines=()):
        self.lines = queue.Queue()
        self.closed = False
        for line in lines:
            self.lines.put(line)

    def finish(self):
        self.lines.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.lines.get(timeout=2)
        if item is None:
            raise StopIteration
        return item

    def close(self):
        self.closed = True


def job(tmp_path, verbose=False):
    config = tmp_path / "config.json"
    config.write_text("{}")
    return controller.RevisionRun(SimpleNamespace(out=str(tmp_path / "run"),
                                  config=str(config), budget_hours=72., verbose=verbose))


def fake_posix(monkeypatch, kill):
    monkeypatch.setattr(controller, "os", SimpleNamespace(name="posix", environ={}, killpg=kill))
    monkeypatch.setattr(controller, "signal", SimpleNamespace(SIGTERM=signal.SIGTERM, SIGKILL=SIGKILL))


def test_progress_is_visible_before_exit_and_details_stay_in_full_log(tmp_path, monkeypatch):
    task = job(tmp_path)
    seen = threading.Event()
    pipe = Pipe(["[RUN] training\n"])

    class Terminal(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if text == "[RUN] training\n":
                seen.set()
            return result

    terminal = Terminal()
    monkeypatch.setattr(controller.sys, "stdout", terminal)
    lines = ["{\"epoch\": 1, \"loss\": 0.1}\n", "unfiltered detail\n",
             "[PROGRESS] train: 1/2 epochs; ETA=1m\n", "[SKIP] completed\n",
             "[PARTIAL] subset\n", "[ERROR] diagnostic summary\n", "[DONE] training\n"]

    class Child:
        pid = 424242
        stdout = pipe

        def wait(self, timeout):
            assert seen.wait(1), "progress was buffered until process exit"
            for line in lines:
                pipe.lines.put(line)
            pipe.finish()
            return 0

    def spawn(*args, **kwargs):
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.STDOUT
        assert kwargs["start_new_session"] and kwargs["encoding"] == "utf-8"
        assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
        return Child()

    fake_posix(monkeypatch, lambda *args: pytest.fail("completed child must not be killed"))
    monkeypatch.setattr(controller.subprocess, "Popen", spawn)
    budget = task.started_path.read_bytes()
    task.command("train", ["fake.py"])
    output = terminal.getvalue().splitlines()
    assert output[0].startswith("[REVISION] train")
    assert output[1:] == ["[RUN] training", *[line.rstrip() for line in lines[2:]]]
    assert (task.out / "controller_logs/train.log").read_text() == "[RUN] training\n" + "".join(lines)
    assert task.started_path.read_bytes() == budget and pipe.closed


def test_verbose_streams_details_and_log_remains_append_only(tmp_path, monkeypatch, capsys):
    task = job(tmp_path, verbose=True)
    payload = "detail\n{\"epoch\": 1}\n[PROGRESS] 1/1\n"
    child = SimpleNamespace(pid=424242, stdout=io.StringIO(payload), wait=lambda timeout: 0)
    folder = task.out / "controller_logs"
    folder.mkdir()
    log = folder / "train.log"
    log.write_text("prior complete log\n")
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: child)
    task.command("train", ["fake.py"])
    assert capsys.readouterr().out.endswith(payload)
    assert log.read_text() == "prior complete log\n" + payload


@pytest.mark.parametrize("leader_ignores_term", [False, True])
def test_timeout_stops_only_owned_tree_even_if_leader_exits_first(tmp_path, monkeypatch,
                                                               leader_ignores_term):
    task = job(tmp_path)
    pipe = Pipe(["[RUN] pending\n", "full detail\n"])
    calls, waits = [], []
    own_group, unrelated_group = {424242, 424243}, {999999}

    class Child:
        pid = 424242
        stdout = pipe

        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1 or (leader_ignores_term and len(waits) == 2):
                raise subprocess.TimeoutExpired("owned child", timeout)
            return 0

    def kill(group, kind):
        assert group == Child.pid
        calls.append((group, kind))
        if kind == signal.SIGTERM:
            own_group.discard(424242)
        else:
            own_group.clear()
            pipe.finish()

    fake_posix(monkeypatch, kill)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: Child())
    budget = task.started_path.read_bytes()
    with pytest.raises(TimeoutError, match="Experiment budget reached"):
        task.command("timeout", ["fake.py"])
    assert calls == [(424242, signal.SIGTERM), (424242, SIGKILL)]
    assert own_group == set() and unrelated_group == {999999}
    assert pipe.closed and task.started_path.read_bytes() == budget
    assert (task.out / "controller_logs/timeout.log").read_text() == "[RUN] pending\nfull detail\n"


def test_nonzero_exit_is_not_hidden_by_successful_output_relay(tmp_path, monkeypatch, capsys):
    task = job(tmp_path)
    payload = "trace detail\n[ERROR] failed\n"
    child = SimpleNamespace(pid=424242, stdout=io.StringIO(payload), wait=lambda timeout: 7)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(RuntimeError, match=r"failed \(exit 7\)"):
        task.command("failed", ["fake.py"])
    output = capsys.readouterr().out
    assert "[ERROR] failed" in output and "trace detail" not in output
    assert (task.out / "controller_logs/failed.log").read_text() == payload


def test_expired_original_budget_refuses_to_spawn(tmp_path, monkeypatch):
    task = job(tmp_path)
    budget = task.started_path.read_bytes()
    monkeypatch.setattr(task, "remaining", lambda: -1)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("expired task spawned"))
    with pytest.raises(TimeoutError, match="budget exhausted"):
        task.command("expired", ["fake.py"])
    assert task.started_path.read_bytes() == budget


def test_interruption_reaps_owned_child_and_reraises(tmp_path, monkeypatch):
    task = job(tmp_path)
    pipe = Pipe()
    calls = []

    class Child:
        pid = 424242
        stdout = pipe
        waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return 0

    def kill(group, kind):
        calls.append((group, kind))
        if kind == SIGKILL:
            pipe.finish()

    fake_posix(monkeypatch, kill)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: Child())
    with pytest.raises(KeyboardInterrupt):
        task.command("interrupted", ["fake.py"])
    assert calls == [(424242, signal.SIGTERM), (424242, SIGKILL)] and pipe.closed


def test_relay_failure_is_reported_instead_of_claiming_stage_success(tmp_path, monkeypatch):
    task = job(tmp_path)

    class BrokenPipe:
        def __iter__(self):
            raise OSError("simulated output read failure")

        def close(self):
            pass

    child = SimpleNamespace(pid=424242, stdout=BrokenPipe(), wait=lambda timeout: 0)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(RuntimeError, match="output relay failed") as error:
        task.command("bad_relay", ["fake.py"])
    assert isinstance(error.value.__cause__, OSError)
