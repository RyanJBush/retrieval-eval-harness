"""CLI tests: exit codes, validation before any model work, failed-run handling, preserved outputs.
Retrievers/clock are FAKES injected through main()'s test hooks (tests/fakes.py); no model, no network."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from fakes import FakeClock, make_builders

from retrieval_eval import cli, export
from retrieval_eval.data import load_dataset

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "synthetic"
IDS = load_dataset(FIXTURE).doc_ids


@pytest.fixture(autouse=True)
def _restore_thread_env(monkeypatch):
    for name in cli.THREAD_ENV:  # main() sets these; make pytest restore the originals afterwards
        monkeypatch.setenv(name, os.environ.get(name, "1"))


def fake(per_method=None):
    clock = FakeClock()
    return {"builders": make_builders(clock, IDS, per_method=per_method), "clock": clock}


def run_cli(tmp_path, *extra, per_method=None, out="out", dataset=FIXTURE):
    kw = fake(per_method)
    rc = cli.main(["--dataset", str(dataset), "--out-dir", str(tmp_path / out), *extra], **kw)
    return rc, tmp_path / out


def snapshot(d: Path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


# ------------------------------------------------------------------ success
def test_default_run_exits_zero_and_writes_the_documented_artifacts(tmp_path, capsys):
    rc, out = run_cli(tmp_path)
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == sorted([*export.CSV_FILES, "run.json"])
    run = export.load_run(out / "run.json")
    assert run["run"]["status"] == "completed" and not (out / "run_failed.json").exists()
    counts = {n: len(export.read_csv(out / n)[1]) for n in export.CSV_FILES}
    assert counts == {"summary.csv": 6, "per_query.csv": 48, "rankings.csv": 80, "latency.csv": 320,
                      "latency_summary.csv": 18}
    for m in ("bm25", "embedding"):
        assert run["results"][m]["counts"] == {"total": 8, "scored": 7, "no_judged_relevant": 1}
    assert export.verify_saved_run(run, dataset_dir=FIXTURE) == []
    text = capsys.readouterr().out
    assert "status: completed" in text and "synthetic demo observations" in text and "microbenchmark" in text


def test_manifest_records_dataset_identity_command_and_fixture_verification(tmp_path):
    rc, out = run_cli(tmp_path, "--k", "1,3", "--repetitions", "2", "--warmup", "1", "--threads", "2")
    assert rc == 0
    run = export.load_run(out / "run.json")
    names = [f["name"] for f in run["dataset"]["files"]]
    assert names == ["documents.jsonl", "queries.jsonl", "qrels.csv"]
    for f in run["dataset"]["files"]:
        assert f["sha256"] == hashlib.sha256((FIXTURE / f["name"]).read_bytes()).hexdigest()
    assert run["dataset"]["fixture_manifest"] == {"name": "FIXTURE.sha256", "present": True, "verified": True}
    assert run["dataset"]["designation"] == "synthetic"
    assert run["config"]["ks"] == [1, 3] and run["config"]["max_k"] == 3 and run["config"]["threads_requested"] == 2
    assert run["run"]["command"][:3] == ["python", "-m", "retrieval_eval.cli"] and "--threads" in run["run"]["command"]
    assert len(run["latency"]["samples"]) == 8 * 2 * 2


def test_thread_environment_is_fixed_before_run(tmp_path):
    rc, _ = run_cli(tmp_path, "--threads", "3", "--repetitions", "1", "--warmup", "0")
    assert rc == 0 and all(os.environ[n] == "3" for n in cli.THREAD_ENV)
    run = export.load_run(tmp_path / "out" / "run.json")
    assert run["environment"]["thread_env"] == {n: "3" for n in cli.THREAD_ENV}


def test_external_designation_is_recorded(tmp_path):
    rc, out = run_cli(tmp_path, "--dataset-designation", "external", "--repetitions", "1", "--warmup", "0")
    run = export.load_run(out / "run.json")
    assert rc == 0 and run["dataset"]["designation"] == "external" and run["dataset"]["synthetic_developer_fixture"] is False


# ------------------------------------------------------------------ invalid configuration: exit 2, nothing built, nothing written
@pytest.mark.parametrize("args", [
    ["--k", "0"], ["--k", "1,,3"], ["--k", "a"], ["--k", "2,2"], ["--k", ""],
    ["--repetitions", "0"], ["--repetitions", "-4"], ["--warmup", "-1"], ["--threads", "0"],
])
def test_invalid_numeric_configuration_exits_2_before_any_work(tmp_path, args, capsys):
    def must_not_build(clk):
        raise AssertionError("a retriever was built for an invalid configuration")

    rc = cli.main(["--dataset", str(FIXTURE), "--out-dir", str(tmp_path / "out"), *args],
                  builders=[("bm25", must_not_build), ("embedding", must_not_build)])
    assert rc == 2 and not (tmp_path / "out").exists()
    assert capsys.readouterr().err.startswith("error:")


@pytest.mark.parametrize("args", [["--repetitions", "x"], ["--warmup", "1.5"], ["--threads", "two"], []])
def test_argparse_level_errors_also_exit_2(tmp_path, args):
    full = args if args == [] else ["--dataset", str(FIXTURE), "--out-dir", str(tmp_path / "o"), *args]
    assert cli.main(full) == 2


def test_help_exits_zero(capsys):
    assert cli.main(["--help"]) == 0 and "--out-dir" in capsys.readouterr().out


def test_unknown_or_incomplete_dataset_exits_2(tmp_path, capsys):
    assert cli.main(["--dataset", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "o")]) == 2
    assert "dataset directory not found" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["--dataset", str(empty), "--out-dir", str(tmp_path / "o")]) == 2
    assert "missing: documents.jsonl" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_malformed_dataset_exits_2_with_file_and_line(tmp_path, capsys):
    bad = tmp_path / "bad"
    shutil.copytree(FIXTURE, bad)
    (bad / "FIXTURE.sha256").unlink()
    (bad / "documents.jsonl").write_text('{"doc_id": "a", "text": "x"}\n{"doc_id": "a", "text": "y"}\n', encoding="utf-8")
    rc = cli.main(["--dataset", str(bad), "--out-dir", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == 2 and "documents.jsonl:2" in err and "duplicate doc_id" in err and not (tmp_path / "o").exists()


def test_modified_frozen_fixture_is_rejected_against_its_manifest(tmp_path, capsys):
    copy = tmp_path / "ds"
    shutil.copytree(FIXTURE, copy)
    with open(copy / "queries.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n")  # still valid JSONL, but the bytes no longer match FIXTURE.sha256
    rc = cli.main(["--dataset", str(copy), "--out-dir", str(tmp_path / "o")], **fake())
    assert rc == 2 and "do not match FIXTURE.sha256" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_dataset_without_manifest_is_accepted_and_recorded_as_unverified(tmp_path):
    ds = tmp_path / "ds"
    shutil.copytree(FIXTURE, ds)
    (ds / "FIXTURE.sha256").unlink()
    rc, out = run_cli(tmp_path, "--repetitions", "1", "--warmup", "0", dataset=ds)
    assert rc == 0 and export.load_run(out / "run.json")["dataset"]["fixture_manifest"] == {
        "name": "FIXTURE.sha256", "present": False, "verified": None}


# ------------------------------------------------------------------ output directory is never overwritten
def test_nonempty_output_directory_is_rejected_and_left_untouched(tmp_path, capsys):
    rc, out = run_cli(tmp_path, "--repetitions", "1", "--warmup", "0")
    assert rc == 0
    before = snapshot(out)
    rc2, _ = run_cli(tmp_path, "--repetitions", "1", "--warmup", "0")  # same directory again
    assert rc2 == 2 and "not empty" in capsys.readouterr().err
    assert snapshot(out) == before  # the completed run is preserved byte for byte


def test_output_path_that_is_a_file_is_rejected(tmp_path):
    f = tmp_path / "out"
    f.write_text("x", encoding="utf-8")
    rc, _ = run_cli(tmp_path)
    assert rc == 2 and f.read_text(encoding="utf-8") == "x"


def test_existing_empty_output_directory_is_accepted(tmp_path):
    (tmp_path / "out").mkdir()
    rc, out = run_cli(tmp_path, "--repetitions", "1", "--warmup", "0")
    assert rc == 0 and (out / "run.json").exists()


# ------------------------------------------------------------------ failures: exit 1, incomplete, no results published
def failure_manifest(out: Path) -> dict:
    assert (out / "run_failed.json").exists()
    assert not (out / "run.json").exists(), "a failed run must never leave a run.json"
    assert not [p for p in out.iterdir() if p.suffix == ".csv"], "no CSV may be published for a failed run"
    doc = json.loads((out / "run_failed.json").read_text(encoding="utf-8"))
    assert doc["status"] == "incomplete" and doc["status"] != "completed"
    keys = set()

    def collect(node):
        if isinstance(node, dict):
            keys.update(node)
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)

    collect(doc)
    assert not keys & {"aggregate", "macro_recall", "mrr", "results", "metrics", "samples"}  # no published results
    return doc


def test_adapter_failure_during_measurement_exits_1_with_stage_method_query(tmp_path, capsys):
    q06 = load_dataset(FIXTURE).queries[5].text
    rc, out = run_cli(tmp_path, per_method={"embedding": {"fail": {(q06, 9): RuntimeError("embedding exploded")}}})
    assert rc == 1
    doc = failure_manifest(out)
    err = doc["errors"][0]
    assert (err["stage"], err["method"], err["query_id"], err["error_type"]) == ("measure", "embedding", "q06", "RuntimeError")
    assert "embedding exploded" in err["message"] and "Traceback" in err["traceback"]
    assert doc["partial"]["measured_samples_collected"] > 0 and "timing" not in doc["partial"]["completed_stages"]
    assert doc["config"]["ks"] == [1, 3, 5] and doc["dataset"]["files"]
    stderr = capsys.readouterr().err
    assert "run incomplete at measure (method=embedding) (query=q06)" in stderr


def test_build_failure_exits_1_and_never_substitutes_an_adapter(tmp_path):
    clock = FakeClock()
    builders = make_builders(clock, IDS)

    def failing_model(clk):
        raise ConnectionError("could not download model")

    builders[1] = ("embedding", failing_model)
    rc = cli.main(["--dataset", str(FIXTURE), "--out-dir", str(tmp_path / "o")], builders=builders, clock=clock)
    assert rc == 1
    doc = failure_manifest(tmp_path / "o")
    assert (doc["errors"][0]["stage"], doc["errors"][0]["method"]) == ("build", "embedding")
    assert doc["errors"][0]["error_type"] == "ConnectionError"


def test_first_call_and_retrieve_failures_exit_1(tmp_path):
    q01 = load_dataset(FIXTURE).queries[0].text
    rc, out = run_cli(tmp_path, per_method={"bm25": {"fail": {(q01, 1): ValueError("bad first call")}}}, out="a")
    assert rc == 1 and failure_manifest(out)["errors"][0]["stage"] == "first_call"
    rc, out = run_cli(tmp_path, per_method={"bm25": {"fail": {(q01, 2): ValueError("bad retrieve")}}}, out="b")
    assert rc == 1 and failure_manifest(out)["errors"][0]["stage"] == "retrieve"


def test_nonfinite_score_from_an_adapter_is_a_failed_run_not_a_result(tmp_path):
    q02 = load_dataset(FIXTURE).queries[1].text
    rc, out = run_cli(tmp_path, per_method={"bm25": {"rankings": {q02: [("d10", float("nan"))]}}})
    doc = failure_manifest(out) if rc == 1 else None
    assert rc == 1 and doc["errors"][0]["query_id"] == "q02"


def test_export_failure_marks_the_run_incomplete_and_exits_1(tmp_path, monkeypatch, capsys):
    def explode(out_dir, run):
        raise export.ExportError("disk full")

    monkeypatch.setattr(export, "write_run_artifacts", explode)
    rc, out = run_cli(tmp_path, "--repetitions", "1", "--warmup", "0")
    assert rc == 1
    doc = failure_manifest(out)
    assert doc["errors"][0]["stage"] == "export" and "disk full" in doc["errors"][0]["message"]
    assert doc["run_id"]  # the run id is retained for traceability
    assert "run incomplete at export" in capsys.readouterr().err


def test_real_export_failure_leaves_only_the_failure_manifest(tmp_path, monkeypatch):
    real = os.replace

    def flaky(src, dst):
        if Path(dst).name == "run.json":
            raise OSError("rename failed")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    rc, out = run_cli(tmp_path, "--repetitions", "1", "--warmup", "0")
    assert rc == 1 and sorted(p.name for p in out.iterdir()) == ["run_failed.json"]


def test_failed_run_directory_is_not_reused_silently(tmp_path):
    q01 = load_dataset(FIXTURE).queries[0].text
    rc, out = run_cli(tmp_path, per_method={"bm25": {"fail": {(q01, 1): ValueError("x")}}})
    assert rc == 1
    rc2, _ = run_cli(tmp_path)  # same dir: not empty (holds run_failed.json) -> rejected, evidence kept
    assert rc2 == 2 and (out / "run_failed.json").exists()


# ------------------------------------------------------------------ ranking-consistency acceptance rule
def test_changed_ordered_ids_during_timing_exit_1_and_retain_the_diagnostic(tmp_path, capsys):
    rc, out = run_cli(tmp_path, "--repetitions", "4", "--warmup", "1",
                      per_method={"embedding": {"drift_after": 5}})
    assert rc == 1
    doc = failure_manifest(out)  # also proves: no run.json, no CSV, no result keys
    err = doc["errors"][0]
    assert err["stage"] == "consistency" and err["method"] == "embedding"
    diag = doc["partial"]["ranking_disagreements"]
    assert diag["total"] >= 1 and diag["details"]
    d0 = diag["details"][0]
    assert {"phase", "method", "query_id", "call_index", "expected_ids", "observed_ids"} <= set(d0)
    assert d0["expected_ids"] != d0["observed_ids"]
    assert "run incomplete at consistency" in capsys.readouterr().err


def test_same_ids_with_small_score_differences_complete_and_are_reported_separately(tmp_path):
    rc, out = run_cli(tmp_path, "--repetitions", "4", "--warmup", "1", per_method={"bm25": {"jitter_calls": {5}}})
    assert rc == 0
    run = export.load_run(out / "run.json")
    rc_block = run["ranking_consistency"]
    assert rc_block["ranking_disagreements"] == 0 and rc_block["score_only_differences"] == 8
    assert 0.0 < rc_block["max_abs_score_difference"] <= 1e-8
    assert export.verify_saved_run(run, dataset_dir=FIXTURE) == []


# ------------------------------------------------------------------ exclusive ownership of the output directory
import threading  # noqa: E402


def _args(out: Path, *extra):
    return ["--dataset", str(FIXTURE), "--out-dir", str(out), "--repetitions", "1", "--warmup", "0", *extra]


def _run_id(out: Path):
    p = out / "run.json"
    return json.loads(p.read_text(encoding="utf-8"))["run"]["run_id"] if p.exists() else None


def test_interleaved_competitor_cannot_replace_the_owners_run(tmp_path, capsys):
    """The reproduced race: A is inside its first builder (output dir looked empty when it started); B then runs to
    completion in the same directory; A resumes. B must be rejected and A's run must be the one that remains."""
    out = tmp_path / "shared"
    out.mkdir()  # an existing EMPTY directory is supported
    seen = {}
    clock_a = FakeClock()
    builders_a = make_builders(clock_a, IDS)
    real_first = builders_a[0][1]

    def pausing_first_builder(clk):
        seen["listing_while_A_builds"] = sorted(p.name for p in out.iterdir())
        seen["B_rc"] = cli.main(_args(out), **fake())
        seen["B_stderr"] = capsys.readouterr().err
        seen["files_after_B"] = snapshot(out)
        return real_first(clk)

    builders_a[0] = ("bm25", pausing_first_builder)
    rc_a = cli.main(_args(out), builders=builders_a, clock=clock_a)
    assert rc_a == 0
    assert seen["listing_while_A_builds"] == [cli.LOCK_NAME]  # A holds the reservation before its first build
    assert seen["B_rc"] != 0 and "reserved by another invocation" in seen["B_stderr"]
    assert seen["files_after_B"] == {cli.LOCK_NAME: seen["files_after_B"][cli.LOCK_NAME]}  # B wrote nothing at all
    assert sorted(p.name for p in out.iterdir()) == sorted([*export.CSV_FILES, "run.json"])  # lock released, run complete
    assert export.verify_saved_run(export.load_run(out / "run.json"), dataset_dir=FIXTURE) == []


def test_owner_finishing_between_the_early_check_and_the_reservation_is_not_overwritten(tmp_path, capsys):
    """B passes its early emptiness check, then A runs to completion (and releases) before B tries to reserve."""
    out = tmp_path / "shared"
    state = {}

    def owner_finishes_first():
        state["A_rc"] = cli.main(_args(out), **fake())
        state["A_run_id"] = _run_id(out)
        state["A_files"] = snapshot(out)

    assert not out.exists()  # B's early check sees a directory that does not exist yet
    rc_b = cli.main(_args(out), _before_reserve=owner_finishes_first, **fake())
    err = capsys.readouterr().err
    assert state["A_rc"] == 0 and state["A_run_id"]
    assert rc_b == 2 and "not empty" in err
    assert snapshot(out) == state["A_files"]  # A's completed run is byte-for-byte intact; B's reservation is gone
    assert _run_id(out) == state["A_run_id"] and not (out / cli.LOCK_NAME).exists()


def test_competitor_that_loses_the_atomic_race_leaves_the_owners_lock_alone(tmp_path, capsys):
    out = tmp_path / "shared"
    out.mkdir()
    owner = cli.OutputReservation(out)

    def owner_reserves_after_the_early_check():
        owner.acquire(["owner"])

    rc = cli.main(_args(out), _before_reserve=owner_reserves_after_the_early_check, **fake())
    assert rc == 2 and "reserved by another invocation" in capsys.readouterr().err
    assert (out / cli.LOCK_NAME).exists()  # the loser did not remove the winner's lock
    assert json.loads((out / cli.LOCK_NAME).read_text(encoding="utf-8"))["token"] == owner.token
    assert [p.name for p in out.iterdir()] == [cli.LOCK_NAME]
    owner.release()
    assert list(out.iterdir()) == []


def test_exactly_one_of_many_concurrent_reservations_wins(tmp_path):
    out = tmp_path / "shared"
    n = 16
    barrier, results = threading.Barrier(n), []

    def attempt():
        r = cli.OutputReservation(out)
        barrier.wait()
        try:
            r.acquire()
            results.append(("won", r))
        except cli.ReservationError:
            results.append(("lost", r))

    threads = [threading.Thread(target=attempt) for _ in range(n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(k for k, _ in results) == ["lost"] * (n - 1) + ["won"]
    winner = next(r for k, r in results if k == "won")
    assert json.loads((out / cli.LOCK_NAME).read_text(encoding="utf-8"))["token"] == winner.token
    winner.release()
    assert list(out.iterdir()) == []


def test_a_stale_lock_is_reported_not_silently_removed(tmp_path, capsys):
    out = tmp_path / "shared"
    out.mkdir()
    (out / cli.LOCK_NAME).write_text('{"token": "dead-run"}', encoding="utf-8")
    rc, _ = run_cli(tmp_path, out="shared")
    err = capsys.readouterr().err
    assert rc == 2 and "reserved by another invocation" in err and "stale lock" in err
    assert (out / cli.LOCK_NAME).read_text(encoding="utf-8") == '{"token": "dead-run"}'  # untouched


def test_release_never_removes_a_lock_it_cannot_prove_it_owns(tmp_path):
    r = cli.OutputReservation(tmp_path / "d").acquire()
    r.lock.write_text('{"token": "someone-else"}', encoding="utf-8")  # replaced behind our back
    r.release()
    assert r.lock.exists()
    r2 = cli.OutputReservation(tmp_path / "e")
    r2.release()  # releasing something never acquired is a no-op


def test_reservation_is_released_on_every_exit_path(tmp_path, monkeypatch):
    q01 = load_dataset(FIXTURE).queries[0].text
    # success
    rc, out = run_cli(tmp_path, "--repetitions", "1", "--warmup", "0", out="ok")
    assert rc == 0 and not (out / cli.LOCK_NAME).exists()
    # run failure
    rc, out = run_cli(tmp_path, per_method={"bm25": {"fail": {(q01, 1): ValueError("x")}}}, out="failed")
    assert rc == 1 and sorted(p.name for p in out.iterdir()) == ["run_failed.json"]
    # export failure
    monkeypatch.setattr(export, "write_run_artifacts", lambda d, r: (_ for _ in ()).throw(export.ExportError("disk full")))
    rc, out = run_cli(tmp_path, "--repetitions", "1", "--warmup", "0", out="export_failed")
    assert rc == 1 and sorted(p.name for p in out.iterdir()) == ["run_failed.json"]
    monkeypatch.undo()

    # interruption (BaseException): propagates, and the reservation is still released
    def interrupted(clk):
        raise KeyboardInterrupt

    clock = FakeClock()
    builders = make_builders(clock, IDS)
    builders[0] = ("bm25", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main(_args(tmp_path / "interrupted"), builders=builders, clock=clock)
    assert list((tmp_path / "interrupted").iterdir()) == []


def test_nested_new_output_path_and_empty_existing_directory_are_reservable(tmp_path):
    rc, out = run_cli(tmp_path, "--repetitions", "1", "--warmup", "0", out="a/b/c")
    assert rc == 0 and (out / "run.json").exists()
    (tmp_path / "empty").mkdir()
    rc, out = run_cli(tmp_path, "--repetitions", "1", "--warmup", "0", out="empty")
    assert rc == 0 and (out / "run.json").exists() and not (out / cli.LOCK_NAME).exists()


def test_failed_acquire_after_a_foreign_file_appears_cleans_only_its_own_lock(tmp_path):
    out = tmp_path / "d"
    out.mkdir()
    (out / "foreign.txt").write_text("x", encoding="utf-8")
    with pytest.raises(cli.ReservationError, match="not empty"):
        cli.OutputReservation(out).acquire()
    assert sorted(p.name for p in out.iterdir()) == ["foreign.txt"]  # our lock removed, their file intact
