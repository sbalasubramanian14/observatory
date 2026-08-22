"""C3: the Python pipeline's CI workflow must genuinely run the golden-set
clustering regression test (tests/golden/test_golden.py), not just the
`not slow` subset.

Spec S4.7: "The Python pipeline has its own workflow running pytest on
push, including the golden-set clustering test." Spec S6: a regression in
that test "would silently wreck the feed and no other test would catch
it." The shipped workflow ran only `pytest -m "not slow"`, which excludes
both `@pytest.mark.slow` tests in tests/golden/test_golden.py -- so CI
never actually executed the one test the spec calls irreplaceable.

These tests parse the real .github/workflows/pipeline.yml (not a copy) and
check its structure/content directly, since there is no CI runner
available to execute the workflow itself in this environment.
"""
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pipeline.yml"


def _load_workflow() -> dict:
    with WORKFLOW.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _all_run_steps(doc: dict) -> list[str]:
    runs = []
    for job in doc.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "run" in step:
                runs.append(step["run"])
    return runs


def test_workflow_yaml_parses():
    doc = _load_workflow()
    assert "jobs" in doc and doc["jobs"], "workflow must parse and define at least one job"


def test_some_job_runs_the_golden_slow_tests_not_just_not_slow():
    """At least one step across all jobs must invoke pytest in a way that
    actually exercises `@pytest.mark.slow` tests (which is where the golden
    clustering test lives) -- i.e. not exclusively `-m "not slow"`. A
    workflow whose only pytest invocation excludes slow tests fails this,
    exactly reproducing the C3 finding.
    """
    doc = _load_workflow()
    run_steps = _all_run_steps(doc)
    pytest_steps = [r for r in run_steps if "pytest" in r]
    assert pytest_steps, "no step in the workflow runs pytest at all"

    def _includes_slow(run_cmd: str) -> bool:
        # A bare `pytest` (no -m filter) or a step explicitly selecting the
        # slow marker both actually execute the golden test; a step whose
        # only marker expression is `not slow` never does.
        if "-m" not in run_cmd:
            return True
        return "not slow" not in run_cmd

    assert any(_includes_slow(r) for r in pytest_steps), (
        f"every pytest step in the workflow excludes slow tests: {pytest_steps!r} "
        "-- the golden-set clustering test (spec S4.7/S6) would never run in CI"
    )


def test_golden_test_module_is_reachable_by_pytests_default_collection():
    """Sanity check on the other half of the claim: the golden test file
    must actually be collectible under a plain `pytest` invocation (no path
    restriction that would exclude tests/golden/), so that "runs pytest
    without -m 'not slow'" really does reach it.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q",
         "tests/golden/test_golden.py"],
        cwd=WORKFLOW.parents[2],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_blend_recovers_ground_truth_over_a_usable_band" in result.stdout
    assert "test_configured_threshold_sits_inside_the_working_band" in result.stdout
