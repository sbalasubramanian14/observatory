from __future__ import annotations
import subprocess
from pathlib import Path
import pytest
from feed.config import load_config
from feed.doctor import DoctorReport, run_doctor
from feed.providers.base import ProviderError, ProviderHealth


def _cfg(tmp_path, *, groq_enabled=True) -> Path:
    p = tmp_path / "feed.toml"
    p.write_text(
        f'[database]\nurl = "sqlite:///{(tmp_path / "t.db").as_posix()}"\n'
        '[[providers.bulk]]\n'
        'name = "groq"\nkind = "openai_compatible"\nmodel = "m"\n'
        'base_url = "https://api.groq.com/openai/v1"\nenv_var = "DOCTOR_TEST_GROQ_KEY"\n'
        f'enabled = {"true" if groq_enabled else "false"}\n',
        encoding="utf-8",
    )
    return p


class _StubProvider:
    def __init__(self, *, healthy=True, detail="", fails_complete=False):
        self._healthy = healthy
        self._detail = detail
        self._fails_complete = fails_complete

    def health(self):
        return ProviderHealth(healthy=self._healthy, detail=self._detail)

    def complete(self, prompt, *, schema=None):
        if self._fails_complete:
            raise ProviderError("simulated: provider unreachable")
        return "OK"


def _ok_gh_run(args, *, cwd, timeout=60.0):
    if args[:2] == ["auth", "status"]:
        return subprocess.CompletedProcess(
            args, 0, stdout="github.com\n  Logged in\n  Token scopes: 'repo', 'workflow'\n", stderr="")
    if args[:2] == ["repo", "view"]:
        return subprocess.CompletedProcess(args, 0, stdout='{"viewerPermission":"ADMIN"}', stderr="")
    raise AssertionError(f"unexpected gh args: {args}")


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """doctor.py reads relative .env / pyproject.toml paths off the cwd --
    every test here must run somewhere that is NOT the real repo root, or a
    test could silently read (and pass because of) the developer's real
    .env / pyproject.toml instead of the fixture it thinks it's testing.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCTOR_TEST_GROQ_KEY", raising=False)


def test_healthy_system_reports_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCTOR_TEST_GROQ_KEY", "sk-test")
    cfg = load_config(_cfg(tmp_path))
    monkeypatch.setattr("feed.cli._build_bulk_provider", lambda entry, **kw: _StubProvider())
    monkeypatch.setattr("feed.pipeline._run_gh", _ok_gh_run)

    report = run_doctor(cfg, probe_providers=True)

    assert report.ok
    assert any(c.name == "groq" and c.status == "OK" for c in report.checks)
    assert any(c.section == "GitHub / almanac" and c.status == "OK" for c in report.checks)


def test_database_unreachable_is_a_failure(tmp_path, monkeypatch):
    cfg = load_config(_cfg(tmp_path, groq_enabled=False))
    monkeypatch.setattr("feed.doctor.make_engine",
                        lambda url: (_ for _ in ()).throw(RuntimeError("no such driver")))
    monkeypatch.setattr("feed.pipeline._run_gh", _ok_gh_run)

    report = run_doctor(cfg, probe_providers=False)

    db_checks = [c for c in report.checks if c.section == "Database"]
    assert db_checks and db_checks[0].status == "FAIL"
    assert not report.ok


def test_missing_env_key_for_an_enabled_provider_is_a_failure(tmp_path, monkeypatch):
    # DOCTOR_TEST_GROQ_KEY deliberately left unset (see _isolated_cwd's delenv).
    cfg = load_config(_cfg(tmp_path, groq_enabled=True))
    monkeypatch.setattr("feed.pipeline._run_gh", _ok_gh_run)

    report = run_doctor(cfg, probe_providers=False)

    env_checks = {c.name: c for c in report.checks if c.section == ".env"}
    assert env_checks["DOCTOR_TEST_GROQ_KEY"].status == "FAIL"
    assert not report.ok


def test_missing_env_key_for_a_disabled_provider_is_only_a_warning(tmp_path, monkeypatch):
    cfg = load_config(_cfg(tmp_path, groq_enabled=False))
    monkeypatch.setattr("feed.pipeline._run_gh", _ok_gh_run)

    report = run_doctor(cfg, probe_providers=False)

    env_checks = {c.name: c for c in report.checks if c.section == ".env"}
    assert env_checks["DOCTOR_TEST_GROQ_KEY"].status == "WARN"


def test_no_probe_flag_never_calls_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCTOR_TEST_GROQ_KEY", "sk-test")
    cfg = load_config(_cfg(tmp_path))
    stub = _StubProvider()
    called = []
    monkeypatch.setattr(stub, "complete", lambda *a, **k: called.append(1) or "OK")
    monkeypatch.setattr("feed.cli._build_bulk_provider", lambda entry, **kw: stub)
    monkeypatch.setattr("feed.pipeline._run_gh", _ok_gh_run)

    run_doctor(cfg, probe_providers=False)

    assert called == []


def test_all_bulk_providers_unreachable_escalates_to_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCTOR_TEST_GROQ_KEY", "sk-test")
    cfg = load_config(_cfg(tmp_path))
    monkeypatch.setattr("feed.cli._build_bulk_provider",
                        lambda entry, **kw: _StubProvider(fails_complete=True))
    monkeypatch.setattr("feed.pipeline._run_gh", _ok_gh_run)

    report = run_doctor(cfg, probe_providers=True)

    tier1 = [c for c in report.checks if c.name == "TIER 1 (bulk)"]
    assert tier1 and tier1[0].status == "FAIL"
    assert not report.ok


def test_gh_not_authenticated_is_a_failure(tmp_path, monkeypatch):
    cfg = load_config(_cfg(tmp_path, groq_enabled=False))

    def _not_logged_in(args, *, cwd, timeout=60.0):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="not logged in\n")

    monkeypatch.setattr("feed.pipeline._run_gh", _not_logged_in)

    report = run_doctor(cfg, probe_providers=False)

    gh_checks = {c.name: c for c in report.checks if c.section == "GitHub / almanac"}
    assert gh_checks["gh auth"].status == "FAIL"
    assert not report.ok


def test_low_disk_space_is_a_warning(tmp_path, monkeypatch):
    import shutil as shutil_module
    cfg = load_config(_cfg(tmp_path, groq_enabled=False))
    monkeypatch.setattr("feed.pipeline._run_gh", _ok_gh_run)
    Usage = type(shutil_module.disk_usage(tmp_path))
    monkeypatch.setattr(
        "feed.doctor.shutil.disk_usage",
        lambda path: Usage(total=100 * 1024**3, used=97 * 1024**3, free=3 * 1024**3),
    )

    report = run_doctor(cfg, probe_providers=False)

    disk_checks = [c for c in report.checks if c.section == "Disk"]
    assert disk_checks and disk_checks[0].status == "WARN"


def test_report_print_includes_hints_for_failures(tmp_path, monkeypatch, capsys):
    cfg = load_config(_cfg(tmp_path, groq_enabled=True))  # key unset -> FAIL
    monkeypatch.setattr("feed.pipeline._run_gh", _ok_gh_run)

    report = run_doctor(cfg, probe_providers=False)
    report.print()

    out = capsys.readouterr().out
    assert "[FAIL]" in out
    assert "->" in out  # a remediation hint was printed


def test_cmd_doctor_exit_code_reflects_report_ok(tmp_path, monkeypatch):
    from feed.cli import main
    monkeypatch.setenv("DOCTOR_TEST_GROQ_KEY", "sk-test")
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("feed.cli._build_bulk_provider", lambda entry, **kw: _StubProvider())
    monkeypatch.setattr("feed.pipeline._run_gh", _ok_gh_run)

    rc = main(["--config", str(cfg), "doctor", "--no-probe"])
    assert rc == 0


def test_cmd_doctor_exit_code_nonzero_when_broken(tmp_path, monkeypatch):
    from feed.cli import main
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(tmp_path, groq_enabled=True)  # key unset -> FAIL

    def _not_logged_in(args, *, cwd, timeout=60.0):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="not logged in\n")
    monkeypatch.setattr("feed.pipeline._run_gh", _not_logged_in)

    rc = main(["--config", str(cfg), "doctor", "--no-probe"])
    assert rc == 1
