from pathlib import Path
import pytest
from feed.config import load_config


def test_loads_defaults_when_file_absent(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.embedding.device == "auto"
    assert cfg.embedding.batch_size == 256
    # merge_threshold is an empirically derived constant that Task 12 may
    # revise from the golden-set band. Assert it is sane, not exact, so the
    # two tasks are not coupled.
    assert 0.30 <= cfg.clustering.merge_threshold <= 0.70


def test_file_overrides_defaults(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text(
        "[embedding]\ndevice = \"cpu\"\nbatch_size = 64\n"
        "[clustering]\nmerge_threshold = 0.62\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.embedding.device == "cpu"
    assert cfg.embedding.batch_size == 64
    assert cfg.clustering.merge_threshold == 0.62
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"  # untouched default


def test_rejects_unknown_device(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text("[embedding]\ndevice = \"tpu\"\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_clustering_weights_must_sum_to_one(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text(
        "[clustering]\ncosine_weight = 0.9\nentity_weight = 0.4\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_config(p)


def test_threshold_for_hit(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    cfg.clustering.merge_thresholds = {"BAAI/bge-small-en-v1.5": 0.695, "MiniLM": 0.412}
    assert cfg.clustering.threshold_for("BAAI/bge-small-en-v1.5") == 0.695
    assert cfg.clustering.threshold_for("MiniLM") == 0.412


def test_threshold_for_miss_falls_back_to_merge_threshold(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    cfg.clustering.merge_thresholds = {"MiniLM": 0.412}
    assert cfg.clustering.threshold_for("some-other-model") == cfg.clustering.merge_threshold


def test_threshold_for_none_falls_back_to_merge_threshold(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    cfg.clustering.merge_thresholds = {"MiniLM": 0.412}
    assert cfg.clustering.threshold_for(None) == cfg.clustering.merge_threshold


def test_merge_thresholds_defaults_to_empty_dict(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.clustering.merge_thresholds == {}


def test_providers_defaults(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.providers.gemini_model == "gemini-flash-latest"
    assert cfg.providers.daily_budget == 20
    assert 0.0 <= cfg.providers.tier2_score_cut <= 1.0


def test_providers_file_overrides_defaults(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text(
        "[providers]\ngemini_model = \"gemini-2.5-flash\"\ndaily_budget = 5\n"
        "tier2_score_cut = 0.8\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.providers.gemini_model == "gemini-2.5-flash"
    assert cfg.providers.daily_budget == 5
    assert cfg.providers.tier2_score_cut == 0.8


def test_providers_daily_budget_must_be_positive(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text("[providers]\ndaily_budget = 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


# --- BulkProviderConfig / multi-provider failover chain --------------------

def test_bulk_defaults_to_empty_list(tmp_path):
    """Matches the existing merge_thresholds convention: the pydantic-level
    default is empty, and the real chain lives in feed.toml."""
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.providers.bulk == []
    assert cfg.providers.max_retries == 2
    assert cfg.providers.backoff_base == 0.5
    assert cfg.providers.rate_limit_disable_threshold == 3


def test_bulk_chain_is_parsed_in_priority_order(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text(
        """
[[providers.bulk]]
name = "groq"
kind = "openai_compatible"
model = "openai/gpt-oss-120b"
base_url = "https://api.groq.com/openai/v1"
env_var = "GROQ_API_KEY"

[[providers.bulk]]
name = "gemini"
kind = "gemini"
model = "gemini-flash-latest"
env_var = "GEMINI_API_KEY"
enabled = false
""",
        encoding="utf-8",
    )
    cfg = load_config(p)

    assert [e.name for e in cfg.providers.bulk] == ["groq", "gemini"]
    assert cfg.providers.bulk[0].enabled is True
    assert cfg.providers.bulk[0].base_url == "https://api.groq.com/openai/v1"
    assert cfg.providers.bulk[1].enabled is False
    assert cfg.providers.bulk[1].base_url is None


def test_openai_compatible_entry_requires_base_url(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text(
        '[[providers.bulk]]\nname = "groq"\nkind = "openai_compatible"\n'
        'model = "m"\nenv_var = "GROQ_API_KEY"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(p)


def test_shipped_feed_toml_configures_the_expected_bulk_chain():
    """The real feed.toml this project ships -- not a tmp_path fixture --
    must define the live-tested chain in priority order, with Cerebras
    disabled by default (measured 402 Payment Required)."""
    cfg = load_config(Path("feed.toml"))

    names = [e.name for e in cfg.providers.bulk]
    assert names == ["groq", "mistral", "openrouter", "gemini", "cerebras"]

    by_name = {e.name: e for e in cfg.providers.bulk}
    assert by_name["cerebras"].enabled is False
    for name in ("groq", "mistral", "openrouter", "gemini"):
        assert by_name[name].enabled is True
    assert by_name["gemini"].kind == "gemini"
    for name in ("groq", "mistral", "openrouter", "cerebras"):
        assert by_name[name].kind == "openai_compatible"
        assert by_name[name].base_url is not None


def test_publish_defaults(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.publish.out_dir == "public"
    assert cfg.publish.retention_days == 90
    assert cfg.publish.page_size == 50


def test_publish_file_overrides_defaults(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text(
        "[publish]\nout_dir = \"bundle\"\nretention_days = 30\npage_size = 10\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.publish.out_dir == "bundle"
    assert cfg.publish.retention_days == 30
    assert cfg.publish.page_size == 10


def test_shipped_feed_toml_publishes_a_two_day_window():
    """The owner's decision (2026-08-25): the live feed shows the last 2
    days of news. `retention_days` gates what the bundle CONTAINS, and
    Story.updated_at derives from item.published_at (cluster.py), so this
    is 2 days of news rather than 2 days of crawl history. The
    PublishConfig code default stays at spec 4.4's 90 for anyone running
    without a config file -- this asserts what THIS repo actually ships,
    and is what `observatory.bat` with no argument produces."""
    cfg = load_config(Path("feed.toml"))
    assert cfg.publish.retention_days == 2


def test_shipped_collect_cap_is_narrower_than_the_publish_window():
    """These two windows are independent and easy to confuse. The feed
    window (retention_days) decides what a reader sees; the collect cap
    (max_backfill_days) decides how far back a fetch reaches. Collect must
    not reach LESS far than the feed shows, or a source polled once a day
    could leave holes inside the published window."""
    cfg = load_config(Path("feed.toml"))
    assert cfg.collect.max_backfill_days <= cfg.publish.retention_days
