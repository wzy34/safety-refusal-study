"""Raw data sources: where each benchmark/reference file comes from and how to fetch it.

Files land in `data/raw/<source>/`. Gated Hugging Face datasets need `HF_TOKEN` and an
accepted license on the dataset page; a missing grant fails with the access step to take.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

_XSTEST = "https://raw.githubusercontent.com/paul-rottger/xstest/main"
_HARMBENCH = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data"
_STRONGREJECT = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main"
XSTEST_COMPLETION_MODELS = ("gpt4", "llama2new", "llama2orig", "mistralguard", "mistralinstruct")


@dataclass(frozen=True)
class RawSource:
    name: str
    group: Literal["benchmark", "reference"]
    hf_repo: str | None = None  # Hugging Face dataset repo
    hf_files: tuple[str, ...] = ()
    urls: dict[str, str] = field(default_factory=dict)  # local filename -> URL
    gated: bool = False
    access_step: str = ""

    def files(self) -> list[str]:
        return [Path(f).name for f in self.hf_files] + list(self.urls)


RAW_SOURCES: dict[str, RawSource] = {s.name: s for s in [
    RawSource("xstest", "benchmark",
              urls={"xstest_prompts.csv": f"{_XSTEST}/xstest_prompts.csv"}),
    RawSource("or_bench", "benchmark", hf_repo="bench-llm/or-bench",
              hf_files=("or-bench-hard-1k.csv", "or-bench-toxic.csv")),
    RawSource("jbb", "benchmark", hf_repo="JailbreakBench/JBB-Behaviors",
              hf_files=("data/harmful-behaviors.csv", "data/benign-behaviors.csv",
                        "data/judge-comparison.csv")),
    RawSource("harmbench", "benchmark",
              urls={"harmbench_behaviors_text_all.csv":
                    f"{_HARMBENCH}/behavior_datasets/harmbench_behaviors_text_all.csv",
                    "text_behaviors_val_set.json":
                    f"{_HARMBENCH}/classifier_val_sets/text_behaviors_val_set.json"}),
    RawSource("sorry_bench", "benchmark", hf_repo="sorry-bench/sorry-bench-202406",
              hf_files=("question.jsonl", "meta_info.py"), gated=True,
              access_step="Accept the SORRY-Bench terms at "
                          "https://huggingface.co/datasets/sorry-bench/sorry-bench-202406 "
                          "with your Hugging Face account, then set HF_TOKEN in .env."),
    RawSource("strongreject", "benchmark",
              urls={"strongreject_dataset.csv":
                    f"{_STRONGREJECT}/strongreject_dataset/strongreject_dataset.csv"}),
    RawSource("wildguardmix", "reference", hf_repo="allenai/wildguardmix",
              hf_files=("test/wildguard_test.parquet", "train/wildguard_train.parquet"),
              gated=True,
              access_step="Accept the WildGuardMix license at "
                          "https://huggingface.co/datasets/allenai/wildguardmix "
                          "with your Hugging Face account, then set HF_TOKEN in .env."),
    RawSource("xstest_completions", "reference",
              urls={f"xstest_v2_completions_{m}.csv":
                    f"{_XSTEST}/model_completions/xstest_v2_completions_{m}.csv"
                    for m in XSTEST_COMPLETION_MODELS}),
]}


class FetchError(Exception):
    pass


def _hf_token() -> str | None:
    from srs.config import load_dotenv_once

    load_dotenv_once()
    return os.environ.get("HF_TOKEN") or None


def _fetch_hf(src: RawSource, dest: Path) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    token = _hf_token()
    if src.gated and not token:
        raise FetchError(f"{src.name}: gated dataset {src.hf_repo} needs HF_TOKEN, which is not "
                         f"set. Access step: {src.access_step}")
    for f in src.hf_files:
        try:
            path = hf_hub_download(src.hf_repo, f, repo_type="dataset", token=token)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise FetchError(f"{src.name}: no access to {src.hf_repo} ({type(e).__name__}). "
                             f"Access step: {src.access_step or 'check the repo name'}") from e
        except HfHubHTTPError as e:
            raise FetchError(f"{src.name}: download of {src.hf_repo}/{f} failed: {e}") from e
        shutil.copyfile(path, dest / Path(f).name)


def _fetch_url(src: RawSource, dest: Path) -> None:
    for fname, url in src.urls.items():
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except OSError as e:
            raise FetchError(f"{src.name}: download of {url} failed: {e}") from e
        (dest / fname).write_bytes(data)


def fetch_source(src: RawSource, raw_dir: Path, force: bool = False) -> Path:
    dest = raw_dir / src.name
    if not force and dest.exists() and all((dest / f).exists() for f in src.files()):
        log.info("%s: already present", src.name)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if src.hf_repo:
        _fetch_hf(src, dest)
    if src.urls:
        _fetch_url(src, dest)
    log.info("%s: fetched %d files", src.name, len(src.files()))
    return dest


def fetch_all(raw_dir: Path, names: list[str] | None = None,
              force: bool = False) -> dict[str, str | None]:
    """Fetch every named source; return {source: None on success, or the error message}."""
    results: dict[str, str | None] = {}
    for name in names or list(RAW_SOURCES):
        try:
            fetch_source(RAW_SOURCES[name], raw_dir, force)
            results[name] = None
        except FetchError as e:
            results[name] = str(e)
    return results


def missing_raw_error(name: str, filename: str | None) -> FetchError:
    """A FetchError naming the dataset, the missing file and the access step to take."""
    src = RAW_SOURCES[name]
    step = (src.access_step + " Then run `srs data fetch`.") if src.gated \
        else "run `srs data fetch`."
    return FetchError(f"{name}: raw file missing ({filename}). Access step: {step}")


def benchmark_loaders():
    """raw source name -> loader(raw_dir) -> list[Prompt] (direct framing, category unset)."""
    from srs.data.sources import harmbench, jbb, or_bench, sorry_bench, strongreject, xstest

    return {"xstest": xstest.load, "or_bench": or_bench.load, "jbb": jbb.load,
            "harmbench": harmbench.load, "sorry_bench": sorry_bench.load,
            "strongreject": strongreject.load}
