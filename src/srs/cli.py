"""`srs` command-line interface. Each pipeline stage is a subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from srs.config import ConfigError, StudyConfig, load_config
from srs.log import setup_logging

app = typer.Typer(help="Safety refusal study pipeline.", no_args_is_help=True,
                  pretty_exceptions_enable=False)
data_app = typer.Typer(help="Fetch benchmarks, build the prompt set and render framings.",
                       no_args_is_help=True)
classic_app = typer.Typer(help="Train the RoBERTa refusal classifier.", no_args_is_help=True)
annotate_app = typer.Typer(help="Draw the annotation sample, run the blind app, import labels.",
                           no_args_is_help=True)
analyze_app = typer.Typer(help="Compute metrics, tests, tables and figures for RQ1-RQ5 and "
                          "the stability check.", no_args_is_help=True)
llm_app = typer.Typer(help="LLM client utilities.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(classic_app, name="classic")
app.add_typer(annotate_app, name="annotate")
app.add_typer(analyze_app, name="analyze")
app.add_typer(llm_app, name="llm")

_state: dict[str, object] = {}


@app.callback()
def main(
    config_dir: Annotated[Path | None,
                          typer.Option(help="Directory with the YAML configs.")] = None,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    setup_logging(log_level.upper())
    _state["config_dir"] = config_dir


def get_cfg() -> StudyConfig:
    if "cfg" not in _state:
        try:
            _state["cfg"] = load_config(_state.get("config_dir"))  # type: ignore[arg-type]
        except ConfigError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(2) from e
    return _state["cfg"]  # type: ignore[return-value]


# --- data ------------------------------------------------------------------------------------


@data_app.command("fetch")
def data_fetch(
    sources: Annotated[list[str] | None, typer.Option(help="Source names (default: all).")] = None,
    force: Annotated[bool, typer.Option(help="Re-download files already present.")] = False,
) -> None:
    """Download the six benchmarks and the reference sets into data/raw/."""
    from srs.data.sources import fetch_all

    cfg = get_cfg()
    results = fetch_all(cfg.path("data_dir") / "raw", sources, force)
    for name, err in results.items():
        if err is None:
            typer.secho(f"ok      {name}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"FAILED  {err}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1 if any(results.values()) else 0)


def _real_client(cfg: StudyConfig, model_ids: list[str]):
    from srs.llm.client import ChatClient

    try:
        return ChatClient.real(cfg, model_ids)
    except ConfigError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e


@data_app.command("build")
def data_build() -> None:
    """Build prompts.parquet (direct framing) with taxonomy, tiers, pairs and splits."""
    import asyncio

    from srs.data.build import build_dataset
    from srs.data.sources import FetchError
    from srs.manifest import RunManifest
    from srs.templates import load_template

    cfg = get_cfg()
    tx = cfg.taxonomy
    client = _real_client(cfg, sorted({m for m in (tx.category_model, tx.tier_screen_model)
                                       if m}))
    with RunManifest("data build", cfg) as m:
        m.extra["templates"] = {n: load_template(cfg.path("prompts_dir"), n).version
                                for n in ("category_assign.md", "tier_screen.md")}
        try:
            res = asyncio.run(build_dataset(cfg, client))
        except FetchError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from e
        m.counts = {"prompts": len(res.prompts), "dedup_removed": len(res.dedup_removed)}
    typer.echo(f"wrote {len(res.prompts)} prompts; report: {res.report_path}")


@data_app.command("frame")
def data_frame(
    limit: Annotated[int | None, typer.Option(help="Only frame the first N RQ3 intents.")] = None,
) -> None:
    """Generate and intent-check the non-direct framings for the RQ3 split."""
    import asyncio

    from srs.data.framing import frame_rq3
    from srs.manifest import RunManifest
    from srs.templates import load_template

    cfg = get_cfg()
    fr = cfg.framings
    client = _real_client(cfg, [fr.rewriter_model, fr.checker_model])
    with RunManifest("data frame", cfg, args={"limit": limit}) as m:
        names = [t.rewrite_prompt for t in fr.templates.values() if t.rewrite_prompt]
        m.extra["templates"] = {n: load_template(cfg.path("prompts_dir"), n).version
                                for n in [*names, fr.intent_check_prompt]}
        res = asyncio.run(frame_rq3(cfg, client, limit))
        m.counts = {"framed_prompts": len(res.framed), "dropped_intents": len(res.dropped)}
    typer.echo(f"framed prompts: {len(res.framed)}; dropped intents: {len(res.dropped)}")


# --- generate / evaluate ---------------------------------------------------------------------


@app.command("generate")
def generate(
    split: Annotated[str, typer.Option("--set", help="Prompt split: pilot, main or rq3.")],
    models: Annotated[list[str] | None, typer.Option(
        help="Evaluated model ids (default: all evaluated models).")] = None,
    limit: Annotated[int | None, typer.Option(help="At most N prompts per model.")] = None,
    stability: Annotated[bool, typer.Option(
        help="Run the stability subset (samples 1..n at the stability temperature).")] = False,
    framed_only: Annotated[bool, typer.Option(
        help="RQ3 only: keep intents that already have framings (e.g. a pilot slice).")] = False,
    dry_run: Annotated[bool, typer.Option(
        help="Report planned calls and estimated tokens; send nothing.")] = False,
) -> None:
    """Collect model responses for a prompt split."""
    import asyncio

    from srs.data.build import load_prompts
    from srs.generation import runner
    from srs.io import read_records
    from srs.manifest import RunManifest
    from srs.schemas import Response

    cfg = get_cfg()
    model_ids = models or cfg.evaluated
    unknown = [m for m in model_ids if m not in cfg.models]
    if unknown:
        typer.secho(f"unknown model ids: {unknown}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    prompts = load_prompts(cfg)
    if stability:
        if split != cfg.experiment.stability.split:
            typer.secho(f"the stability subset is drawn from split "
                        f"'{cfg.experiment.stability.split}'", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        selected = runner.stability_prompts(cfg, prompts)
    else:
        selected = runner.select_split(prompts, split)
    if framed_only:
        if split != "rq3" or stability:
            typer.secho("--framed-only applies to --set rq3 without --stability",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        framed = {p.intent_id for p in selected if p.framing != "direct"}
        selected = [p for p in selected if p.intent_id in framed]
    if not selected:
        typer.secho(f"no prompts in split '{split}' (run `srs data build` first?)",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    jobs = runner.plan_jobs(cfg, selected, model_ids, stability=stability, limit=limit)
    if dry_run:
        existing = read_records(runner.responses_path(cfg), Response)
        rows = runner.dry_run_estimate(cfg, jobs, existing)
        for r in rows:
            cost = f" est_cost=${r['est_cost_usd']} (max ${r['max_cost_usd']})" \
                if "est_cost_usd" in r else ""
            typer.echo(f"{r['model_id']:14} responses={r['responses']:6} calls={r['calls']:6} "
                       f"in~{r['est_input_tokens']:>9} out~{r['est_output_tokens']:>9} "
                       f"(max {r['max_output_tokens']}){cost}")
        typer.echo(f"total calls: {sum(r['calls'] for r in rows)} (dry run, nothing sent)")
        return
    client = _real_client(cfg, model_ids)
    args = {"set": split, "models": model_ids, "limit": limit, "stability": stability,
            "framed_only": framed_only}
    with RunManifest("generate", cfg, args=args) as m:
        counts = asyncio.run(runner.run_generation(cfg, client, jobs, m.run_id))
        m.counts = counts
        m.extra["responses_path"] = str(runner.responses_path(cfg))
    typer.echo(f"run {m.run_id}: {counts}")


@classic_app.command("train")
def classic_train(
    evaluator: Annotated[str, typer.Option(help="Classifier evaluator id.")] = "classic",
    max_items: Annotated[int | None, typer.Option(
        help="Use a seeded random subset of N training items (for quick runs).")] = None,
) -> None:
    """Fine-tune RoBERTa-base on WildGuardMix-train response-refusal labels."""
    import random

    from srs.config import ClassifierEvaluatorConfig
    from srs.data.sources import FetchError, missing_raw_error
    from srs.evaluators import classic
    from srs.manifest import RunManifest
    from srs.reference.external import load_reference_items

    cfg = get_cfg()
    ecfg = cfg.evaluators[evaluator]
    if not isinstance(ecfg, ClassifierEvaluatorConfig):
        typer.secho(f"{evaluator} is not a classifier evaluator", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    raw = cfg.path("data_dir") / "raw"
    try:
        refs, _ = load_reference_items(raw, require_all=True)
        try:
            items = classic.wildguardmix_train_items(raw)
        except FileNotFoundError as e:
            raise missing_raw_error("wildguardmix", e.filename) from e
    except FetchError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    items, removed = classic.apply_leakage_guard(items, refs)
    if max_items is not None and max_items < len(items):
        items = random.Random(cfg.experiment.seed).sample(items, max_items)
    out = cfg.root / ecfg.checkpoint_dir
    with RunManifest("classic train", cfg, args={"evaluator": evaluator,
                                                 "max_items": max_items}) as m:
        report = classic.train_classifier(items, ecfg, out, cfg.experiment.seed, removed)
        m.counts = {"train_items": len(items), "removed_by_leakage_guard": removed}
        m.extra["classifier_version"] = report["version"]
        m.extra["best"] = report["best"]
    typer.echo(f"saved {report['version']} to {out}; best: {report['best']}")


@app.command("evaluate")
def evaluate(
    evaluators: Annotated[list[str] | None, typer.Option(
        help="Evaluator ids (default: all configured).")] = None,
    target: Annotated[str, typer.Option(
        help="'responses' (study responses) or 'external' (human-labeled reference sets).")
    ] = "responses",
    sources: Annotated[list[str] | None, typer.Option(
        help="External target only: judge just these reference sets (e.g. harmbench_val).")
    ] = None,
) -> None:
    """Label responses (or external reference items) with the configured evaluators."""
    from srs.data.build import processed_dir
    from srs.evaluators import judge_models, make_evaluator
    from srs.evaluators.base import (
        eligible_responses,
        item_from_reference,
        judgments_path,
        run_evaluators,
    )
    from srs.generation.runner import responses_path
    from srs.io import read_records, write_records
    from srs.manifest import RunManifest
    from srs.reference.external import load_reference_items
    from srs.schemas import Response

    cfg = get_cfg()
    if target not in ("responses", "external"):
        typer.secho("--target must be 'responses' or 'external'", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    if sources and target != "external":
        typer.secho("--sources applies to --target external", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    ids = evaluators or list(cfg.evaluators)
    unknown = [e for e in ids if e not in cfg.evaluators]
    if unknown:
        typer.secho(f"unknown evaluator ids: {unknown}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    judges = judge_models(cfg, ids)
    client = _real_client(cfg, judges) if judges else None
    with RunManifest("evaluate", cfg, args={"evaluators": ids, "target": target,
                                            "sources": sources}) as m:
        if target == "responses":
            responses = read_records(responses_path(cfg), Response)
            items, skipped = eligible_responses(responses)
            m.counts.update(skipped)
        else:
            refs, missing = load_reference_items(cfg.path("data_dir") / "raw", require_all=False)
            for err in missing.values():
                typer.secho(f"skipping: {err}", fg=typer.colors.YELLOW, err=True)
            write_records(processed_dir(cfg) / "reference_items.parquet", refs)
            if sources:
                unknown_src = set(sources) - {r.source for r in refs}
                if unknown_src:
                    typer.secho(f"no reference items for sources: {sorted(unknown_src)}",
                                fg=typer.colors.RED, err=True)
                    raise typer.Exit(2)
                refs = [r for r in refs if r.source in sources]
            items = [item_from_reference(r) for r in refs]
        evs = [make_evaluator(cfg, e, client) for e in ids]
        m.extra["evaluator_versions"] = {ev.id: ev.version() for ev in evs}
        out = judgments_path(cfg, target)  # type: ignore[arg-type]
        m.counts.update(run_evaluators(evs, items, out, target, m.run_id))  # type: ignore[arg-type]
        m.extra["judgments_path"] = str(out)
    typer.echo(f"run {m.run_id}: {dict(m.counts)}")


# --- annotate --------------------------------------------------------------------------------


@annotate_app.command("sample")
def annotate_sample(
    name: Annotated[str, typer.Option(help="Sample name (e.g. main, pilot).")] = "main",
    size: Annotated[int | None, typer.Option(
        help="Sample size (default: annotation.sample_size; use 100 for the pilot).")] = None,
    splits: Annotated[list[str] | None, typer.Option(
        help="Prompt splits to sample from (default: main and rq3).")] = None,
) -> None:
    """Draw the stratified, weighted self-annotation sample."""
    from srs.data.build import load_prompts
    from srs.evaluators.base import judgments_path
    from srs.generation.runner import responses_path
    from srs.io import read_records
    from srs.manifest import RunManifest
    from srs.reference import sampling
    from srs.schemas import Judgment, Response

    cfg = get_cfg()
    a = cfg.experiment.annotation
    out = sampling.annotation_dir(cfg, name)
    if (out / "pass1.parquet").exists():
        typer.secho(f"{out} already has pass-1 labels; refusing to redraw the sample.",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    use_splits = tuple(splits or ["main", "rq3"])
    units = sampling.eligible_units(read_records(responses_path(cfg), Response), load_prompts(cfg),
                                    read_records(judgments_path(cfg, "responses"), Judgment),
                                    use_splits)
    n = size or a.sample_size
    with RunManifest("annotate sample", cfg, args={"name": name, "size": n,
                                                   "splits": list(use_splits)}) as m:
        sample = sampling.draw_sample(units, n, cfg.experiment.seed, a.disagreement_oversample)
        sampling.write_sample(out, sample, cfg.experiment.seed, len(units), name)
        m.counts = {"eligible": len(units), "sampled": len(sample)}
    typer.echo(f"sampled {len(sample)} of {len(units)} eligible responses into {out}")


@annotate_app.command("app")
def annotate_run_app(
    name: Annotated[str, typer.Option(help="Sample name.")] = "main",
    pass_: Annotated[str, typer.Option("--pass", help="1, 2 or adjudicate.")] = "1",
    port: Annotated[int, typer.Option(help="Local port.")] = 8501,
) -> None:
    """Launch the blind Streamlit annotation app."""
    import os
    import subprocess
    import sys

    from srs.reference import annotations, sampling

    cfg = get_cfg()
    d = sampling.annotation_dir(cfg, name)
    if not (d / "items.parquet").exists():
        typer.secho(f"no sample at {d}; run `srs annotate sample --name {name}` first",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if pass_ not in ("1", "2", "adjudicate"):
        typer.secho("--pass must be 1, 2 or adjudicate", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    env = {**os.environ, "SRS_ANNOTATION_DIR": str(d), "SRS_ANNOTATION_PASS": pass_,
           "SRS_SEED": str(cfg.experiment.seed),
           "SRS_PASS2_MIN_DAYS": str(cfg.experiment.annotation.pass2_min_days)}
    cmd = [sys.executable, "-m", "streamlit", "run", str(annotations.APP_PATH),
           "--server.port", str(port), "--server.address", "localhost",
           "--browser.gatherUsageStats", "false"]
    raise typer.Exit(subprocess.call(cmd, env=env))


@annotate_app.command("import")
def annotate_import(
    name: Annotated[str, typer.Option(help="Sample name.")] = "main",
    single_pass: Annotated[bool, typer.Option(
        "--single-pass", help="Use pass-1 labels as final, with no pass 2 (pilot only).")
    ] = False,
) -> None:
    """Import annotation passes, report test-retest agreement, list disagreements."""
    from srs.manifest import RunManifest
    from srs.reference import annotations, sampling

    cfg = get_cfg()
    d = sampling.annotation_dir(cfg, name)
    with RunManifest("annotate import", cfg,
                     args={"name": name, "single_pass": single_pass}) as m:
        try:
            rep = annotations.import_annotations(d, single_pass=single_pass)
        except ValueError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(2) from e
        m.counts = {k: rep[k] for k in ("n_pass1", "n_pass2", "n_final", "n_disagreements")}
    if single_pass:
        typer.echo(f"{rep['n_final']} single-pass final labels, "
                   f"{len(rep['pending_adjudication'])} items not yet labeled; see {d}")
        return
    typer.echo(f"refusal kappa={rep['refusal_kappa']} harmful kappa={rep['harmful_kappa']}; "
               f"{rep['n_disagreements']} disagreements, "
               f"{len(rep['pending_adjudication'])} awaiting adjudication; see {d}/reliability.md")


# --- analyze ---------------------------------------------------------------------------------


def _analyze(rqs: list[str], annotation: str, splits: list[str] | None = None,
             external: bool = True, borderline: bool | None = None) -> None:
    import importlib
    from dataclasses import replace

    import pandas as pd

    from srs.analysis.data import load_analysis_data
    from srs.manifest import RunManifest, new_run_id

    cfg = get_cfg()
    if splits:
        cfg.experiment.analysis.primary_splits = splits  # type: ignore[assignment]
    if borderline is not None:
        cfg.experiment.analysis.include_borderline = borderline
    run_id = new_run_id("analyze")
    out = cfg.path("results_dir") / "analysis" / run_id
    with RunManifest("analyze", cfg, args={"rqs": rqs, "annotation": annotation,
                                           "splits": cfg.experiment.analysis.primary_splits,
                                           "external": external,
                                           "include_borderline":
                                               cfg.experiment.analysis.include_borderline},
                     out_dir=out, run_id=run_id) as m:
        data = load_analysis_data(cfg, annotation)
        if not external:
            data = replace(data, judg_ext=pd.DataFrame(), refs=pd.DataFrame(),
                           inputs=[i for i in data.inputs if "external" not in i
                                   and "reference_items" not in i])
        if data.resp.empty:
            typer.secho("no responses to analyze (run generate and evaluate first)",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        m.extra.update({"inputs": data.inputs, "seed": cfg.experiment.seed,
                        "primary_evaluator": cfg.experiment.analysis.primary_evaluator,
                        "primary_rule": cfg.experiment.analysis.primary_rule})
        files: list[str] = []
        for rq in rqs:
            mod = importlib.import_module(f"srs.analysis.{rq}")
            files += [str(p.relative_to(out)) for p in mod.run(data, cfg, out / rq)]
        m.extra["outputs"] = sorted(files)
        m.counts = {"files": len(files), "responses": len(data.resp)}
    typer.echo(f"wrote {len(files)} files to {out}")


_SPLITS_HELP = "Splits for the primary analyses (default: analysis.primary_splits, main+rq3)."
_EXTERNAL_HELP = "Use the external reference sets in RQ5 (--no-external: study responses only)."
_BORDERLINE_HELP = ("Include borderline-tier prompts in every metric (default: "
                    "analysis.include_borderline, i.e. clear-tier only).")


def _make_rq_command(rq: str) -> None:
    def command(annotation: Annotated[str, typer.Option(
            help="Self-annotation sample name used as the RQ5 reference.")] = "main",
                splits: Annotated[list[str] | None, typer.Option(help=_SPLITS_HELP)] = None,
                external: Annotated[bool, typer.Option(help=_EXTERNAL_HELP)] = True,
                borderline: Annotated[bool | None, typer.Option(
                    "--borderline/--clear-only", help=_BORDERLINE_HELP)] = None,
                ) -> None:
        _analyze([rq], annotation, splits, external, borderline)

    command.__doc__ = f"Run the {rq.upper()} analysis."
    analyze_app.command(rq)(command)


for _rq in ("rq1", "rq2", "rq3", "rq4", "rq5", "stability"):
    _make_rq_command(_rq)


@analyze_app.command("all")
def analyze_all(
    annotation: Annotated[str, typer.Option(
        help="Self-annotation sample name used as the RQ5 reference.")] = "main",
    splits: Annotated[list[str] | None, typer.Option(help=_SPLITS_HELP)] = None,
    external: Annotated[bool, typer.Option(help=_EXTERNAL_HELP)] = True,
    borderline: Annotated[bool | None, typer.Option(
        "--borderline/--clear-only", help=_BORDERLINE_HELP)] = None,
) -> None:
    """Run every RQ analysis and the stability check."""
    _analyze(["rq1", "rq2", "rq3", "rq4", "rq5", "stability"], annotation, splits, external,
             borderline)


# --- llm -------------------------------------------------------------------------------------


@llm_app.command("ping")
def llm_ping(
    models: Annotated[list[str] | None, typer.Option(help="Model ids (default: all).")] = None,
) -> None:
    """Send one trivial request to each configured model (uncached) and report its status."""
    import asyncio

    from srs.llm.adapters import GenParams
    from srs.llm.client import ChatClient

    cfg = get_cfg()
    ids = models or list(cfg.models)
    try:
        client = ChatClient.real(cfg, ids)
    except ConfigError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e
    msgs = [{"role": "user", "content": "Reply with the single word OK."}]
    params = GenParams(max_tokens=cfg.experiment.generation.max_tokens, temperature=0.0)
    comps = asyncio.run(client.complete_many([(m, msgs, params) for m in ids],
                                             use_cache=False))
    failed = 0
    for mid, c in zip(ids, comps, strict=True):
        detail = c.error or c.block_code or ""
        typer.echo(f"{mid:16} {c.status:12} finish={c.finish_reason!s:14} "
                   f"model={c.model_version!s:28} latency={c.latency_s}s {detail[:80]}")
        failed += c.status != "ok"
    raise typer.Exit(1 if failed else 0)
