"""One-time conversion of WildGuard (allenai/wildguard, gated) to MLX with 8-bit weights.

Prerequisites: accept the WildGuard license on Hugging Face and set HF_TOKEN in .env.
Needs ~15 GB of free disk for the download and ~8 GB for the converted model.

    uv run python scripts/convert_wildguard_mlx.py
"""

from srs.config import SafetyModelEvaluatorConfig, get_secret, load_config


def main() -> None:
    cfg = load_config()
    ecfg = next(e for e in cfg.evaluators.values() if isinstance(e, SafetyModelEvaluatorConfig))
    get_secret("HF_TOKEN")  # fails with a clear message if missing
    out = cfg.root / ecfg.mlx_path
    if (out / "config.json").exists():
        print(f"already converted: {out}")
        return
    from mlx_lm import convert

    convert(hf_path=ecfg.hf_repo, mlx_path=str(out), quantize=True, q_bits=ecfg.q_bits)
    print(f"converted {ecfg.hf_repo} -> {out} ({ecfg.q_bits}-bit)")


if __name__ == "__main__":
    main()
