"""
Simple evaluation entry‑point for Select‑All‑That‑Apply (SATA) benchmarks.

Typical usage
-------------

```bash
# Full evaluation on an OpenAI model.
python3 -m satabench.evaluation.simple_eval --limit 5 --dataset satabench/evaluation/dataset/final_A_multic_choice.csv --model gpt-4o-mini --repair_model gpt-4.1-mini --metrics-out satabench/eval
uation/dataset/gpt4o_predictions.json
```

The script is **agnostic** to the particular metric set – ``metrics.collect_metrics``
may return any key/value pairs. Everything is serialised to ``--metrics-out`` as
JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
# We re‑export symbols to avoid mypy/IDE complaints after the dynamic import.
from ..evaluation.sata_eval import run_inference  # type: ignore  # noqa: E402
from ..evaluation.metrics.metrics import collect_metrics  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)


# ------------------------------------------------------------------------------ CLI ----

def build_arg_parser() -> argparse.ArgumentParser:
    """Return a fully‑populated :class:`argparse.ArgumentParser`."""
    p = argparse.ArgumentParser(
        description=(
            "Run a complete SATA evaluation: model → predictions → metric "
            "aggregation.  Requires `sata_eval.py` and `metrics.py`."
        )
    )

    # I/O
    p.add_argument("--dataset", type=Path, required=True, help="Path to CSV dataset file.")
    p.add_argument("--output", type=Path, default=None, help="Where to write raw predictions (CSV).")
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Where to write aggregated metric JSON.  If omitted, just print to stdout.",
    )

    # Model + decoding controls.  These are forwarded verbatim to `sata_eval.run_inference`.
    p.add_argument("--model", type=str, required=True, help="Model identifier or checkpoint tag.")
    p.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "bedrock", "hf", "local"],
        help="Model hosting provider.  `sata_eval` may dispatch differently based on this.",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    p.add_argument("--top-p", dest="top_p", type=float, default=0.95, help="Nucleus sampling *p* fraction.")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=1000, help="Generation length cap.")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=1, help="Requests per API batch.")

    # Convenience / debugging flags.
    p.add_argument(
        "--debug", action="store_true", help="Run in debug mode (verbose logs + limit rows)."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only evaluate the first *N* rows of the dataset (after any shuffle).",
    )
    p.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset before optional truncation via --limit."
    )
    p.add_argument("--repair_model", type=str, default="gpt-4o-mini", help="Model identifier or checkpoint tag.")

    return p


# ---------------------------------------------------------------------- helpers --------


def _prepare_dataframe(path: Path, shuffle: bool, limit: int | None, debug: bool) -> pd.DataFrame:
    """Load *path* into a :class:`pandas.DataFrame` and optionally subsample."""
    logger.info("Loading dataset: %s", path)
    df = pd.read_csv(path)

    # Expect at least these 2 columns – raise early if absent.
    missing_cols = {c for c in ("text", "answer") if c not in df.columns}
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")

    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if limit is not None:
        df = df.iloc[:limit].copy()

    if debug:
        logger.debug("Preview after loading & slicing:\n%s", df.head())
    return df


# ------------------------------------------------------------------------- main --------


def main(argv: List[str] | None = None) -> None:  # noqa: C901 – not too complex here
    args = build_arg_parser().parse_args(argv)

    df = _prepare_dataframe(
        path=args.dataset,
        shuffle=args.shuffle,
        limit=args.limit,
        debug=args.debug,
    )

    # ----------------------------------------------------------- Inference --------------
    logger.info("Running inference → provider=%s | model=%s", args.provider, args.model)
    new_df = run_inference(
        df=df,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        debug=args.debug,
        repair_model=args.repair_model,
    )

    if len(new_df) != len(df):
        raise RuntimeError(
            "Length mismatch: run_inference returned %d predictions for %d inputs"
            % (len(new_df), len(df))
        )
    
    #df["predicted_answer"] = predictions

    # ---------------------------------------------------------------- save preds -------
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(args.output, index=False)
        logger.info("Wrote raw predictions → %s", args.output)

    # ------------------------------------------------------------ Metric aggregation ----
    logger.info("Scoring predictions vs. ground truth …")
    
    metric_dict: Dict[str, Any] = collect_metrics(
        labels=new_df["answer"].tolist(),
        preds=new_df["predicted_answer"].tolist(),
        maximum=new_df["maximum"].tolist(),
    )

    # Pretty‑print summary.
    widest = max(len(k) for k in metric_dict) + 2
    logger.info("\n" + "\n".join(f"{k:<{widest}}{v}" for k, v in metric_dict.items()))

    # ---------------------------------------------------------------- save metrics ------
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w", encoding="utf-8") as fp:
            json.dump(metric_dict, fp, indent=2)
        logger.info("Wrote metrics JSON → %s", args.metrics_out)


if __name__ == "__main__":  # pragma: no cover
    main()
