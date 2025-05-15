#!/usr/bin/env python3
# sata_eval.py
# ---------------------------------------------------------------------
"""
Select‑All‑That‑Apply evaluation helper.

* Reads a CSV with columns  `text`  (question) and  `answer`  (ground truth)
* Calls a chat model to get predictions   -- or uses debug mode
* Parses each raw model output into canonical letters (`'ACD'` etc.)
  using:
      1. JSON repair helpers you provided
      2. regex fallback
      3. batched OpenAI call for lines still unparsed
* Returns a DataFrame ready for metric computation.
"""
from __future__ import annotations


import json
import logging
import os
import random
import re
import time
from typing import List, Optional, Union
from ..evaluation.model_class import OpenAIBackend, ChoiceBatchParser
import numpy as np
import openai
import pandas as pd

# ---------------------------------------------------------------------
# 0 .  global config / small helpers
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


# Regex used once the text is valid JSON‑like
_SINGLE_LETTER_RE = re.compile(r"[A-O]")

# Regex used in fallback for free‑text answers
_FALLBACK_RE = re.compile(r"([A-O]{1,10})(?=\s*[.:|,}>\]]|\s*$)", re.I)


def check_nan(v) -> bool:
    """Gracefully spot pandas/NumPy NaNs inside mixed‑type columns."""
    try:
        return np.isnan(v)
    except TypeError:
        return False


# ---------------------------------------------------------------------
# 1 .  helpers you shared – slightly tidied
# ---------------------------------------------------------------------
def add_quotes_to_uppercase_values(text: str) -> str:
    pattern = r'("(?P<key>\w+)":\s*)(?P<value>[A-Z]+(?:\s*,\s*[A-Z]+)*)'

    def repl(m):
        key_part = m.group(1)
        v = m.group("value")
        if "," in v:
            letters = [x.strip() for x in v.split(",")]
            quoted = ", ".join(f'"{l}"' for l in letters)
        else:
            quoted = ", ".join(f'"{c}"' for c in v)
        return f"{key_part}{quoted}"

    return re.sub(pattern, repl, text)


def fix_unquoted_strings_in_json(json_str: str) -> str:
    pattern = r"(\[\s*)([^]\[]+?)(\s*\])"

    def repl(m):
        start, items_str, end = m.groups()
        items = items_str.split(",")
        fixed = []
        for it in items:
            it = it.strip()
            if not (
                (it.startswith('"') and it.endswith('"'))
                or re.match(r"^-?\d+(\.\d+)?$", it)
            ):
                it = '"' + it.strip('"\'') + '"'
            fixed.append(it)
        return start + ", ".join(fixed) + end

    return re.sub(pattern, repl, json_str)


# ---------------------------------------------------------------------
# 2 .  robust parsing pipeline
# ---------------------------------------------------------------------
def _extract_letters(obj: Union[dict, list, str, int, float]) -> List[str]:
    letters = set()
    if isinstance(obj, str):
        letters.update(_SINGLE_LETTER_RE.findall(obj.upper()))
    elif isinstance(obj, list):
        for x in obj:
            letters.update(_extract_letters(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            letters.update(_extract_letters(v))
    return list(letters)


def _json_parse(txt: str) -> str:
    """Attempt to coerce *almost* JSON and pull A‑H letters."""
    if check_nan(txt) or not isinstance(txt, str) or not txt.strip():
        return ""

    clean = add_quotes_to_uppercase_values(txt)
    clean = fix_unquoted_strings_in_json(clean)

    try:
        obj = json.loads(clean)
    except Exception:
        return ""

    found = _extract_letters(obj)
    return "".join(sorted(set(found)))


def _regex_fallback(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    hits = _FALLBACK_RE.findall(txt.upper())
    
    return "".join(sorted(set("".join(hits))))


def parse_function(
    raw_list: List[str], llm_batch_size: int = 20, repair_model: str = "gpt-4o-mini"
) -> List[str]:
    """Main public parser — JSON → regex → OpenAI repair."""
    parsed = []
    need_repair_idx = []

    # Pass 1: JSON or regex
    for idx, txt in enumerate(raw_list):
        if isinstance(txt, str) and any(
            bad in txt for bad in ("PARSINGERROR", "I can’t help", "I can’t fulfill")
        ):
            parsed.append("")
            continue

        out = _json_parse(txt)
        if not out:
            out = _regex_fallback(txt)

        parsed.append(out)
        if out == "":
            need_repair_idx.append(idx)

    # Pass 2: OpenAI repair
    if need_repair_idx:
        logger.info("LLM repair needed for %d / %d rows", len(need_repair_idx), len(raw_list))

    for start in range(0, len(need_repair_idx), llm_batch_size):
        batch_idx = need_repair_idx[start : start + llm_batch_size]
        subset = [raw_list[i] for i in batch_idx]

        # simple retry with back‑off
        for attempt in range(4):
            try:
                llm_parser = ChoiceBatchParser(model = repair_model)
                fixed = llm_parser(subset)
                break
            except Exception as e:
                if attempt == 3:
                    logger.error("OpenAI repair failed – giving up: %s", e)
                    fixed = ["" for _ in subset]
                else:
                    wait = 2**attempt + random.random()
                    logger.warning("OpenAI error (%s). Retrying in %.1fs", e, wait)
                    time.sleep(wait)

        for idx, val in zip(batch_idx, fixed):
            parsed[idx] = val

    return parsed



_SYS_PROMPT_INFER = (
    "You are an assistant answering multiple‑select questions. "
    "Respond ONLY with capital letters (A‑H) that you think are correct, "
    "in alphabetical order, with no spaces. If unsure, guess."
)
_USER_TEMPLATE = "Select ALL correct answers.\n\n{question}\n\nAnswer:"


def _make_prompt(q: str) -> List[dict]:
    return [
        {"role": "system", "content": _SYS_PROMPT_INFER},
        {"role": "user", "content": _USER_TEMPLATE.format(question=q)},
    ]


# ---------------------------------------------------------------------
# 5 .  Public entry
# ---------------------------------------------------------------------
def run_inference(
    df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 128,
    debug: bool = False,
    seed: int = 42,
    save_jsonl: Optional[str] = None,
    repair_model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    debug : bool
        If True, skip model calls and fill raw predictions with random letters
        (useful for smoke testing the rest of the pipeline).
    """

    if "text" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must have 'text' and 'answer' columns")


    backend = OpenAIBackend(model, temperature, max_tokens)
    rng = random.Random(seed)

    raw_preds = []
    for q in df["text"]:
        if debug:
            raw = "".join(rng.sample(list("ABCDEFGH"), k=rng.randint(0, 3))) or "N/A"
        else:
            for attempt in range(5):
                try:
                    raw = backend(_make_prompt(q))

                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    wait = 2**attempt
                    logger.warning("Inference error (%s). Retrying in %ds", e, wait)
                    time.sleep(wait)
        raw_preds.append(raw)

    # Parse to canonical letters
    parsed = parse_function(raw_preds, repair_model=repair_model)

    out = pd.DataFrame(
        {
            "id": df.index,
            "prompt": df["text"],
            "answer": df["answer"].astype(str),
            "raw_prediction": raw_preds,
            "predicted_answer": parsed,
            "maximum": df["symbol"],
        }
    )
    
    if save_jsonl:
        out.to_json(save_jsonl, orient="records", lines=True)

    return out




