from __future__ import annotations
import openai
from typing import List, Optional, Union
from dataclasses import dataclass

import re

_OAI_SYS_PROMPT = (
    "You are an expert parser. For each line, extract ALL capital letters "
    "between A and H that represent choices. Respond with *one* "
    "<answer>XYZ</answer> per line in the same order. XYZ must list the "
    "letters alphabetically with no spaces. If no valid letters, return "
    "<answer></answer>."
)

@dataclass()
class ChoiceBatchParser:
    """End‑to‑end parser: numbered text → List[str] of cleaned answers."""
    model: str = "gpt-4o-mini"
    max_tokens: int = 120
    temperature: float = 0.0
    system_prompt: str = _OAI_SYS_PROMPT

    def __call__(self, numbered: str) -> List[str]:
        if not numbered.strip():
            return []

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"{numbered}\n\nParse each line as instructed.",
            },
        ]

        resp = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = resp.choices[0].message.content
        answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)

        n_lines = sum(1 for line in numbered.splitlines() if line.strip())
        cleaned: List[str] = []
        for ans in answers[:n_lines]:
            ans = "".join(sorted(set(ans.strip().upper())))
            cleaned.append(ans if re.fullmatch(r"[A-O]{0,15}", ans) else "")

        cleaned.extend([""] * (n_lines - len(cleaned)))  # pad if needed
        return cleaned
    

@dataclass
class OpenAIBackend:
    model: str
    temperature: float = 0.0
    max_tokens: int = 128
    timeout: int = 60

    def __post_init__(self):
        # openai.api_key must already be set
        pass

    def __call__(self, msgs: List[dict]) -> str:
        resp = openai.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content.strip()
