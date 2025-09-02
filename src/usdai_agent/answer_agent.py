from __future__ import annotations
import json
from typing import Any, Dict

import pandas as pd

try:
    from openai import OpenAI  # OpenAI SDK v1
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class AnswerAgent:
    """LLM-backed explainer that turns a question + params + data sample into a concise answer.

    Usage:
        agent = AnswerAgent(api_key="...", model="gpt-4o-mini", temperature=0.2)
        md = agent.generate(question, params, df)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("OpenAI API key is required for AnswerAgent.")
        if OpenAI is None:
            raise RuntimeError(
                "OpenAI SDK is not available. Install the 'openai' package."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = float(temperature)
        self.timeout = float(timeout)

    # ------------------------ public API ------------------------
    def generate(self, question: str, params: Dict[str, Any], df: pd.DataFrame) -> str:
        brief = self._build_data_brief(df, params)
        system_msg = (
            "You are an agricultural data analyst. Given a user question, the USDA Quick Stats parameters "
            "used, and a compact data excerpt+metrics, write a precise, concise answer in Markdown. "
            "Use only the provided data—do not invent values. Prefer one short paragraph and up to 4 bullets."
        )
        payload = {"question": question, "params": params, "data_brief": brief}
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                timeout=self.timeout,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": json.dumps(payload)},
                ],
            )
            return resp.choices[0].message.content or "(No content)"
        except Exception as e:  # surface clean error text to the UI
            return f"AI explanation error: {e}"

    # ------------------------ helpers ------------------------
    def _build_data_brief(
        self, df: pd.DataFrame, params: Dict[str, Any], max_rows: int = 12
    ) -> Dict[str, Any]:
        brief: Dict[str, Any] = {
            "rows": int(getattr(df, "shape", (0, 0))[0]),
            "cols": int(getattr(df, "shape", (0, 0))[1]),
            "columns": list(df.columns) if hasattr(df, "columns") else [],
            "params": params,
            "sample": [],
            "metrics": {},
        }
        # sample rows
        try:
            head = df.head(max_rows)
            brief["sample"] = head.to_dict(orient="records")
        except Exception:
            pass
        # numeric metrics on 'value'
        try:
            if "value" in df.columns:
                vals = df["value"].astype(str).str.replace(",", "").str.replace("$", "")
                vals_num = pd.to_numeric(vals, errors="coerce")
                if vals_num.notna().any():
                    brief["metrics"]["value_min"] = float(vals_num.min())
                    brief["metrics"]["value_max"] = float(vals_num.max())
                    brief["metrics"]["value_mean"] = float(vals_num.mean())
        except Exception:
            pass
        # distinct years/states
        try:
            if "year" in df.columns:
                brief["metrics"]["years_present"] = sorted(
                    list({str(y) for y in df["year"].dropna().astype(str)})
                )
            if "state_alpha" in df.columns:
                brief["metrics"]["states_present"] = sorted(
                    list({str(s) for s in df["state_alpha"].dropna().astype(str)})
                )
        except Exception:
            pass
        return brief
