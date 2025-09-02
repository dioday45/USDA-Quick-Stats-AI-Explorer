from __future__ import annotations
import json
import re
from typing import Dict, Any, List, Optional

from openai import OpenAI

ALLOWED_PARAMS: List[str] = [
    "commodity_desc",
    "class_desc",
    "prodn_practice_desc",
    "util_practice_desc",
    "statisticcat_desc",
    "unit_desc",
    "sector_desc",
    "group_desc",
    "domain_desc",
    "domaincat_desc",
    "agg_level_desc",
    "state_alpha",
    "state_name",
    "county_ansi",
    "county_name",
    "year",
    "source_desc",
    "freq_desc",
    "reference_period_desc",
    "short_desc",
]

STATE_NAME_TO_ALPHA = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
}

# A few explicit few-shots keep the model grounded
FEW_SHOT_EXAMPLES = [
    {
        "q": "Corn yield in Iowa for 2023",
        "params": {
            "commodity_desc": "CORN",
            "statisticcat_desc": "YIELD",
            "unit_desc": "BU / ACRE",
            "agg_level_desc": "STATE",
            "state_alpha": "IA",
            "year": "2023",
            "sector_desc": "CROPS",
        },
    },
    {
        "q": "Monthly US wheat prices received 2019",
        "params": {
            "commodity_desc": "WHEAT",
            "statisticcat_desc": "PRICE RECEIVED",
            "freq_desc": "MONTHLY",
            "agg_level_desc": "NATIONAL",
            "year": "2019",
            "sector_desc": "CROPS",
        },
    },
    {
        "q": "Soybean production by state in 2021 and 2022",
        "params": {
            "commodity_desc": "SOYBEANS",
            "statisticcat_desc": "PRODUCTION",
            "agg_level_desc": "STATE",
            "year": ["2021", "2022"],
            "sector_desc": "CROPS",
        },
    },
    {
        "q": "Corn yield in IA, IL and NE for 2020 to 2022",
        "params": {
            "commodity_desc": "CORN",
            "statisticcat_desc": "YIELD",
            "unit_desc": "BU / ACRE",
            "agg_level_desc": "STATE",
            "state_alpha": ["IA", "IL", "NE"],
            "year": ["2020", "2021", "2022"],
            "sector_desc": "CROPS",
        },
    },
]

SYSTEM_PROMPT = f"""
You convert a user's question into USDA Quick Stats API parameters.
Return ONLY a single JSON object using keys from this list: {ALLOWED_PARAMS}.
- Fields may be scalars or arrays. If the user clearly requests multiple items (e.g., multiple years or states), return a JSON array for that field (e.g., "year": ["2021","2022"]).
- Allowed multi-value fields include: year, state_alpha, state_name, county_ansi, county_name, class_desc, freq_desc (if truly multi), and short_desc.
Rules:
- Use UPPERCASE for commodity_desc and statisticcat_desc (e.g., CORN, WHEAT, YIELD).
- Prefer agg_level_desc STATE unless the user clearly asks for NATIONAL or COUNTY.
- If the United States as a whole is requested, set agg_level_desc to NATIONAL and omit state.
- If a state is mentioned, set state_alpha to its two-letter code when possible.
- Accept a single year if mentioned; if a range or multiple years are mentioned, return an array of years in ascending order.
- Only include unit_desc when obvious (e.g., YIELD for grains => BU / ACRE).
- For prices, map to statisticcat_desc = PRICE RECEIVED when appropriate.
- Do NOT guess counties.
- Be conservative — fewer parameters are better than wrong parameters.
Return JSON only. No prose. No code block fences.

Examples:
{json.dumps(FEW_SHOT_EXAMPLES, indent=2)}
""".strip()


def _parse_json_object(txt: str) -> Dict[str, Any]:
    txt = txt.strip()
    # direct parse
    try:
        return json.loads(txt)
    except Exception:
        pass
    # extract first {...}
    m = re.search(r"\{[\s\S]*\}", txt)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def _sanitize_params(d: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    if not isinstance(d, dict):
        return clean

    def norm_value(k: str, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if k in ("commodity_desc", "statisticcat_desc", "state_alpha"):
            s = s.upper()
        return s

    def norm_list(k: str, vals: Any) -> List[str]:
        if isinstance(vals, list):
            items = [norm_value(k, x) for x in vals]
        else:
            items = [norm_value(k, vals)]
        items = [x for x in items if x]
        # special handling
        if k == "state_name":
            up = [x.upper() for x in items]
            mapped = [STATE_NAME_TO_ALPHA[x] for x in up if x in STATE_NAME_TO_ALPHA]
            return list(dict.fromkeys(mapped))  # de-dupe, preserve order
        if k == "year":
            # keep as strings, ensure 4-digit
            items = [x for x in items if re.fullmatch(r"\d{4}", x)]
            # sort ascending, dedupe
            return sorted(list(dict.fromkeys(items)))
        # generic de-dupe
        return list(dict.fromkeys(items))

    for k, v in d.items():
        if k not in ALLOWED_PARAMS:
            continue
        if k == "state_name":
            vals = norm_list(k, v)
            if vals:
                clean["state_alpha"] = vals if len(vals) > 1 else vals[0]
            continue
        if k in (
            "year",
            "state_alpha",
            "class_desc",
            "freq_desc",
            "county_ansi",
            "county_name",
            "short_desc",
        ):
            vals = norm_list(k, v)
            if vals:
                clean[k] = vals if len(vals) > 1 else vals[0]
            continue
        nv = norm_value(k, v)
        if nv is not None:
            clean[k] = nv

    return clean


class ParamAgent:
    """
    LLM-backed mapper that turns a plain-English question into USDA Quick Stats params.
    Usage:
        agent = ParamAgent(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
        params = agent.generate("Corn yield in Iowa for 2023")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        timeout: float = 30.0,
    ):
        if not api_key:
            raise ValueError("OpenAI API key is required for ParamAgent.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = float(temperature)
        self.timeout = float(timeout)

    def generate(self, question: str) -> Dict[str, str]:
        """
        Returns a sanitized dict of USDA params. Empty dict on hard failure.
        """
        if not question or not question.strip():
            return {}
        user_prompt = f"User question: {question}\nReturn JSON only."
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=self.timeout,
            )
            content = resp.choices[0].message.content or ""
            raw = _parse_json_object(content)
            return _sanitize_params(raw)
        except Exception:
            return {}
