from __future__ import annotations
import time

import streamlit as st
from src.usdai_agent.usda_client import USDAClient
from usdai_agent.param_agent import ParamAgent
from src.usdai_agent.answer_agent import AnswerAgent

PREFERRED = [
    "commodity_desc",
    "class_desc",
    "prodn_practice_desc",
    "util_practice_desc",
    "statisticcat_desc",
    "unit_desc",
    "sector_desc",
    "group_desc",
    "agg_level_desc",
    "state_alpha",
    "state_name",
    "county_name",
    "year",
    "freq_desc",
    "reference_period_desc",
    "short_desc",
    "domain_desc",
    "value",
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


def sanitize_params(d: dict) -> dict:
    if not isinstance(d, dict):
        return {}
    out = {k: v for k, v in d.items() if v not in (None, "", [], {})}
    if "state_name" in out:
        s = str(out.pop("state_name")).upper()
        if s in STATE_NAME_TO_ALPHA:
            out["state_alpha"] = STATE_NAME_TO_ALPHA[s]
    if "commodity_desc" in out:
        out["commodity_desc"] = str(out["commodity_desc"]).upper()
    if "statisticcat_desc" in out:
        out["statisticcat_desc"] = str(out["statisticcat_desc"]).upper()
    if "state_alpha" in out:
        out["state_alpha"] = str(out["state_alpha"]).upper()
    return out


# ---------- Page ----------
st.set_page_config(
    page_title="USDA Quick Stats AI Explorer", page_icon="🌾", layout="centered"
)

# ---------- Header ----------
st.markdown("# 🌾 USDA Quick Stats AI Explorer")

st.write(
    "Welcome to the **USDA Quick Stats AI Explorer** — a streamlined, AI‑powered interface to USDA agricultural data. "
    "Simply ask a question in plain English (e.g., *Corn yield in Iowa for 2023*), and the AI agent will translate it into the exact USDA Quick Stats parameters, retrieve the data, and generate a concise explanation."
)

st.markdown("### How it works")
st.markdown(
    "1. **Ask** a question in plain English.  \n"
    "2. The **AI Agent** interprets it and builds a validated JSON query.  \n"
    "3. The app **retrieves** the data directly from the USDA Quick Stats API.  \n"
    "4. The **Answer Agent** analyzes the dataset and returns a clear explanation, along with the raw data and parameters for full transparency."
)
st.divider()

# ---------- Sidebar config ----------
st.sidebar.header("Configuration")
usda_key = st.secrets.get("USDA_API_KEY", "")
req_timeout = st.sidebar.number_input(
    "Request timeout (seconds)", min_value=5, max_value=120, value=60
)
max_preview_rows = st.sidebar.number_input(
    "Maximum rows to preview", min_value=50, max_value=5000, value=1000, step=100
)

openai_key = st.secrets.get("OPENAI_API_KEY", "")
model = st.sidebar.selectbox("OpenAI model", options=["gpt-4o-mini", "gpt-4o"])
temperature = st.sidebar.slider(
    "Model temperature", min_value=0.0, max_value=1.0, value=0.1
)

# ---------- Query form ----------
with st.form("query_form", clear_on_submit=False):
    question = st.text_input(
        "Enter your question",
        value="What was the final and forecasted Corn yield in Iowa in 2023 ?",
        key="question",
    )
    col_submit, col_spacer, col_clear = st.columns([1, 0.001, 1], gap="small")
    with col_submit:
        submitted = st.form_submit_button("Submit", type="primary")
    with col_clear:
        cleared = st.form_submit_button("Clear")

# Handle Clear action
if "cleared" in locals() and cleared:
    # Reset user input and results
    st.session_state.pop("question", None)
    st.session_state.pop("last_params", None)
    st.session_state.pop("last_df", None)
    st.session_state.pop("last_explanation", None)
    st.session_state.pop("last_latency_ms", None)
    st.rerun()

if submitted:
    # Key checks
    if not usda_key:
        st.error(
            "No USDA API key provided. Please add `USDA_API_KEY` to `.streamlit/secrets.toml` or enter it in the sidebar."
        )
        st.stop()
    if not openai_key:
        st.error(
            "No OpenAI API key provided. Please add `OPENAI_API_KEY` to `.streamlit/secrets.toml` or enter it in the sidebar."
        )
        st.stop()
    if not ParamAgent:
        st.error(
            "Parameter Agent class not available. Ensure `src/usdai_agent/param_agent.py` exists."
        )
        st.stop()
    if not question.strip():
        st.error("Please enter a valid question.")
        st.stop()

    # Make status box expanded and start overall timer
    status_box = st.status("Starting...", expanded=False)
    t_overall = time.perf_counter()

    # Generate params with the agent (status + progress)
    progress = st.progress(0)
    try:
        t_param_start = time.perf_counter()
        agent = ParamAgent(api_key=openai_key, model=model, temperature=temperature)
        gen_raw = agent.generate(question)
        params = sanitize_params(gen_raw)
        t_param_end = time.perf_counter()
        dt_param_ms = int((t_param_end - t_param_start) * 1000)
        if not params:
            status_box.update(label="The agent returned no parameters.", state="error")
            st.error("The agent returned no parameters. Try a more specific question.")
            st.stop()
        status_box.update(label="Step 1/3: Building query parameters", state="running")
        status_box.write(
            f"**Step 1** · Parameter Agent completed in **{dt_param_ms} ms**."
        )
        progress.progress(33)
    except Exception as e:
        status_box.update(label=f"Agent error: {e}", state="error")
        st.error(f"Agent error: {e}")
        st.stop()

    # Fetch data (status + progress)
    try:
        status_box.update(
            label="Step 2/3: Retrieving data from USDA Quick Stats…", state="running"
        )
        client = USDAClient(usda_key, timeout=float(req_timeout))
        t_fetch_start = time.perf_counter()
        df = client.fetch(params)
        t_fetch_end = time.perf_counter()
        dt_fetch_ms = int((t_fetch_end - t_fetch_start) * 1000)
        status_box.update(label="Step 2/3: Data fetched", state="running")
        status_box.write(
            f"**Step 2** · USDA query completed in **{dt_fetch_ms} ms**. Rows returned: {getattr(df, 'shape', [0])[0]:,}."
        )
        progress.progress(66)
    except Exception as e:
        status_box.update(label=f"USDA API error: {e}", state="error")
        st.error(str(e))
        st.stop()

    # Generate explanation with AnswerAgent (step 3)
    try:
        status_box.update(
            label="Step 3/3: Analyzing results with Answer Agent…", state="running"
        )
        t_answer_start = time.perf_counter()
        expl_agent = AnswerAgent(
            api_key=openai_key, model=model, temperature=temperature
        )
        explanation_md = expl_agent.generate(
            st.session_state.get("question", ""), params, df
        )
        t_answer_end = time.perf_counter()
        dt_answer_ms = int((t_answer_end - t_answer_start) * 1000)
        progress.progress(100)
        status_box.update(label="Step 3/3: Explanation generated", state="running")
        status_box.write(
            f"**Step 3** · Answer Agent completed in **{dt_answer_ms} ms**."
        )
        total_ms = int((time.perf_counter() - t_overall) * 1000)
        status_box.update(
            label=f"Process complete — total time {total_ms} ms.",
            state="complete",
            expanded=False,
        )
    except Exception as e:
        explanation_md = f"AI explanation error: {e}"
        status_box.update(label=explanation_md, state="error")
    st.session_state["last_explanation"] = explanation_md

    # Persist for tabs
    st.session_state["last_params"] = params
    st.session_state["last_df"] = df
    st.session_state["last_latency_ms"] = dt_fetch_ms

# ---------- Results (if any) ----------
if "last_params" in st.session_state and "last_df" in st.session_state:
    params = st.session_state["last_params"]
    df = st.session_state["last_df"]
    latency_ms = st.session_state.get("last_latency_ms", 0)

    rows, cols = df.shape if hasattr(df, "shape") else (0, 0)

    tab_answer, tab_data, tab_params = st.tabs(["Answer", "Data", "Parameters"])

    with tab_answer:
        exp = st.session_state.get("last_explanation")
        if exp:
            st.markdown(exp)
        else:
            st.caption("Submit a query to generate an explanation.")

    with tab_params:
        st.json(params)
        with st.expander("Description of the parameters", expanded=False):
            st.markdown(
                "- `commodity_desc`: Commodity (e.g., CORN, WHEAT)  \n"
                "- `statisticcat_desc`: Metric (e.g., YIELD, PRODUCTION, PRICE RECEIVED)  \n"
                "- `agg_level_desc`: Geographic aggregation level (e.g., STATE, NATIONAL, COUNTY)  \n"
                "- `state_alpha`: Two‑letter state code (used for STATE‑level queries)  \n"
                "- `year`: Target year(s)  \n"
                "- `freq_desc`: Frequency, when applicable (e.g., MONTHLY)  \n"
                "- `unit_desc`: Units of measurement (e.g., BU / ACRE for grain yields)"
            )

    with tab_data:

        if rows == 0:
            st.warning("No results were returned for these parameters.")
        else:
            st.markdown(
                "These are the raw rows returned by USDA for the AI‑generated query parameters. Use the button below to download the full dataset as CSV."
            )
            cols_present = [c for c in PREFERRED if c in df.columns]
            others = [c for c in df.columns if c not in cols_present]
            df_show = df[cols_present + others] if cols_present else df
            st.dataframe(df_show.head(int(max_preview_rows)), width="stretch")
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full dataset (CSV)",
                data=csv_bytes,
                file_name="usda_quickstats_results.csv",
                mime="text/csv",
            )
