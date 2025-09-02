

# 🌾 USDA Quick Stats AI Explorer

An AI-powered Streamlit application that allows users to query the USDA Quick Stats API in plain English.  
The app automatically translates user questions into USDA API parameters, retrieves the data, and generates a clear explanation of the results.

---

## Features

- **Natural language input**: Ask questions in plain English, e.g. *“Corn yield in Iowa for 2023”*.
- **Parameter Agent**: Uses an LLM to convert your question into valid USDA Quick Stats parameters.
- **Data retrieval**: Fetches results directly from the USDA Quick Stats API.
- **Answer Agent**: Analyzes the dataset and produces a concise, human-readable summary.
- **Transparent pipeline**: Parameters, raw data, and explanations are all displayed for review.
- **Progress tracking**: Three-step progress bar with timings for parameter generation, data retrieval, and explanation.

---

## How it works

1. **Enter a question** in plain English.
2. The **Parameter Agent** generates valid USDA query parameters.
3. The app **fetches data** from the USDA Quick Stats API.
4. The **Answer Agent** analyzes the results and generates an explanation.
5. Review outputs in three tabs:
   - **Answer**: AI-generated narrative.
   - **Data**: Raw USDA dataset with CSV download option.
   - **Parameters**: The exact USDA API filters used.

---

## Tech stack

- [Streamlit](https://streamlit.io) – frontend and app framework.
- [httpx](https://www.python-httpx.org) – async HTTP client.
- [OpenAI](https://platform.openai.com/) – LLM-powered agents.
- USDA Quick Stats API.

---

## Setup

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/usdai_agent.git
   cd usdai_agent
   ```

2. **Install dependencies** (using Poetry):
   ```bash
   poetry install
   ```

3. **Set up secrets**: Create a `.streamlit/secrets.toml` file with:
   ```toml
   USDA_API_KEY = "your_usda_api_key"
   OPENAI_API_KEY = "your_openai_api_key"
   ```

4. **Run the app**:
   ```bash
   poetry run streamlit run main.py
   ```

---

## Example queries

- *Corn yield in Iowa for 2023*  
- *Monthly US wheat prices received in 2019*  
- *Soybean production by state in 2021*  
- *US corn yield from 2012 to 2020*  