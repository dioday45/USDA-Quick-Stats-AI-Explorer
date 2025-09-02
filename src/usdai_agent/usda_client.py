from curses import meta
import httpx
from typing import Any, Dict, Tuple
import pandas as pd

USDA_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
DEFAULT_EXAMPLE_PARAMS = {
    "commodity_desc": "CORN",
    "statisticcat_desc": "YIELD",
    "unit_desc": "BU / ACRE",
    "agg_level_desc": "STATE",
    "state_alpha": "IA",
    "year": "2023",
}


class USDAClient:
    """Thin client for USDA Quick Stats API using httpx.

    Usage:
        client = USDAClient(api_key="...")
        df, meta = client.fetch(params={"commodity_desc": "CORN", ...})
    """

    def __init__(self, api_key: str, timeout: float = 60.0) -> None:
        if not api_key:
            raise ValueError("USDA API key is required")
        self.api_key = api_key
        self.timeout = timeout

    def check_connection(self) -> bool:
        """Check the connection to the USDA API."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(
                    USDA_BASE_URL,
                    params={
                        "key": self.api_key,
                        **DEFAULT_EXAMPLE_PARAMS,
                    },  # Small query to check
                )
                resp.raise_for_status()
                return True
        except Exception as e:
            print(f"Error connecting to USDA API: {e}")
            return False

    def fetch(
        self, params: Dict[str, Any] = DEFAULT_EXAMPLE_PARAMS
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch data from the USDA Quick Stats API.

        Args:
            params: Query parameters for the API request.

        Returns:
            A tuple containing a Polars DataFrame with the results and metadata.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(USDA_BASE_URL, params={"key": self.api_key, **params})
                resp.raise_for_status()
                data = resp.json()
                results = data.get("data")
                df = pd.DataFrame(results)
                return df
        except Exception as e:
            print(f"Error fetching USDA data: {e}")
            return pd.DataFrame(), {}
