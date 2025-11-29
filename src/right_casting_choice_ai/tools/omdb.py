from __future__ import annotations

import os
import re
import time
from typing import Optional, Type

import requests
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class OmdbToolInput(BaseModel):
    """Input schema for OMDb lookups."""

    title: str = Field(..., description="Exact movie title to search on OMDb.")
    year: Optional[str] = Field(
        default=None, description="Optional release year to disambiguate results."
    )


class OmdbTool(BaseTool):
    name: str = "omdb_lookup"
    description: str = (
        "Lookup a movie by title (and optional year) on OMDb, returning key details "
        "like BoxOffice, Budget, Ratings, and INR-converted amounts."
    )
    args_schema: Type[BaseModel] = OmdbToolInput
    # Declare as model fields for Pydantic
    api_key: Optional[str] = None
    usd_to_inr_rate: float = 83.0

    def __init__(self, api_key: Optional[str] = None, usd_to_inr_rate: Optional[float] = None):
        super().__init__()
        # Resolve API key (param overrides env if provided)
        self.api_key = api_key or os.getenv("OMDB_API_KEY", "")
        # Resolve conversion rate
        if usd_to_inr_rate is None:
            try:
                usd_to_inr_rate = float(os.getenv("USD_TO_INR_RATE", "83.0"))
            except ValueError:
                usd_to_inr_rate = 83.0
        self.usd_to_inr_rate = usd_to_inr_rate

    def _run(self, title: str, year: Optional[str] = None) -> str:
        if not self.api_key:
            return (
                "OMDb API key missing. Set OMDB_API_KEY in environment to enable lookups."
            )

        params = {"t": title, "plot": "short", "r": "json", "apikey": self.api_key}
        if year:
            params["y"] = year

        try:
            resp = requests.get("https://www.omdbapi.com/", params=params, timeout=20)
            if resp.status_code == 401:
                return "OMDb authentication failed (401). Check OMDB_API_KEY."
            if resp.status_code == 429:
                time.sleep(1.5)
                resp = requests.get("https://www.omdbapi.com/", params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return f"OMDb request error: {e}"

        if not data or data.get("Response") == "False":
            return f"OMDb: no result for '{title}'{f' ({year})' if year else ''}."

        # Normalize monetary fields and compute INR conversions
        box_office_usd = self._parse_usd(data.get("BoxOffice"))
        budget_usd = self._parse_usd(data.get("Budget"))
        box_office_inr = int(box_office_usd * self.usd_to_inr_rate) if box_office_usd is not None else None
        budget_inr = int(budget_usd * self.usd_to_inr_rate) if budget_usd is not None else None

        payload = {
                "Title": data.get("Title") or title,
                "Year": data.get("Year") or "",
                "Actors": data.get("Actors") or "",
                "Poster": data.get("Poster") if data.get("Poster") and data.get("Poster") != "N/A" else "",
                "BoxOfficeINR": box_office_inr,
                "BudgetINR": budget_inr,
                "imdbRating": data.get("imdbRating") or "",
                "imdbID": data.get("imdbID") or "",
                # keep raw for debugging if needed
                "_raw": data,
        }

        # Tools return strings; agents can parse JSON if needed
        try:
            import json

            return json.dumps(payload)
        except Exception:
            return str(payload)

    @staticmethod
    def _parse_usd(value: Optional[str]) -> Optional[int]:
        """Parse strings like "$123,456,789" to integer USD amount. Returns None if N/A."""
        if not value or value == "N/A":
            return None
        # Extract digits
        digits = re.sub(r"[^0-9]", "", value)
        if not digits:
            return None
        try:
            return int(digits)
        except ValueError:
            return None
