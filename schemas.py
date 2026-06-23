"""
schemas.py — Schemi Pydantic per output LLM strutturati.

Allineato alla reference: fino a 3 query riscritte (questions) + intento HOTEL/DB/WEB.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class QueryAnalysis(BaseModel):
    """Risultato dell'analisi query (rewrite + intento), come QueryAnalysis della reference."""

    model_config = ConfigDict(extra="ignore")

    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description=(
            "List of rewritten, self-contained questions suitable for retrieval "
            "(at most 3 items). For a single intent, use one element."
        ),
    )
    clarification_needed: str = Field(
        default="",
        description="Explanation if the question is unclear; empty if clear. MUST BE IN ITALIAN.",
    )
    intento: str = Field(
        description=(
            "Exactly one of: 'HOTEL' (policies/rules/services), "
            "'DB' (bookings/cancellations/availability), "
            "'WEB' (Matera/attractions/restaurants/external info)."
        ),
    )
