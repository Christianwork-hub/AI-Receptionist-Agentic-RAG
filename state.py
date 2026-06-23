"""
state.py — Stato LangGraph per AI Receptionist V3.

Architettura RAG Documentale Principale (Dominio Hotel):
  - Struttura dello stato per tracciare l'evoluzione della query: questionIsClear, originalQuery, rewrittenQuestions, agent_answers (gestito tramite logica di accumulate_or_reset).
  - Estensioni per routing multi-dominio (DB, WEB) e orchestrazione ciclica di tool integrati (gestionale).
"""

from typing import Annotated, List, Optional, Set, TypedDict
import operator

from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get("__reset__") for item in new):
        return []
    return (existing or []) + (new or [])


def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return (a or set()) | (b or set())


class ReceptionistState(MessagesState):
    """Stato unificato: messaggi + campi reference RAG + DB/Web."""

    # Default True: finché rewrite non imposta esplicitamente False, il grafo non va in HITL chiarimento
    questionIsClear: bool = True
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

    intento: str = "HOTEL"
    web_sub_intento: str = "SEARCH"

    query_riscritta: str = ""
    query_chiara: bool = True

    contesto: str = ""
    context_summary: str = ""

    web_results: str = ""
    web_source_urls: str = ""
    approvato: Optional[bool] = None
    web_rag_trovato: bool = False

    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
    retrieval_keys: Annotated[Set[str], set_union] = set()

    email_cliente: Optional[str] = None
