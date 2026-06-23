"""
Orchestrazione del Grafo LangGraph (Workflow Decisionale)
--------------------------------------------------------
Questo modulo definisce la topologia del workflow e le policy di routing dell'agente.
Implementa una logica decisionale avanzata che indirizza le richieste dell'utente verso
il sub-agent documentale (RAG Hotel), il sistema transazionale (DB Prenotazioni) o la ricerca esterna (WEB),
garantendo la corretta aggregazione del contesto.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from state import ReceptionistState 
from nodes import (
    summarize_history,
    rewrite_query,
    request_clarification,
    aggregate_answers,
    rag_db,
    rag_web_check,
    web_router,
    tavily_search_node,
    tavily_research_node,
    tavily_extract_node,
    save_knowledge,
    web_hitl_node,
    fallback_response,
    compress_context,
    postino,
    genera,
    tools_hotel,
)
from rag_hotel_subgraph import hotel_agent_subgraph
from utils import estimate_context_tokens
from config import MAX_TOOL_CALLS, MAX_ITERATIONS, BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR
from langchain_core.messages import HumanMessage

# ── Definizione della Topologia del Workflow ───────────────────────────────────

workflow = StateGraph(ReceptionistState)

# 1. Nodi iniziali: Comprensione dell'intento, query rewriting e gestione disambiguazione (HITL)
workflow.add_node("summarize_history", summarize_history)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("request_clarification", request_clarification)

# 2. Ramo HOTEL — Retrieval da base di conoscenza documentale (policy, orari, servizi)
workflow.add_node("agent", hotel_agent_subgraph)
workflow.add_node("aggregate_answers", aggregate_answers)

# 3. Ramo DATABASE — Esecuzione query strutturate e azioni su gestionali tramite Tool Node
workflow.add_node("rag_db", rag_db)
workflow.add_node("tools", ToolNode(tools_hotel))
workflow.add_node("fallback_response", fallback_response)
workflow.add_node("compress_context", compress_context)

# 4. Ramo WEB — Arricchimento contesto tramite ricerca internet (es. meteo, eventi)
workflow.add_node("rag_web_check", rag_web_check)
workflow.add_node("web_router", web_router)
workflow.add_node("tavily_search_node", tavily_search_node)
workflow.add_node("tavily_research_node", tavily_research_node)
workflow.add_node("tavily_extract_node", tavily_extract_node)
workflow.add_node("web_hitl_node", web_hitl_node)
workflow.add_node("save_knowledge", save_knowledge)

# 5. Nodi finali: Sintesi della risposta e delivery del payload
workflow.add_node("postino", postino)
workflow.add_node("genera", genera)

# ── Entry point ──────────────────────────────────────────────────────────────
workflow.add_edge(START, "summarize_history")
workflow.add_edge("summarize_history", "rewrite_query")


def route_after_rewrite(state: ReceptionistState):
    # Punto di routing principale post-rewriting.
    # Dirige il flusso basandosi sull'intento classificato o forza il blocco HITL in caso di ambiguità.
    if not state.get("questionIsClear", True):
        return "request_clarification"
    intento = (state.get("intento") or "HOTEL").strip().upper()
    if intento == "WEB":
        return "rag_web_check"
    if intento == "DB":
        return "rag_db"
    qs = state.get("rewrittenQuestions") or []
    if not qs:
        return "aggregate_answers"
    return [
        Send("agent", {"question": q, "question_index": idx, "messages": []})
        for idx, q in enumerate(qs)
    ]


workflow.add_conditional_edges(
    "rewrite_query",
    route_after_rewrite,
    {
        "request_clarification": "request_clarification",
        "rag_web_check": "rag_web_check",
        "rag_db": "rag_db",
        "aggregate_answers": "aggregate_answers",
    },
)
workflow.add_edge("request_clarification", "rewrite_query")
workflow.add_edge(["agent"], "aggregate_answers")
workflow.add_edge("aggregate_answers", END)

# ── Ramo DB — Tool loop ──────────────────────────────────────────────────────


def route_after_rag_db(state: ReceptionistState) -> str:
    # Valutazione iterativa post-esecuzione RAG DB:
    # Determina se l'LLM richiede ulteriori esecuzioni di tool o se il contesto è sufficiente.
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)
    last = state["messages"][-1] if state["messages"] else None
    has_tool_calls = last and getattr(last, "tool_calls", None)

    if not has_tool_calls:
        return "postino"
    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"
    return "tools"


workflow.add_conditional_edges(
    "rag_db",
    route_after_rag_db,
    {
        "tools": "tools",
        "fallback_response": "fallback_response",
        "postino": "postino",
    },
)


def route_after_tools(state: ReceptionistState) -> str:
    # Valutazione del carico contestuale (Memory Management).
    # Se la token window stimata supera la soglia consentita, instrada verso il nodo di compressione.
    msgs = state.get("messages", [])
    context_summary = state.get("context_summary", "")
    token_msgs = estimate_context_tokens(msgs)
    token_summary = estimate_context_tokens([HumanMessage(content=context_summary)]) if context_summary else 0
    totale = token_msgs + token_summary
    soglia = BASE_TOKEN_THRESHOLD + int(token_summary * TOKEN_GROWTH_FACTOR)
    return "compress_context" if totale > soglia else "rag_db"


workflow.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "compress_context": "compress_context",
        "rag_db": "rag_db",
    },
)

workflow.add_edge("compress_context", "rag_db")
workflow.add_edge("fallback_response", "postino")
workflow.add_edge("postino", "genera")
workflow.add_edge("genera", END)

# ── Ramo WEB ─────────────────────────────────────────────────────────────────


def route_after_web_check(state: ReceptionistState) -> str:
    return "genera" if state.get("web_rag_trovato", False) else "web_router"


workflow.add_conditional_edges(
    "rag_web_check",
    route_after_web_check,
    {"genera": "genera", "web_router": "web_router"},
)


def route_web_sub_intento(state: ReceptionistState) -> str:
    sub = state.get("web_sub_intento", "SEARCH")
    if sub == "RESEARCH":
        return "tavily_research_node"
    if sub == "EXTRACT":
        return "tavily_extract_node"
    return "tavily_search_node"


workflow.add_conditional_edges(
    "web_router",
    route_web_sub_intento,
    {
        "tavily_search_node": "tavily_search_node",
        "tavily_research_node": "tavily_research_node",
        "tavily_extract_node": "tavily_extract_node",
    },
)

workflow.add_edge("tavily_search_node", "save_knowledge")
workflow.add_edge("tavily_research_node", "save_knowledge")
workflow.add_edge("tavily_extract_node", "save_knowledge")
workflow.add_edge("save_knowledge", "genera")

# ── Persistenza e Checkpointing ────────────────────────────────────────────────
# Configurazione del database per il salvataggio dello stato di ogni esecuzione.
# Garantisce resilienza e persistenza della memoria conversazionale e dello stato del workflow.
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from config import DB_URI

connection_kwargs = {"autocommit": True, "prepare_threshold": None}
pool = ConnectionPool(conninfo=DB_URI, max_size=5, kwargs=connection_kwargs)
memory = PostgresSaver(pool)

try:
    memory.setup()
except Exception:
    pass

app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["request_clarification"],
)
