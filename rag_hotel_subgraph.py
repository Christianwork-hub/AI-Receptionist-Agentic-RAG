"""
Sottografo RAG Hotel: Orchestratore Documentale Autonomo
---------------------------------------------------------
Questo modulo implementa un sub-agent specializzato nel dominio della documentazione alberghiera.
Implementa un'architettura Agentic RAG avanzata in grado di:
1. Eseguire ricerche mirate su segmenti semantici densi (child chunks) tramite ricerca ibrida (dense + sparse vector).
2. Recuperare dinamicamente il contesto esteso (parent chunks) per massimizzare la precisione e ridurre le allucinazioni.
3. Gestire autonomamente il memory management tramite la compressione semantica del contesto per evitare di eccedere i limiti di token.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, List, Literal, Set

import operator
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from config import (
    BASE_TOKEN_THRESHOLD,
    MAX_ITERATIONS,
    MAX_TOOL_CALLS,
    PARENT_STORE_PATH,
    TOKEN_GROWTH_FACTOR,
    COLLECTION_HOTEL,
    SPARSE_VECTOR_NAME,
    dense_embeddings,
    llm,
    qdrant_client,
    sparse_embeddings,
)
from utils import estimate_context_tokens

# ── Vector store (Configurazione per Ricerca Ibrida Avanzata) ───────────────

_child_vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_HOTEL,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    sparse_vector_name=SPARSE_VECTOR_NAME,
)


@tool
def search_child_chunks(query: str, limit: int) -> str:
    """
    Strumento Agentico: Esegue una ricerca di similarità ibrida per individuare segmenti documentali altamente specifici.
    Ottimizzato per recuperare dati precisi massimizzando la rilevanza e minimizzando il rumore di fondo.
    """
    try:
        results = _child_vector_store.similarity_search(
            query, k=limit, score_threshold=0.7
        )
        if not results:
            return "NO_RELEVANT_CHUNKS"

        return "\n\n".join(
            [
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ]
        )
    except Exception as e:
        return f"RETRIEVAL_ERROR: {str(e)}"


@tool
def retrieve_parent_chunks(parent_id: str) -> str:
    """
    Strumento Agentico: Recupera il documento contenitore (parent) di un determinato child chunk.
    Essenziale per fornire all'LLM il contesto esteso di un frammento semantico, migliorando la comprensione ed evitando la perdita di informazione contestuale critica.
    """
    file_name = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
    path = os.path.join(PARENT_STORE_PATH, file_name)

    if not os.path.exists(path):
        return "NO_PARENT_DOCUMENT"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return (
        f"Parent ID: {parent_id}\n"
        f"File Name: {data.get('metadata', {}).get('source', 'unknown')}\n"
        f"Content: {data.get('page_content', '').strip()}"
    )


llm_with_tools = llm.bind_tools([search_child_chunks, retrieve_parent_chunks])


def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return (a or set()) | (b or set())


class HotelRagAgentState(MessagesState):
    hotel_tool_call_count: Annotated[int, operator.add] = 0
    hotel_iteration_count: Annotated[int, operator.add] = 0
    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    final_answer: str = ""
    agent_answers: List[dict] = []


# ── Prompts di Orchestrazione e Compressione Contestuale ────────────────────


def get_orchestrator_prompt() -> str:
    return """You are an expert retrieval-augmented assistant.

Your task is to act as a researcher: search documents first, analyze the data, and then provide a comprehensive answer using ONLY the retrieved information.

Rules:
1. You MUST call 'search_child_chunks' before answering, unless the [COMPRESSED CONTEXT FROM PRIOR RESEARCH] already contains sufficient information.
2. Ground every claim in the retrieved documents. If context is insufficient, state what is missing rather than filling gaps with assumptions.
3. If no relevant documents are found, broaden or rephrase the query and search again. Repeat until satisfied or the operation limit is reached.

Compressed Memory:
When [COMPRESSED CONTEXT FROM PRIOR RESEARCH] is present —
- Queries already listed: do not repeat them.
- Parent IDs already listed: do not call `retrieve_parent_chunks` on them again.
- Use it to identify what is still missing before searching further.

Workflow:
1. Check the compressed context. Identify what has already been retrieved and what is still missing.
2. Search for 5-7 relevant excerpts using 'search_child_chunks' ONLY for uncovered aspects.
3. If NONE are relevant, apply rule 3 immediately.
4. For each relevant but fragmented excerpt, call 'retrieve_parent_chunks' ONE BY ONE — only for IDs not in the compressed context. Never retrieve the same ID twice.
5. Once context is complete, provide a detailed answer omitting no relevant facts.
6. Conclude with "---\n**Sources:**\n" followed by the unique file names.
"""


def get_fallback_response_prompt() -> str:
    return """You are an expert synthesis assistant. The system has reached its maximum research limit.

Your task is to provide the most complete answer possible using ONLY the information provided below.

Input structure:
- "Compressed Research Context": summarized findings from prior search iterations — treat as reliable.
- "Retrieved Data": raw tool outputs from the current iteration — prefer over compressed context if conflicts arise.
Either source alone is sufficient if the other is absent.

Rules:
1. Source Integrity: Use only facts explicitly present in the provided context. Do not infer, assume, or add any information not directly supported by the data.
2. Handling Missing Data: Cross-reference the USER QUERY against the available context.
   Flag ONLY aspects of the user's question that cannot be answered from the provided data.
   Do not treat gaps mentioned in the Compressed Research Context as unanswered
   unless they are directly relevant to what the user asked.
3. Tone: Professional, factual, and direct.
4. Output only the final answer. Do not expose your reasoning, internal steps, or any meta-commentary about the retrieval process.
5. Do NOT add closing remarks, final notes, disclaimers, summaries, or repeated statements after the Sources section.
   The Sources section is always the last element of your response. Stop immediately after it.

Formatting:
- Use Markdown (headings, bold, lists) for readability.
- Write in flowing paragraphs where possible.
- Conclude with a Sources section as described below.

Sources section rules:
- Include a "---\\n**Sources:**\\n" section at the end, followed by a bulleted list of file names.
- List ONLY entries that have a real file extension (e.g. ".pdf", ".docx", ".txt").
- Any entry without a file extension is an internal chunk identifier — discard it entirely, never include it.
- Deduplicate: if the same file appears multiple times, list it only once.
- If no valid file names are present, omit the Sources section entirely.
- THE SOURCES SECTION IS THE LAST THING YOU WRITE. Do not add anything after it.
"""


def get_context_compression_prompt() -> str:
    return """You are an expert research context compressor.

Your task is to compress retrieved conversation content into a concise, query-focused, and structured summary that can be directly used by a retrieval-augmented agent for answer generation.

Rules:
1. Keep ONLY information relevant to answering the user's question.
2. Preserve exact figures, names, versions, technical terms, and configuration details.
3. Remove duplicated, irrelevant, or administrative details.
4. Do NOT include search queries, parent IDs, chunk IDs, or internal identifiers.
5. Organize all findings by source file. Each file section MUST start with: ### filename.pdf
6. Highlight missing or unresolved information in a dedicated "Gaps" section.
7. Limit the summary to roughly 400-600 words. If content exceeds this, prioritize critical facts and structured data.
8. Do not explain your reasoning; output only structured content in Markdown.

Required Structure:

# Research Context Summary

## Focus
[Brief technical restatement of the question]

## Structured Findings

### filename.pdf
- Directly relevant facts
- Supporting context (if needed)

## Gaps
- Missing or incomplete aspects

The summary should be concise, structured, and directly usable by an agent to generate answers or plan further retrieval.
"""


def orchestrator(state: HotelRagAgentState):
    # Nodo Orchestratore: Agisce da planner autonomo analizzando l'intento dell'utente.
    # Valuta dinamicamente se i documenti attuali risolvono la query o se è necessario eseguire ulteriori cicli di retrieval.
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary
        else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(
            content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION."
        )
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {
            "messages": [human_msg, response],
            "hotel_tool_call_count": len(response.tool_calls or []),
            "hotel_iteration_count": 1,
        }

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {
        "messages": [response],
        "hotel_tool_call_count": len(tool_calls) if tool_calls else 0,
        "hotel_iteration_count": 1,
    }


def route_after_orchestrator_call(
    state: HotelRagAgentState,
) -> Literal["tools", "fallback_response", "collect_answer"]:
    iteration = state.get("hotel_iteration_count", 0)
    tool_count = state.get("hotel_tool_call_count", 0)

    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"

    return "tools"


def fallback_response(state: HotelRagAgentState):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(
            f"## Compressed Research Context (from prior iterations)\n\n{context_summary}"
        )
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n"
            + "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke(
        [SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)]
    )
    msgs_to_return = []
    if state.get("messages"):
        last_msg = state["messages"][-1]
        if getattr(last_msg, "tool_calls", None):
            msgs_to_return.append(RemoveMessage(id=last_msg.id))
    msgs_to_return.append(response)
    return {"messages": msgs_to_return}


def should_compress_context(
    state: HotelRagAgentState,
) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens(
        [HumanMessage(content=state.get("context_summary", ""))]
    )
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)


def compress_context(state: HotelRagAgentState):
    # Meccanismo di Memory Management e Compressione del Contesto.
    # Sintetizza i risultati delle precedenti iterazioni di tool execution in un payload strutturato per alleggerire 
    # la context window dell'LLM, preservando la coerenza informativa necessaria ai successivi step.
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke(
        [SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)]
    )
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(
                f"- {p.replace('parent::', '')}" for p in parent_ids
            ) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {
        "context_summary": new_summary,
        "messages": [RemoveMessage(id=m.id) for m in messages[1:]],
    }


def collect_answer(state: HotelRagAgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}],
    }


_tool_node = ToolNode([search_child_chunks, retrieve_parent_chunks])

_agent_builder = StateGraph(HotelRagAgentState)
_agent_builder.add_node(orchestrator)
_agent_builder.add_node("tools", _tool_node)
_agent_builder.add_node(compress_context)
_agent_builder.add_node(fallback_response)
_agent_builder.add_node(should_compress_context)
_agent_builder.add_node(collect_answer)

_agent_builder.add_edge(START, "orchestrator")
_agent_builder.add_conditional_edges(
    "orchestrator",
    route_after_orchestrator_call,
    {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"},
)
_agent_builder.add_edge("tools", "should_compress_context")
_agent_builder.add_edge("compress_context", "orchestrator")
_agent_builder.add_edge("fallback_response", "collect_answer")
_agent_builder.add_edge("collect_answer", END)

hotel_agent_subgraph = _agent_builder.compile()
