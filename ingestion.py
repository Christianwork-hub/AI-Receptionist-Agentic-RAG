"""
ingestion.py — Pipeline di ingestion: PDF → Markdown → Hybrid Search in Qdrant.


Pipeline:
  1. PDF → Markdown via opendataloader-pdf (preserva struttura header)
  2. Markdown → Parent chunks con chunking SEMANTICO (MarkdownHeaderTextSplitter + merge/split/clean)
  3. Parent → Child chunks (RecursiveCharacterTextSplitter per retrieval preciso)
  4. Child → Qdrant con ricerca ibrida (dense bge-m3 + sparse BM25)
  5. Parent → disco come file JSON individuali (un file per parent_id, es. regole_hotel_parent_0.json)

Crea DUE collection Qdrant:
  - conoscenza_hotel: statica, da PDF
  - conoscenza_web: dinamica, inizialmente vuota (si popola via HITL)

Garanzie:
  - Idempotente: rieseguire non crea duplicati
  - Mai sovrascrive collection già popolate
  - Batch loading per contenere RAM

Eseguire UNA VOLTA prima di avviare l'applicazione:
    python ingestion.py
"""

import os
import atexit
import logging

logging.getLogger("qdrant_client").setLevel(logging.CRITICAL)

def _chiudi_qdrant():
    try:
        from config import qdrant_client
        qdrant_client.close()
    except Exception:
        pass

atexit.register(_chiudi_qdrant)

from pathlib import Path
from qdrant_client.http import models
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from config import (
    qdrant_client, dense_embeddings, sparse_embeddings,
    COLLECTION_HOTEL, COLLECTION_WEB, SPARSE_VECTOR_NAME, EMBEDDING_DIM,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    MIN_PARENT_SIZE, MAX_PARENT_SIZE, HEADERS_TO_SPLIT_ON,
    MARKDOWN_DIR, PARENT_STORE_PATH,
)

import json


# ── Creazione Collection ────────────────────────────────────────────────────

def _crea_collection_ibrida(nome_collection: str):
    """Crea una collection Qdrant con supporto ricerca ibrida (dense + sparse).
    Se la collection esiste già, non fa nulla.
    """
    if not qdrant_client.collection_exists(nome_collection):
        print(f"📦 Creazione collection ibrida '{nome_collection}'...")
        qdrant_client.create_collection(
            collection_name=nome_collection,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams()
            },
        )
        print(f"   ✅ Collection '{nome_collection}' creata (dense + sparse).")
    else:
        print(f"   ✅ Collection '{nome_collection}' già esistente.")


def _collection_popolata(nome_collection: str) -> bool:
    """Ritorna True se la collection esiste e ha almeno un vettore."""
    if not qdrant_client.collection_exists(nome_collection):
        return False
    info = qdrant_client.get_collection(nome_collection)
    return (info.points_count or 0) > 0


def _get_vector_store(nome_collection: str) -> QdrantVectorStore:
    """Ritorna un QdrantVectorStore configurato per ricerca ibrida."""
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=nome_collection,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


# ── Conversione PDF → Markdown ──────────────────────────────────────────────

def _converti_pdf_a_markdown(pdf_path: str, output_dir: str):
    """Converte un PDF in Markdown via opendataloader-pdf."""
    import opendataloader_pdf

    os.makedirs(output_dir, exist_ok=True)
    print(f"📄 Conversione '{pdf_path}' → Markdown...")
    opendataloader_pdf.convert(
        input_path=[pdf_path],
        output_dir=output_dir,
        format="markdown",
    )
    print(f"   ✅ Conversione completata.")


# ── Chunking Semantico (fedele alla repo di riferimento) ───────────────────

def _merge_small_parents(chunks: list, min_size: int) -> list:
    """Unisce chunk piccoli al precedente finché non raggiungono min_size."""
    if not chunks:
        return []

    merged, current = [], None

    for chunk in chunks:
        if current is None:
            current = chunk
        else:
            current.page_content += "\n\n" + chunk.page_content
            for k, v in chunk.metadata.items():
                if k in current.metadata:
                    current.metadata[k] = f"{current.metadata[k]} -> {v}"
                else:
                    current.metadata[k] = v

        if len(current.page_content) >= min_size:
            merged.append(current)
            current = None

    if current:
        if merged:
            merged[-1].page_content += "\n\n" + current.page_content
            for k, v in current.metadata.items():
                if k in merged[-1].metadata:
                    merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                else:
                    merged[-1].metadata[k] = v
        else:
            merged.append(current)

    return merged


def _split_large_parents(chunks: list, max_size: int, child_overlap: int) -> list:
    """Spezza chunk troppo grandi con RecursiveCharacterTextSplitter."""
    split_chunks = []
    for chunk in chunks:
        if len(chunk.page_content) <= max_size:
            split_chunks.append(chunk)
        else:
            large_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_size,
                chunk_overlap=child_overlap,
            )
            sub_chunks = large_splitter.split_documents([chunk])
            split_chunks.extend(sub_chunks)
    return split_chunks


def _clean_small_chunks(chunks: list, min_size: int) -> list:
    """Post-processing: unisce residui troppo piccoli al vicino."""
    cleaned = []
    for i, chunk in enumerate(chunks):
        if len(chunk.page_content) < min_size:
            if cleaned:
                cleaned[-1].page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in cleaned[-1].metadata:
                        cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                    else:
                        cleaned[-1].metadata[k] = v
            elif i < len(chunks) - 1:
                chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                for k, v in chunk.metadata.items():
                    if k in chunks[i + 1].metadata:
                        chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                    else:
                        chunks[i + 1].metadata[k] = v
            else:
                cleaned.append(chunk)
        else:
            cleaned.append(chunk)
    return cleaned


def _splitta_markdown_in_parent(md_path: str) -> list:
    """Chunking semantico strutturato: MarkdownHeaderTextSplitter → merge → split_large → clean.

    Il processo garantisce la conservazione del contesto seguendo queste fasi:
      1. Split per header Markdown (#, ##, ###)
      2. Merge chunk piccoli (< MIN_PARENT_SIZE) con il precedente
      3. Split chunk grandi (> MAX_PARENT_SIZE)
      4. Post-process: unisci residui piccoli
    """
    with open(md_path, "r", encoding="utf-8") as f:
        testo = f.read()

    if not testo.strip():
        return []

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    chunks = header_splitter.split_text(testo)

    if not chunks:
        return []

    merged = _merge_small_parents(chunks, MIN_PARENT_SIZE)
    split = _split_large_parents(merged, MAX_PARENT_SIZE, CHILD_CHUNK_OVERLAP)
    cleaned = _clean_small_chunks(split, MIN_PARENT_SIZE)

    return cleaned


def _splitta_in_child(parent_doc) -> list:
    """Genera child chunks da un parent document per retrieval preciso."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )
    return splitter.split_documents([parent_doc])


# ── Parent Store su disco (file JSON individuali) ───────────────────────────

def _salva_parent_json(parent_id: str, page_content: str, metadata: dict):
    """Salva un parent chunk come file JSON individuale (es. regole_hotel_parent_0.json).

    Fedele alla repo di riferimento: un file per parent_id, NON un indice unico.
    """
    os.makedirs(PARENT_STORE_PATH, exist_ok=True)
    filepath = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
    doc_dict = {"page_content": page_content, "metadata": metadata}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, ensure_ascii=False, indent=2)


def _parent_json_exists(parent_id: str) -> bool:
    filepath = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
    return os.path.exists(filepath)


# ── Pipeline Principale ─────────────────────────────────────────────────────

def popola_conoscenza(pdf_path: str = "Regole_Hotel.pdf"):
    """Pipeline completa: PDF → Markdown → Parent/Child → Qdrant + Disco.

    1. Crea entrambe le collection (hotel + web)
    2. Converte il PDF in Markdown
    3. Splitta in parent chunks (semantico: header → merge → split → clean)
    4. Per ogni parent: genera child chunks con RecursiveCharacterTextSplitter
    5. Indicizza i child in Qdrant con ricerca ibrida
    6. Salva ogni parent su disco come file JSON individuale
    """
    print("=" * 60)
    print("🚀 Avvio ingestion con Hybrid Search + Chunking Semantico")
    print("=" * 60)

    _crea_collection_ibrida(COLLECTION_HOTEL)
    _crea_collection_ibrida(COLLECTION_WEB)

    if _collection_popolata(COLLECTION_HOTEL):
        print(f"⚠️  Collection '{COLLECTION_HOTEL}' già popolata. Nessuna sovrascrittura.")
        print("=" * 60)
        return

    md_dir = MARKDOWN_DIR
    os.makedirs(md_dir, exist_ok=True)
    pdf_stem = Path(pdf_path).stem
    md_path = os.path.join(md_dir, f"{pdf_stem}.md")

    if not os.path.exists(md_path):
        _converti_pdf_a_markdown(pdf_path, md_dir)
    else:
        print(f"   ✅ Markdown già presente: {md_path}")

    if not os.path.exists(md_path):
        md_files = list(Path(md_dir).glob("*.md"))
        if md_files:
            md_path = str(md_files[0])
            print(f"   📄 Trovato: {md_path}")
        else:
            print("❌ Nessun file Markdown generato.")
            return

    # Chunking semantico dei parent
    parent_chunks = _splitta_markdown_in_parent(md_path)
    print(f"   → {len(parent_chunks)} parent chunks generati.")

    vector_store = _get_vector_store(COLLECTION_HOTEL)
    totale_child = 0
    nuovi_parent = 0

    # Svuota il parent store esistente e ricrea (idempotente con collection nuova)
    os.makedirs(PARENT_STORE_PATH, exist_ok=True)
    for f in os.listdir(PARENT_STORE_PATH):
        if f.endswith(".json"):
            os.remove(os.path.join(PARENT_STORE_PATH, f))

    all_child_docs = []

    for i, parent_doc in enumerate(parent_chunks):
        parent_content = parent_doc.page_content.strip()
        if not parent_content:
            continue

        parent_id = f"{pdf_stem}_parent_{i}"

        # Aggiorna metadata del parent con parent_id e source
        parent_doc.metadata.update({
            "source": f"{pdf_stem}.pdf",
            "parent_id": parent_id,
            "parent_idx": i,
        })

        # Salva parent su disco come file JSON individuale
        _salva_parent_json(
            parent_id=parent_id,
            page_content=parent_content,
            metadata=parent_doc.metadata,
        )
        nuovi_parent += 1

        # Genera child chunks mantenendo i metadata del parent (incluso parent_id)
        child_docs = _splitta_in_child(parent_doc)
        # Assicura che ogni child abbia il parent_id nel metadata
        for cd in child_docs:
            cd.metadata["parent_id"] = parent_id
            cd.metadata["source"] = f"{pdf_stem}.pdf"
            cd.metadata["categoria"] = "policy"

        all_child_docs.extend(child_docs)
        totale_child += len(child_docs)
        print(f"   Parent {i + 1}/{len(parent_chunks)}: {len(child_docs)} child chunks generati.")

    if all_child_docs:
        vector_store.add_documents(all_child_docs)

    print("=" * 60)
    print(f"✅ Ingestion completata.")
    print(f"   • {nuovi_parent} parent chunks salvati su disco (JSON individuali).")
    print(f"   • {totale_child} child chunks indicizzati in Qdrant (hybrid).")
    print(f"   • Collection '{COLLECTION_WEB}' pronta (vuota, si popola via HITL).")
    print("=" * 60)


if __name__ == "__main__":
    popola_conoscenza()