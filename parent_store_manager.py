

"""
parent_store_manager.py — Gestione parent chunks su disco.

I parent chunks sono contesti ampi (2000+ caratteri) che forniscono
contesto completo per la generazione della risposta.
I child chunks (in Qdrant) servono per il retrieval preciso;
una volta trovati, si recuperano i parent corrispondenti da disco.

Pattern: Parent Document Retriever
  1. Cerca child chunks in Qdrant (piccoli, precisi)
  2. Ottieni i parent_id dai metadata dei child
  3. Carica i parent chunks da disco (ampi, completi)
"""

import json
from pathlib import Path
from config import PARENT_STORE_PATH


class ParentStoreManager:
    """Gestisce il salvataggio e caricamento dei parent chunks su disco (JSON)."""

    def __init__(self, store_path: str = PARENT_STORE_PATH):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._index_file = self.store_path / "parent_index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict:
        """Carica l'indice dei parent chunks da disco."""
        if self._index_file.exists():
            with open(self._index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Persiste l'indice su disco."""
        with open(self._index_file, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)

    def save_parent(self, parent_id: str, content: str, metadata: dict = None):
        """Salva un singolo parent chunk su disco."""
        self._index[parent_id] = {
            "content": content,
            "metadata": metadata or {},
        }
        self._save_index()

    def save_parents_batch(self, parents: list):
        """Salva multipli parent chunks in batch.

        Args:
            parents: Lista di tuple (parent_id, content, metadata)
        """
        for parent_id, content, metadata in parents:
            self._index[parent_id] = {
                "content": content,
                "metadata": metadata or {},
            }
        self._save_index()

    def load_parent(self, parent_id: str):
        """Carica un parent chunk dal disco. Ritorna None se non trovato."""
        entry = self._index.get(parent_id)
        return entry["content"] if entry else None

    def load_parents(self, parent_ids: list) -> list:
        """Carica multipli parent chunks. Ritorna solo quelli trovati."""
        results = []
        for pid in parent_ids:
            content = self.load_parent(pid)
            if content:
                results.append(content)
        return results

    def exists(self, parent_id: str) -> bool:
        """Verifica se un parent chunk esiste nello store."""
        return parent_id in self._index

    def count(self) -> int:
        """Ritorna il numero di parent chunks nello store."""
        return len(self._index)

    def clear(self):
        """Svuota completamente lo store: elimina l'indice e tutti i file JSON individuali."""
        # Rimuovi tutti i file .json nella cartella (parent individuali + indice)
        for f in self.store_path.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass
        # Reset indice in memoria
        self._index = {}
        print("   [ParentStoreManager] Store svuotato.")

    def clear(self):
        """Svuota completamente lo store: elimina l'indice e tutti i file JSON individuali."""
        # Rimuovi tutti i file .json nella cartella (parent individuali + indice)
        for f in self.store_path.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass
        # Reset indice in memoria
        self._index = {}
        print("   [ParentStoreManager] Store svuotato.")