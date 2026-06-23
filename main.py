"""
File Principale da Terminale
----------------------------
Questo file serve per avviare il bot dal "Terminale" (schermo nero) invece che dal sito web.
È molto utile per noi sviluppatori per fare test veloci. 
"""

import logging
import atexit
from langchain_core.messages import HumanMessage
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from config import DB_URI, qdrant_client
from graph import workflow

# Silenzia log verbosi
logging.getLogger("qdrant_client").setLevel(logging.CRITICAL)


def _chiudi_qdrant():
    try:
        qdrant_client.close()
    except Exception:
        pass

atexit.register(_chiudi_qdrant)


def estrai_ultima_risposta_ai(stato) -> str:
    """Cerca tra tutti i messaggi della chat e ci restituisce l'ultima cosa che ha detto il bot."""
    messaggi = stato.values.get("messages", [])
    for msg in reversed(messaggi):
        if hasattr(msg, "type") and msg.type == "ai" and getattr(msg, "content", ""):
            return msg.content
        if isinstance(msg, dict) and msg.get("type") == "ai":
            return msg.get("content", "")
    return "(Nessuna risposta)"


def ensure_checkpoints_table(pool: ConnectionPool) -> None:
    """Controlla se nel database esiste già il 'registro' delle conversazioni passate. Se non c'è, lo crea da zero."""
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = %s AND table_schema = current_schema();",
                ("checkpoints",),
            )
            if cur.fetchone() is None:
                print("⚙️  Tabella 'checkpoints' mancante, la creo...")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint JSONB NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}',
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    );
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx "
                    "ON checkpoints(thread_id);"
                )
                print("✅ Tabella 'checkpoints' creata.")
            else:
                print("✅ Tabella 'checkpoints' già presente.")


def _gestisci_hitl_clarification(app, config: dict) -> bool:
    """
    Gestisce i momenti in cui il bot si ferma perché non ha capito la richiesta
    e ha bisogno che noi gli chiariamo la situazione (scrivendo dal terminale).
    """
    stato = app.get_state(config)
    if not (stato.next and "request_clarification" in stato.next):
        return False

    print("\n" + "─" * 45)
    print("❓  CHIARIMENTO RICHIESTO")
    # L'ultimo messaggio AI contiene già la domanda di chiarimento
    msgs = stato.values.get("messages", [])
    for m in reversed(msgs):
        if hasattr(m, "type") and m.type == "ai" and m.content:
            print(f"   AI: {m.content}")
            break
    print("─" * 45)

    try:
        risposta_ospite = input("👤 Ospite (chiarimento): ").strip()
    except (EOFError, KeyboardInterrupt):
        return False

    if not risposta_ospite:
        return False

    # Aggiorna lo stato con la risposta dell'ospite e riprendi
    app.update_state(config, {"messages": [HumanMessage(content=risposta_ospite)]})
    return True


def esegui_app():
    # Un piccolo messaggio di benvenuto grafico nel terminale
    print("\n" + "=" * 55)
    print("  🏨  Grand Hotel Sassi — AI Receptionist V3  ")
    print("=" * 55)

    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": None,
    }

    try:
        # Apriamo la connessione con il database PostgreSQL
        with ConnectionPool(conninfo=DB_URI, max_size=10, kwargs=connection_kwargs) as pool:
            print("⚙️  Connessione al database...")
            # memory è la "scatola nera" dove il bot salverà le conversazioni per ricordarsele dopo
            memory = PostgresSaver(pool)

            try:
                memory.setup()
                ensure_checkpoints_table(pool)
                print("✅ Checkpointer LangGraph pronto.")
            except Exception as e:
                print(f"❌ ERRORE CRITICO nel setup delle tabelle: {e}")
                print("Assicurati che l'utente del DB abbia i permessi di CREATE TABLE.")
                return

            # Compila il grafo (solo interrupt su chiarimento query; salvataggio web automatico)
            app = workflow.compile(
                checkpointer=memory,
                interrupt_before=["request_clarification"],
            )

            print("\n💡 Inserisci un Thread ID (es. 'ospite_1') per iniziare.")
            thread_id = input("🔧 Thread ID: ").strip() or "default"
            config = {"configurable": {"thread_id": thread_id}}

            print(f"\n✅ Sessione '{thread_id}' avviata. Scrivi 'esci' per terminare.\n")

            while True:
                try:
                    domanda = input("👤 Ospite: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not domanda:
                    continue
                if domanda.lower() in ("esci", "q", "exit"):
                    break

                print("⚙️  Elaborazione...")

                try:
                    # Prima invocazione
                    app.invoke(
                        {"messages": [HumanMessage(content=domanda)]},
                        config=config,
                    )

                    # ── Gestione HITL request_clarification ───────────────
                    while _gestisci_hitl_clarification(app, config):
                        app.invoke(None, config=config)

                    # Risposta finale
                    stato_finale = app.get_state(config)
                    risposta = estrai_ultima_risposta_ai(stato_finale)
                    print(f"\n🤖 [Receptionist]: {risposta}\n")

                except Exception as e:
                    print(f"❌ Errore durante l'elaborazione: {e}")
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"💥 Errore fatale database: {e}")


if __name__ == "__main__":
    esegui_app()