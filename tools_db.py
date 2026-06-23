"""
Gli Strumenti per il Database (Prenotazioni)
--------------------------------------------
Qui diamo al bot la capacità di "leggere" e "scrivere" nel nostro database delle prenotazioni (PostgreSQL).
Le operazioni e le query sono rigorosamente controllate per evitare che il bot possa fare danni.
"""

import re
import psycopg
from datetime import datetime, date
from typing import Optional
from langchain_core.tools import tool
from config import DB_URI


def _db_uri_ok() -> bool:
    return bool(DB_URI and str(DB_URI).strip())

def _sanitizza_nome(nome: str) -> str:
    """Rimuove caratteri speciali potenzialmente pericolosi e limita la lunghezza."""
    if not nome:
        return ""
    # Permette lettere, numeri (a volte presenti nei nomi), spazi, accenti base, apici e trattini
    pulito = re.sub(r'[^\w\s\-\'À-ÿ]', '', nome)
    return pulito.strip()[:100]

# ── SQL HARDCODED ────────────────────────────────────────────────────────────

_SQL_STANZA_DISPONIBILE = """
    SELECT numero_stanza, prezzo_base, tipologia
    FROM stanze
    WHERE tipologia ILIKE %s
    AND id NOT IN (
        SELECT stanza_id FROM prenotazioni
        WHERE check_in < %s::date AND check_out > %s::date
    )
    LIMIT 1
"""

_SQL_ALTERNATIVE_DISPONIBILI = """
    SELECT tipologia, prezzo_base
    FROM stanze
    WHERE id NOT IN (
        SELECT stanza_id FROM prenotazioni
        WHERE check_in < %s::date AND check_out > %s::date
    )
    GROUP BY tipologia, prezzo_base
    LIMIT 3
"""

_SQL_CHECK_OMONIMIA = """
    SELECT id FROM prenotazioni
    WHERE nome_cliente ILIKE %s
    AND check_in < %s::date AND check_out > %s::date
"""

_SQL_STANZA_PER_PRENOTAZIONE = """
    SELECT id, prezzo_base, numero_stanza FROM stanze
    WHERE tipologia ILIKE %s
    AND id NOT IN (
        SELECT stanza_id FROM prenotazioni
        WHERE check_in < %s::date AND check_out > %s::date
    )
    LIMIT 1
"""

_SQL_INSERISCI_PRENOTAZIONE = """
    INSERT INTO prenotazioni (stanza_id, nome_cliente, check_in, check_out, prezzo_totale, stato)
    VALUES (%s, %s, %s::date, %s::date, %s, 'Confermata')
    RETURNING id
"""

_SQL_CERCA_PRENOTAZIONI_NOME = """
    SELECT id, check_in, check_out, stato, stanza_id
    FROM prenotazioni
    WHERE nome_cliente ILIKE %s
"""

_SQL_CERCA_PRENOTAZIONI_NOME_DATE = """
    SELECT id, check_in, check_out
    FROM prenotazioni
    WHERE nome_cliente ILIKE %s
"""

# Modificato per mitigare IDOR: richiede sia ID che nome esatto
_SQL_CANCELLA_PER_ID = "DELETE FROM prenotazioni WHERE id = %s AND nome_cliente ILIKE %s RETURNING id"


# ── TOOL: verifica_disponibilita ─────────────────────────────────────────────

@tool
def verifica_disponibilita(check_in: str, check_out: str, tipologia: str = "Matrimoniale") -> str:
    """Verifica disponibilità stanza e calcola il preventivo.
    Chiamare SEMPRE prima di crea_prenotazione.
    NON chiede e NON raccoglie email, telefono o altri dati non elencati.

    Args:
        check_in: Data check-in YYYY-MM-DD
        check_out: Data check-out YYYY-MM-DD
        tipologia: Singola | Matrimoniale | Tripla | Quadrupla | Suite
    """
    if not _db_uri_ok():
        return "ERRORE DB: DATABASE_URI non configurata nel file .env"

    try:
        data_in = datetime.strptime(check_in, "%Y-%m-%d").date()
        data_out = datetime.strptime(check_out, "%Y-%m-%d").date()
        oggi = datetime.now().date()
        
        if data_in < oggi:
            return "ERRORE: Non è possibile verificare disponibilità per date passate."
            
        giorni = (data_out - data_in).days
        if giorni <= 0:
            return "ERRORE: La data di check-out deve essere successiva al check-in."
        if giorni > 30:
            return "ERRORE: Non è possibile prenotare per più di 30 notti consecutive."
    except ValueError:
         return "ERRORE: Formato data non valido. Usa YYYY-MM-DD."

    try:
        with psycopg.connect(DB_URI, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(_SQL_STANZA_DISPONIBILE, (f"%{tipologia}%", check_out, check_in))
                stanza = cur.fetchone()

                if stanza:
                    prezzo_totale = stanza[1] * giorni
                    return (
                        f"DISPONIBILE: Stanza {stanza[0]} ({stanza[2]}) libera. "
                        f"Prezzo totale per {giorni} notti: {prezzo_totale}€. "
                        "ISTRUZIONE: Comunica il preventivo all'ospite e chiedi ESPRESSAMENTE "
                        "se vuole confermare. NON usare crea_prenotazione finché l'ospite non conferma."
                    )

                cur.execute(_SQL_ALTERNATIVE_DISPONIBILI, (check_out, check_in))
                alternative = cur.fetchall()

                if alternative:
                    lista = "\n".join(f"- {a[0]} a {a[1] * giorni}€ totali" for a in alternative)
                    return (
                        f"ATTENZIONE: Nessuna '{tipologia}' disponibile per quelle date. "
                        f"Alternative disponibili:\n{lista}\n"
                        "ISTRUZIONE: Proponi le alternative all'ospite."
                    )
                return "TUTTO ESAURITO: Nessuna stanza disponibile in queste date."

    except Exception as e:
        return f"ERRORE DB: {e}"


# ── TOOL: crea_prenotazione ──────────────────────────────────────────────────

@tool
def crea_prenotazione(
    nome_cliente: str,
    check_in: str,
    check_out: str,
    tipologia: str = "Matrimoniale",
    conferma_multipla: bool = False,
) -> str:
    """Crea la prenotazione nel database. Usare SOLO dopo conferma esplicita dell'ospite.
    I soli dati richiesti sono: nome_cliente, check_in, check_out, tipologia.
    NON richiedere email, telefono, documento o altri dati non elencati.

    Args:
        nome_cliente: Nome e cognome del cliente
        check_in: Data check-in YYYY-MM-DD
        check_out: Data check-out YYYY-MM-DD
        tipologia: Singola | Matrimoniale | Tripla | Quadrupla | Suite
        conferma_multipla: True solo se l'ospite vuole una seconda stanza nonostante omonimia
    """
    if not _db_uri_ok():
        return "ERRORE DB: DATABASE_URI non configurata nel file .env"
        
    nome_pulito = _sanitizza_nome(nome_cliente)
    if len(nome_pulito) < 3:
        return "ERRORE: Nome cliente non valido o troppo corto."

    try:
        data_in = datetime.strptime(check_in, "%Y-%m-%d").date()
        data_out = datetime.strptime(check_out, "%Y-%m-%d").date()
        oggi = datetime.now().date()
        
        if data_in < oggi:
            return "ERRORE: Non è possibile creare prenotazioni per date passate."
            
        giorni = (data_out - data_in).days
        if giorni <= 0:
            return "ERRORE: La data di check-out deve essere successiva al check-in."
        if giorni > 30:
            return "ERRORE: Non è possibile prenotare per più di 30 notti consecutive."
    except ValueError:
         return "ERRORE: Formato data non valido. Usa YYYY-MM-DD."

    try:
        with psycopg.connect(DB_URI, autocommit=True) as conn:
            with conn.cursor() as cur:

                # Controllo omonimia
                if not conferma_multipla:
                    cur.execute(_SQL_CHECK_OMONIMIA, (f"%{nome_pulito}%", check_out, check_in))
                    esistenti = cur.fetchall()
                    if esistenti:
                        ids = [str(r[0]) for r in esistenti]
                        return (
                            f"ATTENZIONE: {nome_pulito} ha già prenotazioni attive "
                            f"(ID: {', '.join(ids)}). "
                            "Chiedere se desidera davvero una seconda stanza. "
                            "Se sì, richiamare con conferma_multipla=True."
                        )

                # Cerca stanza libera
                cur.execute(_SQL_STANZA_PER_PRENOTAZIONE, (f"%{tipologia}%", check_out, check_in))
                stanza = cur.fetchone()
                if not stanza:
                    return "ERRORE: Nessuna stanza disponibile. Riprovare con tipologia diversa."

                stanza_id, prezzo_base, numero_stanza = stanza
                prezzo_totale = giorni * prezzo_base

                cur.execute(
                    _SQL_INSERISCI_PRENOTAZIONE,
                    (stanza_id, nome_pulito, check_in, check_out, prezzo_totale),
                )
                nuovo_id = cur.fetchone()[0]

        return (
            f"SUCCESSO: Prenotazione ID {nuovo_id} creata per {nome_pulito}. "
            f"Stanza {numero_stanza} ({tipologia}), {giorni} notti, Totale: {prezzo_totale}€."
        )

    except Exception as e:
        return f"ERRORE DB: {e}"


# ── TOOL: trova_prenotazioni ─────────────────────────────────────────────────

@tool
def trova_prenotazioni(nome_cliente: str) -> str:
    """Cerca tutte le prenotazioni per nome cliente.
    Usare per visualizzare le prenotazioni o per trovare l'ID prima di cancellare.

    Args:
        nome_cliente: Nome del cliente (ricerca parziale, min 3 caratteri)
    """
    if not _db_uri_ok():
        return "ERRORE DB: DATABASE_URI non configurata nel file .env"
        
    nome_pulito = _sanitizza_nome(nome_cliente)
    if len(nome_pulito) < 3:
        return "ERRORE DI SICUREZZA: Fornire un nome cliente più specifico (almeno 3 caratteri alfanumerici)."

    try:
        with psycopg.connect(DB_URI, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(_SQL_CERCA_PRENOTAZIONI_NOME, (f"%{nome_pulito}%",))
                trovate = cur.fetchall()

        if not trovate:
            return f"Nessuna prenotazione trovata per '{nome_pulito}'."

        lista = "\n".join(
            f"- ID {p[0]}: dal {p[1]} al {p[2]} (Stato: {p[3]}, Stanza: {p[4]})"
            for p in trovate
        )
        return f"Trovate {len(trovate)} prenotazioni per {nome_pulito}:\n{lista}"

    except Exception as e:
        return f"ERRORE DB: {e}"


# ── TOOL: cancella_prenotazione ──────────────────────────────────────────────

@tool
def cancella_prenotazione(nome_cliente: str, id_prenotazione: Optional[int] = None, data_check_in_conferma: Optional[str] = None) -> str:
    """Cancella una prenotazione per ID.
    Richiede sempre il nome del cliente e (per sicurezza) la data di check_in oltre all'ID per cancellare.
    Se l'ID o la data non sono forniti, mostra le prenotazioni e chiede di domandare la data di check-in all'ospite.

    Args:
        nome_cliente: Nome del cliente (fondamentale per sicurezza)
        id_prenotazione: ID della prenotazione da cancellare
        data_check_in_conferma: Data di check-in originaria (es. 2024-12-25) per confermare l'identità
    """
    if not _db_uri_ok():
        return "ERRORE DB: DATABASE_URI non configurata nel file .env"
        
    nome_pulito = _sanitizza_nome(nome_cliente)
    if len(nome_pulito) < 3:
        return "ERRORE DI SICUREZZA: Nome cliente non valido o troppo corto."

    try:
        with psycopg.connect(DB_URI, autocommit=True) as conn:
            with conn.cursor() as cur:

                if id_prenotazione is None or data_check_in_conferma is None:
                    cur.execute(_SQL_CERCA_PRENOTAZIONI_NOME_DATE, (f"%{nome_pulito}%",))
                    trovate = cur.fetchall()

                    if not trovate:
                        return f"ERRORE: Nessuna prenotazione trovata per '{nome_pulito}'."

                    lista = "\n".join(
                        f"- ID {p[0]}: dal {p[1]} al {p[2]}" for p in trovate
                    )
                    return (
                        f"ATTENZIONE: Trovate prenotazioni per {nome_pulito}:\n{lista}\n"
                        "ISTRUZIONE: Chiedere all'ospite quale ID cancellare E la sua data di check-in per sicurezza. "
                        "Poi richiamare il tool passando sia id_prenotazione che data_check_in_conferma."
                    )

                # Verifica extra: assicurarsi che la data_check_in_conferma combaci con il record prima di cancellare
                cur.execute("SELECT check_in FROM prenotazioni WHERE id = %s", (id_prenotazione,))
                record = cur.fetchone()
                
                if not record:
                    return f"ERRORE: Prenotazione ID {id_prenotazione} non trovata."
                    
                data_db_str = record[0].strftime("%Y-%m-%d") if hasattr(record[0], 'strftime') else str(record[0])
                if data_check_in_conferma != data_db_str:
                    return "ERRORE DI SICUREZZA: La data di check-in fornita non corrisponde a quella della prenotazione. Cancellazione annullata."

                # Esecuzione cancellazione sicura (richiede anche nome_cliente nel WHERE)
                cur.execute(_SQL_CANCELLA_PER_ID, (id_prenotazione, f"%{nome_pulito}%"))
                cancellato = cur.fetchone()

        if cancellato:
            return f"SUCCESSO: Prenotazione ID {id_prenotazione} per {nome_pulito} cancellata correttamente."
        return f"ERRORE: Prenotazione ID {id_prenotazione} per il nome fornito non trovata (Possibile tentativo di IDOR bloccato)."

    except Exception as e:
        return f"ERRORE DB: {e}"


# Lista esportata per il bind in nodes.py
tools_hotel = [verifica_disponibilita, crea_prenotazione, trova_prenotazioni, cancella_prenotazione]