"""
Gli Strumenti per Internet (Tavily)
-----------------------------------
Se l'utente chiede informazioni "in tempo reale" o fuori dai documenti (es. "Che tempo fa a Matera?"), 
il bot usa questo file per collegarsi a internet e fare ricerche mirate e sicure.
"""

from tavily import TavilyClient
from config import TAVILY_API_KEY, llm
import re
from urllib.parse import urlparse

# Inizializza il client Tavily una sola volta al caricamento del modulo
_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

def _valida_pertinenza_web(query: str) -> bool:
    """Usa l'LLM per verificare se la query è pertinente al dominio turistico di Matera."""
    prompt = f"""Valuta se la seguente query è pertinente a un contesto turistico e di accoglienza per la città di Matera (o dintorni).
Devi accettare richieste su: meteo a Matera, ristoranti a Matera, trasporti, storia locale, musei, eventi a Matera/Basilicata.
Devi RIFIUTARE richieste su: ricette di cucina generali, città diverse da Matera (es. Milano, Roma, ecc.), programmazione, materie scolastiche, o cose non inerenti a un viaggio a Matera.

Rispondi SOLO con SI o NO.

Query: {query}
Risposta:"""
    try:
        risposta = llm.with_config(temperature=0.0).invoke(prompt).content.strip().upper()
        return "SI" in risposta
    except Exception as e:
        print(f"   ⚠️ [WEB VALIDATOR] Errore: {e}")
        return True # Fallback di sicurezza


def _verifica_url_sicuro(url: str) -> bool:
    """Evita l'estrazione da indirizzi IP privati o localhost."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if not netloc or "localhost" in netloc or netloc.startswith("127.") or netloc.startswith("192.168.") or netloc.startswith("10."):
            return False
        return True
    except:
        return False


def _formatta_risultati(response: dict, query: str) -> str:
    """Formatta i risultati di tavily.search() in testo strutturato per il contesto LLM."""
    parti = []

    # Risposta sintetizzata da Tavily (se disponibile)
    answer = response.get("answer", "")
    if answer:
        parti.append(f"📌 Risposta sintetizzata:\n{answer}")

    # Risultati individuali con fonte e contenuto
    results = response.get("results", [])
    for i, r in enumerate(results[:5], 1):
        title = r.get("title", "Senza titolo")
        url = r.get("url", "")
        content = r.get("content", "")
        if len(content) > 1500:
            content = content[:1500] + "..."
        parti.append(
            f"--- Fonte {i} ---\n"
            f"Titolo: {title}\n"
            f"URL: {url}\n"
            f"Contenuto:\n{content}"
        )

    if not parti:
        return f"Nessun risultato trovato per: '{query}'."

    return "\n\n".join(parti)


def esegui_tavily_search(query: str) -> str:
    """Ricerca web RAPIDA per domande semplici e dirette.

    Usa search_depth=basic per minimizzare latenza e costi.
    Ideale per: orari, indirizzi, info puntuali.
    """
    if not _client:
        return "ERRORE: TAVILY_API_KEY non configurata nel file .env"
        
    if not _valida_pertinenza_web(query):
        print(f"   🚫 [TAVILY SEARCH] Query bloccata dal validatore: '{query}'")
        return "RICHIESTA FUORI DOMINIO: Non ho il permesso di cercare sul web argomenti estranei a Matera, alla Basilicata o ai servizi turistico-alberghieri."

    query_geofenced = query
    q_lower = query.lower()
    if "matera" not in q_lower and "sassi" not in q_lower and "basilicata" not in q_lower:
        query_geofenced += " Matera"

    print(f"   🔍 [TAVILY SEARCH] Query originale: '{query}' -> Query inviata: '{query_geofenced}'")
    try:
        response = _client.search(
            query=query_geofenced,
            max_results=5,
            search_depth="basic",
            include_answer=True,
        )
        risultato = _formatta_risultati(response, query)
        n = len(response.get("results", []))
        print(f"   ✅ [TAVILY SEARCH] {n} risultati trovati.")
        return risultato
    except Exception as e:
        print(f"   ⚠️ [TAVILY SEARCH] Errore: {e}")
        return f"Errore nella ricerca Tavily: {e}"


def esegui_tavily_research(query: str) -> str:
    """Ricerca web APPROFONDITA per domande complesse e multi-sfaccettate.

    Usa search_depth=advanced con più risultati per copertura completa.
    Ideale per: itinerari, confronti, analisi dettagliate.
    """
    if not _client:
        return "ERRORE: TAVILY_API_KEY non configurata nel file .env"
        
    if not _valida_pertinenza_web(query):
        print(f"   🚫 [TAVILY RESEARCH] Query bloccata dal validatore: '{query}'")
        return "RICHIESTA FUORI DOMINIO: Non ho il permesso di cercare sul web argomenti estranei a Matera, alla Basilicata o ai servizi turistico-alberghieri."

    query_geofenced = query
    q_lower = query.lower()
    if "matera" not in q_lower and "sassi" not in q_lower and "basilicata" not in q_lower:
        query_geofenced += " Matera"

    print(f"   🔬 [TAVILY RESEARCH] Query originale: '{query}' -> Query inviata: '{query_geofenced}'")
    try:
        response = _client.search(
            query=query_geofenced,
            max_results=8,
            search_depth="advanced",
            include_answer=True,
        )
        risultato = _formatta_risultati(response, query)
        n = len(response.get("results", []))
        print(f"   ✅ [TAVILY RESEARCH] {n} risultati trovati.")
        return risultato
    except Exception as e:
        print(f"   ⚠️ [TAVILY RESEARCH] Errore: {e}")
        return f"Errore nella ricerca approfondita Tavily: {e}"


def esegui_tavily_extract(urls) -> str:
    """Estrae contenuto completo da URL specifici.

    Ideale per: estrarre info dettagliate da una pagina web specifica
    (es. menu ristorante, programma evento, orari museo).

    Args:
        urls: Singolo URL (str) o lista di URL
    """
    if not _client:
        return "ERRORE: TAVILY_API_KEY non configurata nel file .env"

    if isinstance(urls, str):
        urls = [urls]
        
    urls_sicuri = [u for u in urls if _verifica_url_sicuro(u)]
    if not urls_sicuri:
        return "ERRORE DI SICUREZZA: Nessun URL valido e sicuro fornito per l'estrazione."

    print(f"   🕷️ [TAVILY EXTRACT] Estrazione da {len(urls_sicuri)} URL (originali: {len(urls)})...")
    try:
        response = _client.extract(urls=urls_sicuri)

        parti = []
        results = response.get("results", [])
        for i, r in enumerate(results, 1):
            url = r.get("url", "")
            raw_content = r.get("raw_content", "")
            if len(raw_content) > 3000:
                raw_content = raw_content[:3000] + "..."
            parti.append(
                f"--- Contenuto da URL {i} ---\n"
                f"URL: {url}\n\n"
                f"{raw_content}"
            )

        if not parti:
            return "Nessun contenuto estratto dagli URL forniti."

        print(f"   ✅ [TAVILY EXTRACT] {len(results)} pagine estratte.")
        return "\n\n".join(parti)
    except Exception as e:
        print(f"   ⚠️ [TAVILY EXTRACT] Errore: {e}")
        return f"Errore nell'estrazione Tavily: {e}"