"""
utils.py — Funzioni di utilità per il progetto AI Receptionist.

Fornisce:
  - Stima conteggio token per gestione budget contesto
"""


def estimate_context_tokens(messages: list) -> int:
    """Stima il conteggio token per una lista di messaggi LangChain.

    Usa tiktoken (tokenizer GPT-4) se disponibile, altrimenti
    approssima con ~4 caratteri per token.

    Args:
        messages: Lista di oggetti messaggio con attributo 'content'

    Returns:
        Conteggio token approssimativo
    """
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return sum(
            len(encoding.encode(str(msg.content)))
            for msg in messages
            if hasattr(msg, "content") and msg.content
        )
    except ImportError:
        # Fallback: ~4 caratteri per token (approssimazione)
        return sum(
            len(str(msg.content)) // 4
            for msg in messages
            if hasattr(msg, "content") and msg.content
        )
