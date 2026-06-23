import os
import uuid
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from graph import app
from config import MARKDOWN_DIR, qdrant_client, COLLECTION_HOTEL, COLLECTION_WEB
from parent_store_manager import ParentStoreManager

# --- CSS Personalizzato (Rich Dark Theme) ---
custom_css = """
    .gradio-container { 
        max-width: 1100px !important;
        background: #0f0f0f !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    .tabs { border-bottom: 1px solid #3f3f3f !important; }
    button[role="tab"][aria-selected="true"] {
        border-bottom: 2px solid #3b82f6 !important;
        color: #3b82f6 !important;
        background: transparent !important;
    }
    .chatbot { border-radius: 12px !important; background: #1a1a1a !important; }
    .message.user { background: #3b82f6 !important; color: white !important; }
    .message.bot { background: #262626 !important; color: #e5e5e5 !important; border: 1px solid #3f3f3f !important; }
    input, textarea { background: #1a1a1a !important; color: #e5e5e5 !important; border: 1px solid #3f3f3f !important; }
    .primary-btn { background: #3b82f6 !important; color: white !important; }
    .stop-btn { background: #ef4444 !important; color: white !important; }
    footer { display: none !important; }
"""

# --- Logica di Gestione Documenti ---
def get_file_list():
    if not os.path.exists(MARKDOWN_DIR):
        return "📭 Nessun documento presente."
    files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith(".md")]
    if not files:
        return "📭 Nessun documento presente."
    return "\n".join([f"📄 {f}" for f in files])

def clear_all_knowledge():
    # Pulisce Qdrant
    if qdrant_client.collection_exists(COLLECTION_HOTEL):
        qdrant_client.delete_collection(COLLECTION_HOTEL)
    if qdrant_client.collection_exists(COLLECTION_WEB):
        qdrant_client.delete_collection(COLLECTION_WEB)
    # Pulisce Parent Store
    store = ParentStoreManager()
    store.clear()
    # Pulisce cartella markdown
    if os.path.exists(MARKDOWN_DIR):
        for f in os.listdir(MARKDOWN_DIR):
            os.remove(os.path.join(MARKDOWN_DIR, f))
    return "🗑️ Base di conoscenza ripulita correttamente.", get_file_list()

# --- Logica Chat con LangGraph ---
def chat_handler(message, history, thread_id):
    """thread_id: gr.State — una sessione Gradio = un thread checkpoint (evita stato HITL corrotto)."""
    tid = thread_id if thread_id else str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}, "recursion_limit": 50}

    inputs = {"messages": [HumanMessage(content=message)]}

    state = app.get_state(config)

    if state.next:
        app.update_state(config, inputs)
        app.invoke(None, config=config)
    else:
        app.invoke(inputs, config=config)

    final_state = app.get_state(config)
    msgs = final_state.values.get("messages", [])

    for m in reversed(msgs):
        if isinstance(m, AIMessage) and m.content:
            return m.content, tid

    return "Mi scusi, non sono riuscito a generare una risposta.", tid

# --- Costruzione UI Gradio ---
with gr.Blocks(title="AI Receptionist - Grand Hotel Sassi") as demo:
    gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #3b82f6; margin-bottom: 0;">🏨 Grand Hotel Sassi di Matera</h1>
            <p style="color: #a3a3a3;">AI Receptionist — Assistente Virtuale di Concierge</p>
        </div>
    """)
    
    with gr.Tabs():
        # --- TAB CHAT ---
        with gr.Tab("💬 Reception Chat"):
            thread_state = gr.State(str(uuid.uuid4()))

            chatbot = gr.Chatbot(
                height=600,
                show_label=False,
                layout="bubble",
                placeholder="<strong>Benvenuto al Grand Hotel Sassi!</strong><br>Come posso aiutarla oggi? (Prenotazioni, Info Hotel, Consigli su Matera...)"
            )
            
            with gr.Row():
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Scrivi qui il tuo messaggio...",
                    scale=8
                )
                submit_btn = gr.Button("Invia", variant="primary", scale=1)

            def respond(message, chat_history, tid):
                bot_message, new_tid = chat_handler(message, chat_history, tid)
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": bot_message})
                return "", chat_history, new_tid

            submit_btn.click(respond, [txt, chatbot, thread_state], [txt, chatbot, thread_state])
            txt.submit(respond, [txt, chatbot, thread_state], [txt, chatbot, thread_state])

        # --- TAB DOCUMENTI ---
        with gr.Tab("📚 Gestione Conoscenza"):
            gr.Markdown("### Documenti Indicizzati")
            gr.Markdown("Visualizza i file Markdown attualmente presenti nella base di conoscenza dell'hotel.")
            
            file_display = gr.Textbox(
                value=get_file_list(),
                label="File in 'markdown_docs/'",
                lines=10,
                interactive=False
            )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Aggiorna Lista")
                clear_btn = gr.Button("🗑️ Svuota Base di Conoscenza", variant="stop")
            
            status_msg = gr.Markdown("")
            
            refresh_btn.click(get_file_list, None, file_display)
            clear_btn.click(clear_all_knowledge, None, [status_msg, file_display])

    gr.HTML("""
        <div style="text-align: center; color: #525252; font-size: 0.8em; margin-top: 20px;">
            Powered by LangGraph & DeepSeek v4-flash | Sviluppato per Stage Progetti AI
        </div>
    """)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False, css=custom_css)
