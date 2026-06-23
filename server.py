"""
File Principale: Server Web
---------------------------
Questo è il file che avvia l'intera applicazione. 
Crea un piccolo server web che ci permette di chattare con il bot tramite un'interfaccia grafica sul browser.
Basta lanciarlo con `python server.py` e andare su http://localhost:8000
"""

import contextlib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_core.messages import HumanMessage

# Importiamo dal nostro progetto esistente
from graph import app as langgraph_app, pool
from main import ensure_checkpoints_table, estrai_ultima_risposta_ai

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Appena avviamo il server, controlliamo che il database sia pronto a salvare le nostre chat
    ensure_checkpoints_table(pool)
    print("✅ Server FastAPI avviato. DB verificato.")
    yield
    print("🛑 Server in chiusura...")

app = FastAPI(lifespan=lifespan)

# Permettiamo le richieste CORS (utile se separi frontend e backend in futuro)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Prepariamo la memoria del bot per ricordarsi di questa specifica chat (tramite il thread_id)
    config = {"configurable": {"thread_id": req.thread_id}}
    
    stato = langgraph_app.get_state(config)
    
    # Se il bot era rimasto in attesa di una nostra risposta (ad es. per chiarimenti)
    if stato.next and ("request_clarification" in stato.next or "save_knowledge" in stato.next):
        # Gli passiamo il nostro messaggio e lo facciamo ripartire
        langgraph_app.update_state(config, {"messages": [HumanMessage(content=req.message)]})
        langgraph_app.invoke(None, config=config)
    else:
        # Esecuzione normale: inviamo il messaggio e lasciamo che il bot faccia il suo ragionamento
        langgraph_app.invoke({"messages": [HumanMessage(content=req.message)]}, config=config)
        
    # Una volta finito, recuperiamo la risposta finale che il bot ha preparato
    stato_finale = langgraph_app.get_state(config)
    risposta = estrai_ultima_risposta_ai(stato_finale)
    
    return ChatResponse(response=risposta)

# Colleghiamo la cartella "static" (dove c'è l'interfaccia del sito) all'indirizzo principale
# Così se apriamo localhost:8000 vediamo subito la chat!
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    # Facciamo partire il server. (Disabilitiamo il reload automatico perché 
    # il nostro database per i documenti preferisce gestire un solo processo alla volta)
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
