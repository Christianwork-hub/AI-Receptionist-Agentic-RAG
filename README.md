# 🏨 Grand Hotel Sassi - AI Receptionist V3 (Agentic RAG)

![Presentazione Chatbot](docs/Immagini_Test_Chatbot/presentazione%20chatbot.png)

Benvenuto nella repository del **Grand Hotel Sassi AI Receptionist**, un assistente virtuale di livello Enterprise costruito su un'architettura **Agentic RAG (Retrieval-Augmented Generation)** basata su LangGraph.

Questo progetto non è un semplice chatbot basato su prompt, ma un vero e proprio **agente autonomo** capace di ragionare, decidere quale strada intraprendere, usare strumenti complessi (come database e motori di ricerca) e chiedere l'intervento umano se necessario.

---

## 🗂️ Struttura del Progetto

Ecco come è strutturato il codice sorgente e dove si trova la logica. Ogni file ha uno scopo preciso per mantenere l'architettura pulita e scalabile.

```text
AI_Receptionist_V3/
│
├── server.py                 # (Entry Point Web) Avvia FastAPI, espone l'API /api/chat e serve la UI HTML.
├── main.py                   # (Entry Point CLI) Avvia il bot nel terminale per fare test rapidi come sviluppatore.
├── app_gradio.py             # Interfaccia grafica alternativa usando Gradio.
│
├── graph.py                  # (Il Cervello Principale) Costruisce la "mappa" di LangGraph. Definisce i nodi e i "router" condizionali.
├── nodes.py                  # Definisce le funzioni (i "nodi" della mappa) eseguiti da graph.py.
├── state.py                  # Definisce lo "Stato" (la memoria a breve termine) che viaggia tra i nodi del grafo.
├── config.py                 # File di configurazione centrale (API keys, scelta dell'LLM, impostazioni vettoriali).
├── schemas.py                # Definisce i modelli dati (es. la struttura per estrarre l'intento dell'utente).
│
├── rag_hotel_subgraph.py     # Sottografo dedicato ESCLUSIVAMENTE alla lettura dei documenti dell'hotel (Parent-Child RAG).
├── ingestion.py              # Script per caricare i PDF/Testi dentro il database vettoriale Qdrant.
├── parent_store_manager.py   # Gestisce i "Parent Chunks" (i documenti originali grandi) per il RAG.
│
├── tools_db.py               # Gli Strumenti Database: funzioni SQL sicure per verificare, creare e cancellare prenotazioni.
├── tools_web.py              # Gli Strumenti Web: funzioni che interrogano Tavily API per le ricerche su Matera.
├── utils_email.py            # Modulo che genera e invia le email HTML di conferma/cancellazione prenotazione.
├── utils.py                  # Funzioni di utilità generale (es. stima dei token per la compressione della memoria).
│
├── .env.example              # Template delle variabili d'ambiente (il vero .env viene ignorato da Git).
├── .gitignore                # File che impedisce l'upload su GitHub di dati sensibili e file temporanei.
└── requirements.txt          # Tutte le dipendenze Python necessarie per avviare il progetto.
```

---

## 🧠 L'Architettura: Come Funziona il Flusso Logico

Il cuore del progetto è in **`graph.py`**. LangGraph usa un approccio a "stati" e "nodi". Quando un utente scrive un messaggio, ecco cosa succede passo dopo passo:

### 1. Inizializzazione e Comprensione (`nodes.py` e `graph.py`)
1. **`summarize_history`**: Se la conversazione è molto lunga, il bot fa un piccolo riassunto interno per non sovraccaricare la memoria dell'LLM.
2. **`rewrite_query`**: L'LLM analizza il messaggio dell'utente e cerca di capire qual è il suo **Intento** principale (HOTEL, DB, WEB). Se la domanda è posta male o manca di contesto, la riscrive in modo ottimizzato per la ricerca. Se la domanda è incomprensibile, imposta una flag per chiedere aiuto (HITL).
3. **Router Principale (`route_after_rewrite`)**: A questo punto, il codice controlla l'intento e "dirotta" l'esecuzione verso uno dei tre rami principali.

### 2. I Tre Rami Operativi

#### Ramo 1: RAG Documentale (Intento: `HOTEL`) - *Script: `rag_hotel_subgraph.py`*
Se l'utente chiede "A che ora è la colazione?" o "Posso portare il cane?".
- Il grafo rimanda a un **sottografo** specializzato. 
- L'LLM attiva un tool (`search_child_chunks`) per cercare nel database vettoriale locale (**Qdrant**) le risposte tra i regolamenti dell'hotel.
- Se il pezzetto di testo non basta, usa (`retrieve_parent_chunks`) per scaricare l'intero documento originale e leggerlo tutto.
- Una volta ottenute le info, genera la risposta.

> **Esempio pratico (Test RAG):**
> ![Test RAG Documentale](docs/Immagini_Test_Chatbot/test%20RAG.png)

#### Ramo 2: Gestione Database (Intento: `DB`) - *Script: `tools_db.py` e `nodes.py`*
Se l'utente dice "Voglio prenotare per domani" o "Cancella la mia stanza".
- Il bot viene instradato verso un loop di strumenti. L'LLM usa i tool definiti in **`tools_db.py`**.
- **Sicurezza SQL**: Le query in `tools_db.py` sono rigidamente *hardcoded*. L'LLM non può scrivere codice SQL (evitando SQL Injection), ma passa solo i parametri (es. date, nome).
- Il bot verifica prima la disponibilità. Se c'è, formula un preventivo. **Solo se l'utente conferma**, usa il tool `crea_prenotazione`.
- Appena il tool risponde con "SUCCESSO", il nodo `postino` rileva la parola chiave e chiama **`utils_email.py`** per inviare un'email automatica al cliente.

> **Esempi pratici (Prenotazione ed Email):**
> Il bot calcola il prezzo, si ferma per chiedere conferma (Human-In-The-Loop) e poi procede inviando l'email HTML formattata.
> ![Richiesta Prenotazione](docs/Immagini_Test_Chatbot/Richiesta%20prenotazione.png)
> ![Conferma Prenotazione](docs/Immagini_Test_Chatbot/Conferma%20prenotazione.png)
> ![Email di Conferma](docs/Immagini_Test_Chatbot/Invio%20email%20conferma%20prenotazione.png)
> 
> **La sicurezza prima di tutto:** Prima di cancellare un record, il bot esige una data di verifica.
> ![Cancellazione Sicura](docs/Immagini_Test_Chatbot/Richiesta%20cancellazione.png)
> ![Email Cancellazione](docs/Immagini_Test_Chatbot/invio%20email%20conferma%20cancellazione.png)

#### Ramo 3: Ricerca Turistica (Intento: `WEB`) - *Script: `tools_web.py`*
Se l'utente chiede "Che tempo fa a Matera?" o "Eventi stasera a Matera?" o "Vorrei mangiare una pizza nei sassi di Matera".
- Il sistema capisce che l'informazione è esterna all'hotel e devia verso i nodi Web.
- Un validatore interno blocca qualsiasi ricerca non inerente a Matera/Turismo (evita che il bot venga usato come un generico ChatGPT).
- Il bot interroga le **Tavily API** per ottenere risultati dal web in tempo reale, li sintetizza e li presenta al cliente.

> **Esempio pratico (Ricerca Locale):**
> ![Test Web Tavily](docs/Immagini_Test_Chatbot/test%20WEB%20con%20Tavily.png)

### 3. Human-In-The-Loop (HITL) - *Nodo: `request_clarification`*
L'architettura non dà nulla per scontato. Se il router iniziale reputa la domanda troppo generica (es. "Voglio quella", "Aiuto"), l'esecuzione del grafo si **interrompe (interrupt)**. Il bot chiede all'utente o all'operatore di chiarire, e riprende a funzionare solo dopo aver ricevuto una risposta sensata.

---

## 💬 Che tipo di Query può fare l'utente?

Ecco alcuni esempi pratici di cosa può gestire questo sistema:

- **Domande Documentali (RAG)**: *"Quanto costa il parcheggio?"*, *"A che ora devo lasciare la stanza?"*, *"Avete un menu per celiaci?"*
- **Domande Operative (DB)**: *"C'è una stanza matrimoniale dal 15 al 18 agosto?"*, *"Ho prenotato a nome Mario Rossi, puoi cancellarla?"*, *"Puoi prenotare una Suite per due notti da domani?"*
- **Domande Geografiche (WEB)**: *"Che meteo ci sarà domani a Matera?"*, *"Quali sono i migliori ristoranti tipici vicino ai Sassi?"*


---

## 🛠️ Guida all'Installazione Step-by-Step

*(Nota: L'ambiente virtuale e il file `.env` non sono inclusi nella repository per questioni di sicurezza. Seguendo questi passaggi avvierai tutto in locale).*

### 1. Clonare la Repository
```bash
git clone https://github.com/Christianwork-hub/AI-Receptionist-Agentic-RAG.git
cd AI-Receptionist-Agentic-RAG
```

### 2. Creare e Configurare il file `.env` (OBBLIGATORIO)
Nel progetto troverai un file chiamato **`.env.example`**. 
Rinominalo in **`.env`** e compila i campi con le tue chiavi API. 
**ATTENZIONE: Senza le API Key di DeepSeek e Tavily il progetto andrà in errore!**
(Tranquillo, il `.gitignore` impedirà che il tuo `.env` venga caricato online).

### 3. Configurare Ollama (Embeddings Locali)
Per analizzare i documenti testuali senza pagare API costose, il sistema usa embedding locali.
Assicurati di aver installato [Ollama](https://ollama.com/) sul tuo PC e scarica il modello eseguendo:
```bash
ollama run bge-m3:latest
```

---

### METODO 1: Avvio Rapido con Docker (Raccomandato)
Se hai **Docker** installato sul tuo computer, l'installazione richiede un solo comando.
Docker scaricherà le dipendenze, creerà il database PostgreSQL (`hotel_db`), creerà le tabelle `stanze` e `prenotazioni` e popolerà i dati mock per te.

1. Assicurati che Docker sia in esecuzione.
2. Dal terminale esegui:
```bash
docker-compose up -d --build
```
3. Il server web è ora attivo all'indirizzo **[http://localhost:8000](http://localhost:8000)**.

*(Se non vuoi usare Docker, segui il Metodo 2 qui sotto, altrimenti salta alla sezione Ingestion).*

---

### METODO 2: Installazione Manuale (Senza Docker)

1. **Creare l'Ambiente Virtuale (VENV)**
**Su Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**Su macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Installare le Dipendenze**
```bash
pip install -r requirements.txt
```

3. **Configurazione Database PostgreSQL**
Il progetto si interfaccia con PostgreSQL tramite la variabile `DATABASE_URI` nel file `.env`.
Devi creare manualmente un database `hotel_db` ed eseguire le query SQL presenti nel file `init.sql` per creare le tabelle e inserire i dati di prova.

### 6. Configurare Ollama (Embeddings Locali)
Per analizzare i documenti testuali senza pagare API costose, il sistema usa embedding locali.
Assicurati di aver installato [Ollama](https://ollama.com/) e, da un altro terminale, esegui:
```bash
ollama run bge-m3:latest
```

---

## ▶️ Come Avviare l'Applicazione

Una volta configurato il tutto (tramite Docker o Manualmente), il bot è pronto ad operare:

### Popolare il Database Vettoriale (Ingestion) - OBBLIGATORIO
Se cloni il progetto per la prima volta, il database Qdrant sarà vuoto! Il bot non saprà rispondere a nessuna domanda sull'hotel.
1. Inserisci i documenti di testo o PDF del tuo hotel nella cartella `markdown_docs/` (creala se non esiste).
2. "Insegna" i documenti al bot caricandoli nel database vettoriale eseguendo (se usi Docker, fallo dal terminale del tuo PC o nel container):
```bash
python ingestion.py
```

### Avvio Web e Interfaccia Grafica
Se non stai usando Docker, avvia il server manualmente:
```bash
python server.py
```
👉 Vai sul tuo browser all'indirizzo: **[http://localhost:8000](http://localhost:8000)**

### Avvio in Modalità Terminale (Sviluppatori)
Utile per diagnosticare il sistema e osservare i "log di pensiero" passo passo dell'AI.
```bash
python main.py
```
