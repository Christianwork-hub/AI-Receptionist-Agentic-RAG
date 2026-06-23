# Usa l'immagine ufficiale Python leggera
FROM python:3.11-slim

# Imposta la cartella di lavoro
WORKDIR /app

# Installa dipendenze di sistema utili
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia i requisiti
COPY requirements.txt .

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice sorgente
COPY . .

# Esponi la porta 8000 per FastAPI
EXPOSE 8000

# Comando di default per avviare il server
CMD ["python", "server.py"]
