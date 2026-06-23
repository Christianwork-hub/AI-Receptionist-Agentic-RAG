"""
Il Postino (Invio Email)
------------------------
Questo file si occupa di inviare un'email grafica automatica (verde per conferme, rossa per cancellazioni)
ogni volta che un'operazione nel database va a buon fine ("SUCCESSO").
"""

import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

# Caricamento configurazioni da .env
EMAIL_MITTENTE = os.getenv("EMAIL_MITTENTE")
EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_DESTINATARIO = os.getenv("EMAIL_DESTINATARIO")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))

def _estrai_dati_prenotazione(contesto: str) -> dict:
    """
    Analizza la stringa di successo restituita dai tool del database
    per estrarre ID, Nome Cliente e tipo di operazione.
    """
    # Estrazione ID (cerca 'ID' seguito da numeri)
    id_match = re.search(r"ID (\d+)", contesto)
    id_pren = id_match.group(1) if id_match else "N/D"

    # Estrazione Nome Cliente (testo tra 'per' e il punto o 'cancellata')
    nome_match = re.search(r"per (.*?)(?:\.| cancellata)", contesto)
    nome_cli = nome_match.group(1).strip() if nome_match else "Cliente"

    # Determinazione tipo operazione
    if "cancellata" in contesto.lower():
        tipo_op = "Cancellazione prenotazione"
        stato_breve = "Cancellata"
        colore = "#FF3B30"  # Apple Red
        icona = "🗑️"
    else:
        tipo_op = "Conferma prenotazione"
        stato_breve = "Confermata"
        colore = "#34C759"  # Apple Green
        icona = "🛎️"

    return {
        "id_prenotazione": id_pren,
        "nome_cliente": nome_cli,
        "tipo": tipo_op,
        "stato_breve": stato_breve,
        "colore": colore,
        "icona": icona,
        "testo_originale": contesto
    }

def _genera_html_email(dati: dict) -> str:
    """Crea il corpo dell'email in formato HTML professionale e coerente con il sito, basato sul nuovo layout."""
    
    # Determina colori e testi in base al tipo di operazione
    if "Cancellata" in dati['stato_breve']:
        color_theme = "#E74C3C" # Rosso per cancellazione
        titolo_email = f"CANCELLAZIONE PRENOTAZIONE - AI Receptionist - {dati['id_prenotazione']}"
    else:
        color_theme = "#48C0B5" # Verde acqua per conferma (come da immagine)
        titolo_email = f"CONFERMA PRENOTAZIONE - AI Receptionist - {dati['id_prenotazione']}"
        
    return f"""
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 20px; background-color: #f4f4f4; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333333; }}
            .container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; }}
            .header-logo {{ text-align: center; padding: 25px 0; background-color: #f9f9f9; }}
            .header-logo h1 {{ margin: 0; font-family: 'Times New Roman', Times, serif; font-size: 26px; color: #555555; font-weight: normal; letter-spacing: 1px; }}
            .hero-image {{ width: 100%; height: auto; display: block; }}
            .main-title {{ text-align: center; font-size: 15px; font-weight: bold; padding: 25px 20px; margin: 0; color: #000000; letter-spacing: 0.5px; }}
            .divider {{ border: none; border-top: 1px solid #e5e5e5; margin: 0 30px; }}
            .section {{ padding: 25px 30px; }}
            .section-title {{ color: {color_theme}; font-size: 14px; font-weight: bold; margin: 0 0 15px 0; text-transform: uppercase; letter-spacing: 0.5px; }}
            .details-row {{ margin-bottom: 6px; font-size: 14px; line-height: 1.5; }}
            .details-row strong {{ color: #000000; }}
            .footer {{ padding: 25px 30px; font-size: 12px; color: #666666; line-height: 1.6; background-color: #ffffff; border-top: 1px solid #e5e5e5; }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header testuale -->
            <div class="header-logo">
                <h1>Grand Hotel Sassi</h1>
            </div>

            <!-- Immagine principale -->
            <img class="hero-image" src="https://images.unsplash.com/photo-1596720426673-e4e14290f0cc?auto=format&fit=crop&w=600&q=80" alt="Matera Sassi Hotel">

            <!-- Titolo Principale -->
            <h2 class="main-title">{titolo_email}</h2>

            <hr class="divider">

            <!-- Dettagli Prenotazione -->
            <div class="section">
                <div class="section-title">DETTAGLI PRENOTAZIONE:</div>
                <div class="details-row"><strong>Hotel:</strong> Grand Hotel Sassi</div>
                <div class="details-row"><strong>Codice prenotazione:</strong> {dati['id_prenotazione']}</div>
                <div class="details-row"><strong>Nome:</strong> {dati['nome_cliente']}</div>
                <div class="details-row"><strong>Stato:</strong> {dati['stato_breve'].upper()}</div>
            </div>

            <hr class="divider">

            <!-- Informazioni Aggiuntive -->
            <div class="section">
                <div class="section-title">INFORMAZIONI AGGIUNTIVE:</div>
                <div class="details-row" style="margin-top: 10px;">
                    L'operazione &egrave; stata registrata correttamente dal sistema.<br><br>
                    <strong>Dettaglio tecnico:</strong><br>
                    {dati['testo_originale']}
                </div>
            </div>

            <!-- Footer -->
            <div class="footer">
                <table width="100%" cellpadding="0" cellspacing="0" border="0">
                    <tr>
                        <td style="font-size: 12px; color: #666666;">
                            Via Muro, 78 - 75100 Matera, Italy<br>
                            Tel. +39 0835 123456 - Fax +39 0835 123457<br>
                            <a href="mailto:booking@grandhotelsassi.com" style="color: #666666; text-decoration: none;">booking@grandhotelsassi.com</a>
                        </td>
                        <td align="right" style="font-size: 12px; color: #666666; vertical-align: bottom;">
                            <em>AI Receptionist V3</em>
                        </td>
                    </tr>
                </table>
            </div>

        </div>
    </body>
    </html>
    """

def invia_email_hotel(contesto: str):
    """
    Funzione principale per l'invio della notifica.
    Filtra solo i messaggi di successo e invia l'email formattata.
    """
    if not all([EMAIL_MITTENTE, EMAIL_PASSWORD, EMAIL_DESTINATARIO]):
        print("   [EMAIL] Credenziali SMTP non configurate. Email saltata.")
        return

    # Processiamo solo se l'operazione sul DB è andata a buon fine
    if "SUCCESSO" not in contesto:
        return

    dati = _estrai_dati_prenotazione(contesto)
    
    # Costruiamo l'oggetto dell'email in base al tipo
    oggetto = f"[Hotel] {dati['icona']} {dati['tipo']} #{dati['id_prenotazione']} - {dati['nome_cliente']}"
    
    corpo_html = _genera_html_email(dati)

    # Configurazione MIME
    msg = MIMEMultipart("alternative")
    msg["Subject"] = oggetto
    msg["From"] = EMAIL_MITTENTE
    msg["To"] = EMAIL_DESTINATARIO
    msg.attach(MIMEText(corpo_html, "html", "utf-8"))

    # Invio SMTP
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.login(EMAIL_MITTENTE, EMAIL_PASSWORD)
            server.sendmail(EMAIL_MITTENTE, EMAIL_DESTINATARIO, msg.as_string())
        print(f"   [EMAIL] Notifica di {dati['tipo']} inviata con successo.")
    except Exception as e:
        print(f"   [EMAIL] Errore durante l'invio: {e}")