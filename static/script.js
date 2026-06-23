// Genera un Thread ID univoco per l'utente (salvato nel localStorage per mantenere la sessione)
function getThreadId() {
    let threadId = localStorage.getItem('receptionist_thread_id');
    if (!threadId) {
        // Creiamo un ID stile "web_ospite_a1b2c3d4"
        threadId = 'web_ospite_' + Math.random().toString(36).substring(2, 10);
        localStorage.setItem('receptionist_thread_id', threadId);
    }
    return threadId;
}

const threadId = getThreadId();

function closeChatPopup(event) {
    if (event) {
        event.stopPropagation(); // Evita di aprire la chat se clicchi sulla X
    }
    const popup = document.getElementById('chat-popup');
    if (popup) {
        popup.classList.add('hidden');
    }
}

// Apre e chiude il widget della chat
function toggleChat() {
    const widget = document.getElementById('chat-widget');
    widget.classList.toggle('hidden');

    closeChatPopup();

    // Se la chat viene aperta, mettiamo il focus sull'input
    if (!widget.classList.contains('hidden')) {
        setTimeout(() => {
            document.getElementById('chat-input').focus();
        }, 300); // aspettiamo l'animazione
    }
}

// Aggiunge un messaggio alla UI
function appendMessage(sender, text) {
    const messagesDiv = document.getElementById('chat-messages');

    const messageWrapper = document.createElement('div');
    messageWrapper.classList.add('message');
    messageWrapper.classList.add(sender === 'user' ? 'user-message' : 'ai-message');

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');

    if (sender === 'ai') {
        // Usiamo marked.js (incluso nell'HTML) per renderizzare il markdown restituito dall'LLM
        contentDiv.innerHTML = marked.parse(text);
    } else {
        // Testo semplice per l'utente per evitare XSS
        contentDiv.textContent = text;
    }

    messageWrapper.appendChild(contentDiv);
    messagesDiv.appendChild(messageWrapper);

    // Scrolla sempre in basso
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Mostra l'indicatore "Sta scrivendo..."
function showLoading() {
    const messagesDiv = document.getElementById('chat-messages');
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('loading-dots');
    loadingDiv.id = 'loading-indicator';
    loadingDiv.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    messagesDiv.appendChild(loadingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Rimuove l'indicatore
function removeLoading() {
    const loadingDiv = document.getElementById('loading-indicator');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

// Invia il messaggio al backend FastAPI
async function sendMessage() {
    const inputField = document.getElementById('chat-input');
    const message = inputField.value.trim();
    if (!message) return;

    // Mostra subito il messaggio dell'utente
    appendMessage('user', message);
    inputField.value = '';

    showLoading();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                thread_id: threadId,
                message: message
            })
        });

        if (!response.ok) {
            throw new Error('Errore HTTP ' + response.status);
        }

        const data = await response.json();
        removeLoading();

        // Mostra la risposta dell'AI
        appendMessage('ai', data.response);

    } catch (error) {
        console.error("Errore di rete:", error);
        removeLoading();
        appendMessage('ai', 'Scusa, si è verificato un errore di connessione con il server. Riprova tra pochi istanti.');
    }
}

// Permette l'invio premendo Invio sulla tastiera
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}
