-- Creazione delle tabelle base per il database dell'Hotel

CREATE TABLE IF NOT EXISTS stanze (
    id SERIAL PRIMARY KEY,
    numero_stanza VARCHAR(10) NOT NULL,
    tipologia VARCHAR(50) NOT NULL,
    prezzo_base NUMERIC(10, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS prenotazioni (
    id SERIAL PRIMARY KEY,
    stanza_id INT REFERENCES stanze(id),
    nome_cliente VARCHAR(100) NOT NULL,
    check_in DATE NOT NULL,
    check_out DATE NOT NULL,
    prezzo_totale NUMERIC(10, 2),
    stato VARCHAR(50) DEFAULT 'Confermata'
);

-- Inserimento di dati mock per poter testare il bot da subito
INSERT INTO stanze (numero_stanza, tipologia, prezzo_base) 
VALUES 
    ('101', 'Matrimoniale', 120.00),
    ('102', 'Singola', 80.00),
    ('201', 'Suite', 250.00)
ON CONFLICT DO NOTHING; -- Nel caso si riavvii il container
