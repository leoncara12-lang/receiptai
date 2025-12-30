# ReceiptAI Single (progetto unico, 1 comando)

Questa versione è **tutto in una sola cartella**:
- Backend FastAPI (API + serve il sito)
- Frontend statico (HTML/JS/CSS)
- Database SQLite locale (documenti, chunk, receipts)

## Avvio super semplice (locale)
1) Installare Python 3.11+
2) Da terminale nella cartella del progetto:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

3) Apri: http://localhost:8000

## Cosa fa
1) Carichi un PDF o un file di testo
2) Incolli una bozza di risposta
3) Premi "Verifica"
4) Vedi semaforo + claim + evidenze + receipt salvata

## Nota
Questo MVP usa embedding (SentenceTransformers) e retrieval con similarità cosine, calcolata in locale.
È pensato per pochi documenti (POC / demo). Per scalare, usa Postgres+pgvector o un vector DB.
