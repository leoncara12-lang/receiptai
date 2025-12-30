\
import os
import sqlite3
import json
import uuid
import hmac
import hashlib
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from io import BytesIO
from sentence_transformers import SentenceTransformer

# ----------------------------
# Config (puoi cambiare via env)
# ----------------------------
DB_PATH = os.getenv("RECEIPTAI_DB", "receiptai.sqlite3")
SIGNING_SECRET = os.getenv("RECEIPT_SIGNING_SECRET", "change-me-super-secret")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

THRESH_SUPPORTED = float(os.getenv("THRESH_SUPPORTED", "0.78"))
THRESH_WEAK = float(os.getenv("THRESH_WEAK", "0.67"))

# CORS: in locale basta "*"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# ----------------------------
# App + model
# ----------------------------
app = FastAPI(title="ReceiptAI Single", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# DB helpers
# ----------------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()
    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      source TEXT NOT NULL DEFAULT 'upload',
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      document_id INTEGER NOT NULL,
      locator TEXT NOT NULL,
      text TEXT NOT NULL,
      embedding_json TEXT NOT NULL,
      FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS receipts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      receipt_id TEXT NOT NULL UNIQUE,
      policy_profile TEXT NOT NULL,
      overall_status TEXT NOT NULL,
      input_text TEXT NOT NULL,
      output_text TEXT NOT NULL,
      signature TEXT NOT NULL,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS claims (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      receipt_fk INTEGER NOT NULL,
      claim_key TEXT NOT NULL,
      text TEXT NOT NULL,
      type TEXT NOT NULL,
      risk TEXT NOT NULL,
      verdict TEXT NOT NULL,
      confidence REAL NOT NULL,
      FOREIGN KEY(receipt_fk) REFERENCES receipts(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS evidence (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      claim_fk INTEGER NOT NULL,
      source_type TEXT NOT NULL,
      source_id TEXT NOT NULL,
      locator TEXT NOT NULL,
      quote_hash TEXT NOT NULL,
      similarity REAL NOT NULL,
      excerpt TEXT NOT NULL,
      FOREIGN KEY(claim_fk) REFERENCES claims(id) ON DELETE CASCADE
    );
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup():
    init_db()

# ----------------------------
# Text utilities
# ----------------------------
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def clamp(s: str, n: int = 380) -> str:
    s = normalize_ws(s)
    return s if len(s) <= n else s[:n-1] + "…"

def chunk_text(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i + CHUNK_SIZE)
        out.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - CHUNK_OVERLAP)
    return out

def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t.strip():
            pages.append(t)
    return "\n\n".join(pages)

# ----------------------------
# Embeddings + similarity
# ----------------------------
def embed(texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # normalized => dot

# ----------------------------
# Receipt + signing
# ----------------------------
def sign_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    mac = hmac.new(SIGNING_SECRET.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return "hmac-sha256:" + mac

def overall_status_from_claims(claims: List[Dict[str, Any]]) -> str:
    verdicts = [c["verdict"] for c in claims]
    if any(v in ("UNSUPPORTED", "CONFLICT") for v in verdicts):
        return "RED"
    if any(v == "WEAK" for v in verdicts):
        return "YELLOW"
    return "GREEN"

def hash_excerpt(excerpt: str) -> str:
    h = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    return "sha256:" + h

# ----------------------------
# Core logic
# ----------------------------
def extract_claims(text: str) -> List[Dict[str, str]]:
    claims = []
    for i, s in enumerate(split_sentences(text), start=1):
        lower = s.lower()
        risk = "medium"
        if any(k in lower for k in ["€", "prezzo", "costo", "rimborso", "garanzia", "sla", "penale", "contratto", "legale"]):
            risk = "high"
        ctype = "fact"
        if any(k in lower for k in ["credo", "penso", "secondo me", "opinione"]):
            ctype = "opinion"
            risk = "low"
        claims.append({"id": f"c{i}", "text": s, "type": ctype, "risk": risk})
    return claims

def get_all_chunks(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute("""
      SELECT c.id as chunk_id, c.document_id, c.locator, c.text, c.embedding_json, d.title
      FROM chunks c JOIN documents d ON d.id = c.document_id
      ORDER BY c.id ASC
    """).fetchall()
    out = []
    for r in rows:
        out.append({
            "chunk_id": r["chunk_id"],
            "document_id": r["document_id"],
            "doc_title": r["title"],
            "source_id": f"doc:{r['document_id']}",
            "locator": r["locator"],
            "text": r["text"],
            "embedding": np.array(json.loads(r["embedding_json"]), dtype=np.float32),
        })
    return out

def retrieve_top_k(query_vec: np.ndarray, chunks: List[Dict[str, Any]], top_k: int):
    scored = []
    for ch in chunks:
        sim = cosine_sim(query_vec, ch["embedding"])
        scored.append((sim, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:max(1, top_k)]

def verify_claim(claim_text: str, chunks: List[Dict[str, Any]], top_k: int):
    qv = embed([claim_text])[0]
    top = retrieve_top_k(qv, chunks, top_k=top_k)

    evid = []
    best = 0.0
    for sim, ch in top[:3]:
        best = max(best, sim)
        ex = clamp(ch["text"])
        evid.append({
            "source_type": "internal_doc",
            "source_id": ch["source_id"],
            "locator": ch["locator"],
            "quote_hash": hash_excerpt(ex),
            "similarity": float(sim),
            "excerpt": ex
        })

    if best >= THRESH_SUPPORTED:
        return ("SUPPORTED", float(best), evid)
    if best >= THRESH_WEAK:
        return ("WEAK", float(best), evid)
    return ("UNSUPPORTED", float(best), evid)

def rewrite_safely(claims: List[Dict[str, Any]]) -> str:
    supported = [c for c in claims if c["verdict"] == "SUPPORTED"]
    weak = [c for c in claims if c["verdict"] == "WEAK"]
    bad = [c for c in claims if c["verdict"] not in ("SUPPORTED", "WEAK")]

    lines = []
    if supported:
        lines.append("✅ Informazioni verificate dalle fonti disponibili:")
        for c in supported:
            lines.append(f"- {c['text']}")
    if weak:
        lines.append("\n🟡 Informazioni parzialmente supportate (potrebbero richiedere conferma):")
        for c in weak:
            lines.append(f"- {c['text']}")
    if bad:
        lines.append("\n🔴 Non posso confermare alcune parti con le fonti disponibili:")
        for c in bad:
            lines.append(f"- {c['text']}")
    return "\n".join(lines) if lines else "Non ho abbastanza informazioni verificabili dalle fonti disponibili."

# ----------------------------
# API Schemas
# ----------------------------
class VerifyRequest(BaseModel):
    text: str
    policy_profile: str = "default"
    top_k: int = 5

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/documents/upload")
async def upload_document(title: str, file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File vuoto")

    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        raw_text = extract_text_from_pdf(data)
    else:
        raw_text = data.decode("utf-8", errors="ignore")

    raw_text = raw_text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Testo non estratto (PDF vuoto o non leggibile)")

    chunks = chunk_text(raw_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Nessun chunk creato")

    vecs = embed(chunks)

    conn = db()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute("INSERT INTO documents (title, source, created_at) VALUES (?, ?, ?)", (title, "upload", now))
    doc_id = cur.lastrowid

    for i, (ch, v) in enumerate(zip(chunks, vecs), start=1):
        cur.execute(
            "INSERT INTO chunks (document_id, locator, text, embedding_json) VALUES (?, ?, ?, ?)",
            (doc_id, f"chunk:{i}", ch, json.dumps(v.tolist()))
        )

    conn.commit()
    conn.close()

    return {"id": doc_id, "title": title}

@app.get("/documents")
def list_documents():
    conn = db()
    rows = conn.execute("SELECT id, title, source, created_at FROM documents ORDER BY id DESC LIMIT 200").fetchall()
    conn.close()
    return [{"id": r["id"], "title": r["title"], "source": r["source"], "created_at": r["created_at"]} for r in rows]

@app.post("/verify")
def verify(req: VerifyRequest):
    conn = db()
    chunks = get_all_chunks(conn)
    if not chunks:
        conn.close()
        raise HTTPException(status_code=400, detail="Carica prima almeno un documento.")

    raw_claims = extract_claims(req.text)

    out_claims = []
    for c in raw_claims:
        verdict, conf, evid = verify_claim(c["text"], chunks, top_k=req.top_k)
        out_claims.append({
            "id": c["id"],
            "text": c["text"],
            "type": c["type"],
            "risk": c["risk"],
            "verdict": verdict,
            "confidence": conf,
            "evidence": evid
        })

    overall = overall_status_from_claims(out_claims)
    output_text = rewrite_safely(out_claims)

    receipt_id = "rcpt_" + uuid.uuid4().hex[:16]
    payload = {
        "receipt_id": receipt_id,
        "policy_profile": req.policy_profile,
        "overall_status": overall,
        "input_text": req.text,
        "output_text": output_text,
        "claims": out_claims
    }
    signature = sign_payload(payload)

    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO receipts (receipt_id, policy_profile, overall_status, input_text, output_text, signature, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (receipt_id, req.policy_profile, overall, req.text, output_text, signature, now)
    )
    receipt_fk = cur.lastrowid

    for c in out_claims:
        cur.execute(
            "INSERT INTO claims (receipt_fk, claim_key, text, type, risk, verdict, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (receipt_fk, c["id"], c["text"], c["type"], c["risk"], c["verdict"], c["confidence"])
        )
        claim_fk = cur.lastrowid
        for e in c["evidence"][:3]:
            cur.execute(
                "INSERT INTO evidence (claim_fk, source_type, source_id, locator, quote_hash, similarity, excerpt) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (claim_fk, e["source_type"], e["source_id"], e["locator"], e["quote_hash"], e["similarity"], e["excerpt"])
            )

    conn.commit()
    conn.close()

    return {"receipt_id": receipt_id, "overall_status": overall, "output_text": output_text, "claims": out_claims, "signature": signature}

@app.get("/receipt/{receipt_id}")
def get_receipt(receipt_id: str):
    conn = db()
    rec = conn.execute("SELECT * FROM receipts WHERE receipt_id = ?", (receipt_id,)).fetchone()
    if not rec:
        conn.close()
        raise HTTPException(status_code=404, detail="Receipt non trovata")

    claim_rows = conn.execute("SELECT * FROM claims WHERE receipt_fk = ? ORDER BY id ASC", (rec["id"],)).fetchall()
    claims = []
    for cr in claim_rows:
        ev_rows = conn.execute("SELECT * FROM evidence WHERE claim_fk = ? ORDER BY id ASC", (cr["id"],)).fetchall()
        claims.append({
            "id": cr["claim_key"],
            "text": cr["text"],
            "type": cr["type"],
            "risk": cr["risk"],
            "verdict": cr["verdict"],
            "confidence": cr["confidence"],
            "evidence": [{
                "source_type": er["source_type"],
                "source_id": er["source_id"],
                "locator": er["locator"],
                "quote_hash": er["quote_hash"],
                "similarity": er["similarity"],
                "excerpt": er["excerpt"],
            } for er in ev_rows]
        })

    conn.close()
    return {
        "receipt_id": rec["receipt_id"],
        "policy_profile": rec["policy_profile"],
        "overall_status": rec["overall_status"],
        "input_text": rec["input_text"],
        "output_text": rec["output_text"],
        "signature": rec["signature"],
        "created_at": rec["created_at"],
        "claims": claims
    }

# ----------------------------
# Frontend (static)
# ----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
