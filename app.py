# app.py
import os
import json
import time
import sqlite3
import hashlib
from typing import List, Optional, Dict, Any

import numpy as np
from openai import OpenAI
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# -----------------------------
# Konfiguration
# -----------------------------
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

DB_PATH = os.getenv("K22BOT_DB_PATH", "knowledge.db")
KNOWLEDGE_JSON = os.getenv("K22BOT_KNOWLEDGE_JSON", "knowledge.json")
LOG_PATH = os.getenv("K22BOT_LOG_PATH", "conversation_log.json")

TOP_K = int(os.getenv("K22BOT_TOP_K", "4"))

# Score-Stufen (bitte nach realen Logs feinjustieren)
DIRECT_ANSWER_THRESHOLD = float(os.getenv("K22BOT_DIRECT_THRESHOLD", "0.78"))
RAG_THRESHOLD = float(os.getenv("K22BOT_RAG_THRESHOLD", "0.68"))

# Rate Limit (sehr simpel, in-memory; für mehrere Worker besser Redis/Flask-Limiter nutzen)
RATE_LIMIT_WINDOW_SEC = int(os.getenv("K22BOT_RL_WINDOW_SEC", "60"))
RATE_LIMIT_MAX_REQ = int(os.getenv("K22BOT_RL_MAX_REQ", "30"))

client = OpenAI()  # API-Key aus ENV (OPENAI_API_KEY)

app = Flask(__name__, static_folder="static")
CORS(app)  # optional: später auf Domains einschränken

# in-memory rate-limit store: {ip: [timestamps]}
_rl_store: Dict[str, List[float]] = {}

# --- oben bei Konfiguration ergänzen ---
ADMIN_API_KEY = os.getenv("K22BOT_ADMIN_API_KEY", "")  # unbedingt setzen!

def require_admin(req) -> bool:
    # Header: X-Admin-Key: <key>
    key = req.headers.get("X-Admin-Key", "")
    return bool(ADMIN_API_KEY) and key == ADMIN_API_KEY


def db_query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -----------------------------
# Hilfsfunktionen
# -----------------------------
def build_embedding_input(frage: str, antwort: str, freitext: str, kategorie: str) -> str:
    parts = [frage, antwort, freitext, kategorie]
    return "\n".join([p.strip() for p in parts if p and p.strip()])


def get_embedding(text: str) -> np.ndarray:
    emb = client.embeddings.create(input=text, model=OPENAI_EMBED_MODEL).data[0].embedding
    return np.array(emb, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def sha256_short(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def rate_limit_ok(ip: str) -> bool:
    now = time.time()
    bucket = _rl_store.get(ip, [])
    # nur Requests im Fenster behalten
    bucket = [t for t in bucket if now - t <= RATE_LIMIT_WINDOW_SEC]
    if len(bucket) >= RATE_LIMIT_MAX_REQ:
        _rl_store[ip] = bucket
        return False
    bucket.append(now)
    _rl_store[ip] = bucket
    return True


# -----------------------------
# Datenbank
# -----------------------------
def init_db():
    # knowledge.json laden
    if not os.path.exists(KNOWLEDGE_JSON):
        raise FileNotFoundError(f"{KNOWLEDGE_JSON} nicht gefunden.")

    with open(KNOWLEDGE_JSON, encoding="utf-8") as f:
        knowledge = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frage TEXT,
            antwort TEXT,
            freitext TEXT,
            kategorie TEXT,
            embedding BLOB
        )
    """)

    c.execute("SELECT COUNT(*) FROM knowledge")
    count = c.fetchone()[0]

    if count == 0:
        print("Erzeuge Embeddings und befülle Wissensdatenbank...")
        for entry in knowledge:
            frage = entry.get("frage", "") or ""
            antwort = entry.get("antwort", "") or ""
            freitext = entry.get("freitext", "") or ""
            kategorie = entry.get("kategorie", "") or ""

            text = build_embedding_input(frage, antwort, freitext, kategorie)
            emb = client.embeddings.create(input=text, model=OPENAI_EMBED_MODEL).data[0].embedding
            emb_bytes = np.array(emb, dtype=np.float32).tobytes()

            c.execute(
                "INSERT INTO knowledge (frage, antwort, freitext, kategorie, embedding) VALUES (?, ?, ?, ?, ?)",
                (frage, antwort, freitext, kategorie, emb_bytes),
            )
        conn.commit()
        print("Datenbank initialisiert!")
    else:
        print(f"Datenbank vorhanden ({count} Einträge).")

    conn.close()


def find_top_contexts(question: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    q_emb = get_embedding(question)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT frage, antwort, freitext, kategorie, embedding FROM knowledge")

    scored: List[Dict[str, Any]] = []
    for frage, antwort, freitext, kategorie, emb_bytes in c.fetchall():
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        score = cosine_similarity(q_emb, emb)
        scored.append({
            "frage": frage,
            "antwort": antwort,
            "freitext": freitext,
            "kategorie": kategorie,
            "score": score
        })

    conn.close()
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


# -----------------------------
# LLM Antwort
# -----------------------------
def ask_openai(question: str, contexts: Optional[List[Dict[str, Any]]] = None) -> str:
    system = (
        "Du bist ein Assistenzsystem für ein Schlaflabor. "
        "Beantworte ausschließlich Fragen zum Schlaflabor (Abläufe, Vorbereitung, Geräte, Termine, Befunde allgemein). "
        "Wenn es nicht zum Schlaflabor passt, antworte exakt: 'Ich beantworte nur Fragen zum Schlaflabor.' "
        "Antworte kurz, klar und verständlich."
    )

    messages = [{"role": "system", "content": system}]

    if contexts:
        # nur die wichtigsten Infos als Kontext, nicht endlos
        ctx_lines = []
        for c in contexts:
            line = f"- Frage: {c.get('frage','')}\n  Antwort: {c.get('antwort','')}"
            ft = (c.get("freitext") or "").strip()
            if ft:
                line += f"\n  Zusatz: {ft}"
            cat = (c.get("kategorie") or "").strip()
            if cat:
                line += f"\n  Kategorie: {cat}"
            ctx_lines.append(line)

        messages.append({"role": "system", "content": "Wissensbasis:\n" + "\n\n".join(ctx_lines)})

    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        max_tokens=220,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# -----------------------------
# Logging (datensparsam)
# -----------------------------
def append_log(question: str, best: Optional[Dict[str, Any]], mode: str):
    entry = {
        "ts": int(time.time()),
        "q_hash": sha256_short(question),
        "q_preview": question[:160],
        "mode": mode,  # "direct" | "rag" | "fallback"
        "best_score": (best or {}).get("score"),
        "kategorie": (best or {}).get("kategorie"),
    }

    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        else:
            data = []

        data.append(entry)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        # Logging darf den Bot nicht killen
        pass


# -----------------------------
# Routes
# -----------------------------
@app.route("/chatbot", methods=["POST"])
def chatbot():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
    if not rate_limit_ok(ip):
        return jsonify({"answer": "Zu viele Anfragen. Bitte kurz warten und erneut versuchen."}), 429

    data = request.json or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"answer": "Bitte stelle eine Frage."}), 400

    contexts = find_top_contexts(question, k=TOP_K)
    best = contexts[0] if contexts else None
    best_score = float(best["score"]) if best else 0.0

    # Entscheidung
    if best_score >= DIRECT_ANSWER_THRESHOLD:
        answer = best["antwort"]
        mode = "direct"
        used_contexts = [best]
    elif best_score >= RAG_THRESHOLD:
        answer = ask_openai(question, contexts=contexts)
        mode = "rag"
        used_contexts = contexts
    else:
        answer = ask_openai(question, contexts=None)
        mode = "fallback"
        used_contexts = None

    append_log(question, best, mode)

    return jsonify({
        "answer": answer,
        "mode": mode,
        "best_score": best_score,
        "context": used_contexts
    })

@app.route("/admin/knowledge", methods=["GET"])
def admin_list_knowledge():
    if not require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401

    # optional: ?q=suche&cat=Allgemein&limit=50&offset=0
    q = (request.args.get("q") or "").strip()
    cat = (request.args.get("cat") or "").strip()
    limit = min(int(request.args.get("limit", "50")), 200)
    offset = max(int(request.args.get("offset", "0")), 0)

    where = []
    params: List[Any] = []

    if q:
        where.append("(frage LIKE ? OR antwort LIKE ? OR freitext LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])

    if cat:
        where.append("kategorie = ?")
        params.append(cat)

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT id, frage, antwort, freitext, kategorie
        FROM knowledge
        {where_sql}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    rows = db_query(sql, tuple(params))
    return jsonify({"items": rows, "limit": limit, "offset": offset})


@app.route("/admin/knowledge/<int:item_id>", methods=["GET"])
def admin_get_knowledge(item_id: int):
    if not require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401

    rows = db_query(
        "SELECT id, frage, antwort, freitext, kategorie FROM knowledge WHERE id = ?",
        (item_id,),
    )
    if not rows:
        return jsonify({"error": "Not found"}), 404
    return jsonify(rows[0])



@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")

# -----------------------------
# Start
# -----------------------------
if __name__ == "__main__":
    init_db()
    # debug nur lokal aktivieren: FLASK_DEBUG=1
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug)
