# app.py
import os
import json
import time
import sqlite3
import hashlib
from typing import List, Optional, Dict, Any, Tuple

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

DIRECT_ANSWER_THRESHOLD = float(os.getenv("K22BOT_DIRECT_THRESHOLD", "0.78"))
RAG_THRESHOLD = float(os.getenv("K22BOT_RAG_THRESHOLD", "0.68"))

RATE_LIMIT_WINDOW_SEC = int(os.getenv("K22BOT_RL_WINDOW_SEC", "60"))
RATE_LIMIT_MAX_REQ = int(os.getenv("K22BOT_RL_MAX_REQ", "30"))

ADMIN_API_KEY = os.getenv("K22BOT_ADMIN_API_KEY", "")  # unbedingt setzen!
PRUNE_MISSING = os.getenv("K22BOT_PRUNE_MISSING", "0") == "1"

client = OpenAI()

app = Flask(__name__, static_folder="static")
CORS(app)

_rl_store: Dict[str, List[float]] = {}
_db_ready = False


# -----------------------------
# Admin / DB Helper
# -----------------------------
def require_admin(req) -> bool:
    key = req.headers.get("X-Admin-Key", "")
    return bool(ADMIN_API_KEY) and key == ADMIN_API_KEY


def db_query(sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -----------------------------
# Allgemeine Hilfsfunktionen
# -----------------------------
def build_embedding_input(frage: str, antwort: str, freitext: str, kategorie: str) -> str:
    parts = [frage, antwort, freitext, kategorie]
    return "\n".join([p.strip() for p in parts if p and p.strip()])


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_short(text: str, n: int = 12) -> str:
    return sha256_hex(text)[:n]


def get_embedding(text: str) -> np.ndarray:
    emb = client.embeddings.create(input=text, model=OPENAI_EMBED_MODEL).data[0].embedding
    return np.array(emb, dtype=np.float32)


def embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    resp = client.embeddings.create(input=texts, model=OPENAI_EMBED_MODEL)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rate_limit_ok(ip: str) -> bool:
    now = time.time()
    bucket = _rl_store.get(ip, [])
    bucket = [t for t in bucket if now - t <= RATE_LIMIT_WINDOW_SEC]
    if len(bucket) >= RATE_LIMIT_MAX_REQ:
        _rl_store[ip] = bucket
        return False
    bucket.append(now)
    _rl_store[ip] = bucket
    return True


# -----------------------------
# Logging (datensparsam)
# -----------------------------
def append_log(question: str, best: Optional[Dict[str, Any]], mode: str):
    entry = {
        "ts": int(time.time()),
        "q_hash": sha256_short(question),
        "q_preview": question[:160],
        "mode": mode,
        "best_score": (best or {}).get("score"),
        "kategorie": (best or {}).get("kategorie"),
        "best_key_hash": (best or {}).get("key_hash"),
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
        pass


# -----------------------------
# JSON Laden
# -----------------------------
def load_knowledge_json() -> List[Dict[str, str]]:
    if not os.path.exists(KNOWLEDGE_JSON):
        raise FileNotFoundError(f"{KNOWLEDGE_JSON} nicht gefunden.")

    with open(KNOWLEDGE_JSON, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("knowledge.json muss eine Liste von Einträgen sein.")

    cleaned: List[Dict[str, str]] = []
    for entry in data:
        cleaned.append({
            "frage": (entry.get("frage") or "").strip(),
            "antwort": (entry.get("antwort") or "").strip(),
            "freitext": (entry.get("freitext") or "").strip(),
            "kategorie": (entry.get("kategorie") or "").strip(),
        })
    return cleaned


# -----------------------------
# DB Schema + Migration
# -----------------------------
REQUIRED_COLUMNS: Dict[str, str] = {
    "key_hash": "TEXT",
    "frage": "TEXT",
    "antwort": "TEXT",
    "freitext": "TEXT",
    "kategorie": "TEXT",
    "embedding": "BLOB",
    "updated_at": "INTEGER",
}

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return c.fetchone() is not None

def get_table_columns(conn: sqlite3.Connection, name: str) -> List[str]:
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({name})")
    return [row[1] for row in c.fetchall()]  # name

def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    Stellt sicher, dass die Tabelle knowledge das neue Schema hat.
    - Wenn Tabelle nicht existiert: neu anlegen
    - Wenn existiert: fehlende Spalten per ALTER TABLE ergänzen
    - Wenn das fehlschlägt: Tabelle umbenennen, neu anlegen (safe fallback)
    """
    c = conn.cursor()

    if not table_exists(conn, "knowledge"):
        c.execute("""
            CREATE TABLE knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT UNIQUE,
                frage TEXT,
                antwort TEXT,
                freitext TEXT,
                kategorie TEXT,
                embedding BLOB,
                updated_at INTEGER
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_key_hash ON knowledge(key_hash)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_updated_at ON knowledge(updated_at)")
        conn.commit()
        return

    cols = get_table_columns(conn, "knowledge")

    # wenn key_hash fehlt -> erst versuchen zu migrieren
    try:
        # fehlende Spalten ergänzen
        for col, coltype in REQUIRED_COLUMNS.items():
            if col not in cols:
                c.execute(f"ALTER TABLE knowledge ADD COLUMN {col} {coltype}")
        conn.commit()
    except sqlite3.OperationalError:
        # Fallback: Tabelle umbenennen und neu erstellen
        ts = int(time.time())
        old_name = f"knowledge_old_{ts}"
        c.execute(f"ALTER TABLE knowledge RENAME TO {old_name}")

        c.execute("""
            CREATE TABLE knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT UNIQUE,
                frage TEXT,
                antwort TEXT,
                freitext TEXT,
                kategorie TEXT,
                embedding BLOB,
                updated_at INTEGER
            )
        """)
        # Best effort: alte Daten übernehmen (ohne embeddings/key_hash -> wird beim Sync neu erstellt)
        old_cols = get_table_columns(conn, old_name)
        take = [x for x in ["frage", "antwort", "freitext", "kategorie"] if x in old_cols]
        if take:
            cols_sql = ", ".join(take)
            c.execute(f"INSERT INTO knowledge ({cols_sql}) SELECT {cols_sql} FROM {old_name}")
        conn.commit()

    # Indizes erst nach Spalten sicher anlegen
    cols = get_table_columns(conn, "knowledge")
    if "key_hash" in cols:
        c.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_key_hash ON knowledge(key_hash)")
    if "updated_at" in cols:
        c.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_updated_at ON knowledge(updated_at)")
    conn.commit()


# -----------------------------
# Sync (JSON -> DB Cache)
# -----------------------------
def sync_knowledge_from_json() -> Dict[str, Any]:
    knowledge = load_knowledge_json()

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    c = conn.cursor()

    # vorhandene Hashes
    c.execute("SELECT key_hash FROM knowledge WHERE key_hash IS NOT NULL")
    existing = {row[0] for row in c.fetchall() if row[0]}

    to_upsert: List[Dict[str, Any]] = []
    json_hashes: set[str] = set()

    for entry in knowledge:
        text = build_embedding_input(entry["frage"], entry["antwort"], entry["freitext"], entry["kategorie"])
        kh = sha256_hex(text)
        json_hashes.add(kh)

        if kh in existing:
            continue

        to_upsert.append({
            "key_hash": kh,
            "frage": entry["frage"],
            "antwort": entry["antwort"],
            "freitext": entry["freitext"],
            "kategorie": entry["kategorie"],
            "text": text,
        })

    upserted = 0
    if to_upsert:
        texts = [x["text"] for x in to_upsert]
        embs = embeddings_batch(texts)
        now_ts = int(time.time())

        for x, emb in zip(to_upsert, embs):
            emb_bytes = np.array(emb, dtype=np.float32).tobytes()
            c.execute("""
                INSERT INTO knowledge (key_hash, frage, antwort, freitext, kategorie, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key_hash) DO UPDATE SET
                    frage=excluded.frage,
                    antwort=excluded.antwort,
                    freitext=excluded.freitext,
                    kategorie=excluded.kategorie,
                    embedding=excluded.embedding,
                    updated_at=excluded.updated_at
            """, (
                x["key_hash"], x["frage"], x["antwort"], x["freitext"], x["kategorie"], emb_bytes, now_ts
            ))
            upserted += 1

        conn.commit()

    deleted = 0
    if PRUNE_MISSING:
        c.execute("SELECT key_hash FROM knowledge WHERE key_hash IS NOT NULL")
        db_hashes = {row[0] for row in c.fetchall() if row[0]}
        to_delete = list(db_hashes - json_hashes)
        if to_delete:
            c.executemany("DELETE FROM knowledge WHERE key_hash = ?", [(h,) for h in to_delete])
            conn.commit()
            deleted = len(to_delete)

    c.execute("SELECT COUNT(*) FROM knowledge")
    total = int(c.fetchone()[0])

    conn.close()

    return {
        "json_count": len(knowledge),
        "upserted": upserted,
        "deleted": deleted,
        "db_total": total,
        "db_path": DB_PATH,
        "knowledge_json": KNOWLEDGE_JSON,
    }


@app.before_request
def ensure_db_ready():
    global _db_ready
    if _db_ready:
        return
    sync_knowledge_from_json()
    _db_ready = True


# -----------------------------
# Retrieval aus DB
# -----------------------------
def find_top_contexts(question: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    q_emb = get_embedding(question)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT key_hash, frage, antwort, freitext, kategorie, embedding FROM knowledge")

    scored: List[Dict[str, Any]] = []
    for key_hash, frage, antwort, freitext, kategorie, emb_bytes in c.fetchall():
        if not emb_bytes:
            continue
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        score = cosine_similarity(q_emb, emb)
        scored.append({
            "key_hash": key_hash,
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


@app.route("/admin/knowledge", methods=["GET"], strict_slashes=False)
def admin_list_knowledge():
    if not require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401

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
        SELECT id, key_hash, frage, antwort, freitext, kategorie, updated_at
        FROM knowledge
        {where_sql}
        ORDER BY updated_at DESC, id DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    rows = db_query(sql, tuple(params))
    return jsonify({"items": rows, "limit": limit, "offset": offset})


@app.route("/admin/knowledge/<int:item_id>", methods=["GET"], strict_slashes=False)
def admin_get_knowledge(item_id: int):
    if not require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401

    rows = db_query(
        "SELECT id, key_hash, frage, antwort, freitext, kategorie, updated_at "
        "FROM knowledge WHERE id = ?",
        (item_id,),
    )
    if not rows:
        return jsonify({"error": "Not found"}), 404
    return jsonify(rows[0])


@app.route("/admin/reload", methods=["POST"], strict_slashes=False)
def admin_reload():
    if not require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401

    stats = sync_knowledge_from_json()
    return jsonify({"ok": True, "stats": stats})


@app.route("/admin/diag", methods=["GET"], strict_slashes=False)
def admin_diag():
    if not require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("PRAGMA table_info(knowledge)")
    cols = [{"cid": r[0], "name": r[1], "type": r[2]} for r in c.fetchall()]
    c.execute("SELECT COUNT(*) FROM knowledge")
    total = int(c.fetchone()[0])
    conn.close()

    return jsonify({
        "db_path": DB_PATH,
        "knowledge_json": KNOWLEDGE_JSON,
        "log_path": LOG_PATH,
        "prune_missing": PRUNE_MISSING,
        "total_rows": total,
        "columns": cols
    })


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


@app.route("/admin", methods=["GET"], strict_slashes=False)
def admin_ui():
    return send_from_directory("static", "admin.html")


# -----------------------------
# Start (nur lokal)
# -----------------------------
if __name__ == "__main__":
    sync_knowledge_from_json()
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug)
