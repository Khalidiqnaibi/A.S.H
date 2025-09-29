from flask import Flask, request, jsonify
import sqlite3
import json
import time
import os

DB_PATH = os.getenv("META_DB_PATH", "metadata.db")

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT,
        created_at INTEGER
    )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/metadata/save", methods=["POST"])
def save_metadata():
    data = request.json
    key = data.get("key", "default_metadata")
    value = json.dumps(data.get("value", {}))
    created_at = int(time.time())

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO metadata (key, value, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at
    """, (key, value, created_at))
    conn.commit()
    conn.close()

    return jsonify({"status": "ok", "key": key}), 200


@app.route("/metadata/<key>", methods=["GET"])
def load_metadata(key):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()

    if row:
        return jsonify({"key": key, "value": json.loads(row[0])}), 200
    else:
        return jsonify({"error": "Key not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2020, debug=True)
