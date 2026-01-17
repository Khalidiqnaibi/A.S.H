# app.py
"""Flask + Socket.IO server for ASH.

Events:
- client emits "client_time" with {time: ISOstring}
- client emits "user_message" with {user: <name>, message: <text>}
- server emits "system" with {msg: <text>}
- server emits "ash_response" with {text: <response>}
"""

import sys
import traceback
from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from typing import Dict
from datetime import datetime

try:
    from ASH2.src.py.ash import ash  # preferred: import the instantiated object
except Exception:
    # fallback: try importing class and instantiate
    try:
        from ASH2.src.py.ash import ASH as ASHClass
        ash = ASHClass()
    except Exception as e:
        print("Failed to import ASH:", e, file=sys.stderr)
        raise

# Derive a default user name from ash if present
DEFAULT_USER = getattr(ash, "user", None) or getattr(ash, "name", "User")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = app.config.get("SECRET_KEY", "dev-secret")

# SocketIO setup: allow all origins (development). Adjust in production.
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10_000_000, 
    manage_session=False
)


# Per-sid client state storage
client_states: Dict[str, Dict] = {}  # sid -> {"client_time": ..., "history": [...]}

@app.route("/")
def index():
    # Render index.html and pass 'user' so your client template can use it
    return render_template("index.html", user=DEFAULT_USER)

@socketio.on("connect")
def on_connect():
    sid = request.sid
    print(f"[SOCKET] connect: {sid}", file=sys.stderr, flush=True)
    client_states[sid] = {"client_time": None, "history": []}
    join_room(sid)
    emit("system", {"msg": "Ash is up!"}, room=sid)

@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    print(f"[SOCKET] disconnect: {sid}", file=sys.stderr, flush=True)
    emit("system", {"msg": "Ash says bye!"}, room=sid)
    # cleanup
    client_states.pop(sid, None)
    try:
        leave_room(sid)
    except Exception:
        pass

@socketio.on("user_message")
def handle_user_message(data):
    """
    Expects:
      data = {
        "user": "Khalid",    # optional
        "message": "hello ash"
      }
    Behavior:
      - builds query as "<user>: <message>" (keeps compatibility)
      - passes context={'client_time': <stored client time>} into ash.run
      - emits ash_response { text: <response> } back to the sender
    """
    sid = request.sid
    try:
        msg = (data.get("message", "") if isinstance(data, dict) else "") or ""
        user = (data.get("user") if isinstance(data, dict) else None) or DEFAULT_USER
        msg = msg.strip()
        if not msg:
            return

        print(f"[SOCKET] user_message from {sid} user={user} msg={msg}", file=sys.stderr, flush=True)

        # prepare context with client_time if present
        state = client_states.setdefault(sid, { "history": []})

        # Form the query string the same way your older app did
        query = f"{user}: {msg}"

        # Call ASH.run with context; ash.run may raise — handle gracefully
        try:
            response = ash.run(msg)
        except TypeError:
            # backward compatibility: some ASH.run definitions accept only (query)
            response = ash.run(query)
        except Exception as e:
            print("[ERROR] ash.run raised an exception:", e, file=sys.stderr, flush=True)
            traceback.print_exc()
            response = "Sorry — something failed inside the assistant."

        # Optionally store in per-sid history (for debugging or reuse)
        state["history"].append({"user": user, "message": msg, "response_preview": str(response)[:300], "time": datetime. now().isoformat() if 'datetime' in globals() else None})

        # Emit back to the client who sent it
        emit("ash_response", {"text": response}, room=sid)

    except Exception as exc:
        print("[ERROR] handle_user_message failed:", exc, file=sys.stderr, flush=True)
        traceback.print_exc()
        emit("ash_response", {"text": "Server error handling message."}, room=sid)


# Start server
if __name__ == "__main__":
    print("Starting ASH Socket.IO server on 0.0.0.0:5000", file=sys.stderr)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
