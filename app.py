# app.py
"""Flask + Socket.IO server for ASH.

Events:
- client emits "client_time" with {time: ISOstring}
- client emits "user_message" with {user: <name>, message: <text>}
- client emits "user_voice" with binary ArrayBuffer audio data
- server emits "system" with {msg: <text>}
- server emits "voice_transcript" with {text: <transcribed_text>}
- server emits "ash_response" with {text: <response>}
"""

import sys
import traceback
import threading
import io
from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from typing import Dict
from datetime import datetime
import speech_recognition as sr
from pydub import AudioSegment
import os
os.environ["PYTHONUTF8"] = "1"

from tools import TTSEngine

try:
    from src.py.ash import ash  # preferred: import the instantiated object
except Exception:
    # fallback: try importing class and instantiate
    try:
        from src.py.ash import ASH as ASHClass
        ash = ASHClass()
    except Exception as e:
        print("Failed to import ASH:", e, file=sys.stderr)
        raise

# Derive a default user name from ash if present
DEFAULT_USER = getattr(ash, "user", None) or getattr(ash, "name", "User")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = app.config.get("SECRET_KEY", "dev-secret")

# SocketIO setup: keeping your exact parameters intact
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_timeout=120,
    ping_interval=25,
    max_http_buffer_size=10_000_000, 
    manage_session=False,
    async_mode='threading'
)

tts_engine = TTSEngine(model_path="models/kokoro-v1.0.onnx", voices_path="models/voices-v1.0.bin")

# Per-sid client state storage
client_states: Dict[str, Dict] = {}  # sid -> {"client_time": ..., "history": [...]}

# Lock to prevent overlapping requests per sid
_processing_locks: Dict[str, threading.Lock] = {}


def _get_user_name(sid):
    state = client_states.get(sid, {})
    return state.get("user") or DEFAULT_USER


def _execute_ash_pipeline(sid, user, msg):
    """Unified background execution core that prevents multi-click race conditions

    while talking to ASH.
    """
    msg = msg.strip()
    if not msg:
        return

    # Prevent overlapping requests from the same client
    lock = _processing_locks.setdefault(sid, threading.Lock())
    if not lock.acquire(blocking=False):
        print(f"[SOCKET] skipping duplicate request from {sid} (still processing)", file=sys.stderr, flush=True)
        socketio.emit("ash_response", {"text": "Still thinking about your last message..."}, room=sid)
        return

    def _process():
        try:
            state = client_states.setdefault(sid, {"history": []})
            query = f"{user}: {msg}"

            try:
                response = ash.run(msg)
            except TypeError:
                # backward compatibility profile loop
                response = ash.run(query)
            except Exception as e:
                print("[ERROR] ash.run raised an exception:", e, file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                response = "Sorry — something failed inside the assistant."

            state["history"].append({
                "user": user,
                "message": msg,
                "response_preview": str(response)[:300],
                "time": datetime.now().isoformat()
            })

            # Emit back to the client who sent it
            socketio.emit("ash_response", {"text": response}, room=sid)
            
            # Stream the TTS Audio Chunks
            for audio_chunk in tts_engine.stream_audio(response):
                    socketio.emit("ash_audio", {"audio": audio_chunk}, room=sid)

        except Exception as exc:
            print("[ERROR] handle_user_message background task failed:", exc, file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            socketio.emit("ash_response", {"text": "Server error handling message."}, room=sid)
        finally:
            lock.release()

    socketio.start_background_task(_process)


@app.route("/")
def index():
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
    client_states.pop(sid, None)
    _processing_locks.pop(sid, None)
    try:
        leave_room(sid)
    except Exception:
        pass


@socketio.on("user_message")
def handle_user_message(data):
    sid = request.sid
    try:
        msg = (data.get("message", "") if isinstance(data, dict) else "") or ""
        user = (data.get("user") if isinstance(data, dict) else None) or DEFAULT_USER
        
        print(f"[SOCKET] user_message from {sid} user={user} msg={msg}", file=sys.stderr, flush=True)
        _execute_ash_pipeline(sid, user, msg)
    except Exception as exc:
        print("[ERROR] handle_user_message top-level fail:", exc, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        emit("ash_response", {"text": "Server error handling message."}, room=sid)


@socketio.on("user_voice")
def handle_user_voice(audio_bytes):
    """Processes incoming binary audio byte payloads entirely in RAM memory,

    transcribes via speech_recognition, and hands off execution cleanly.
    """
    sid = request.sid
    if not audio_bytes:
        return

    print(f"[SOCKET] received binary voice chunk ({len(audio_bytes)} bytes) from {sid}", file=sys.stderr, flush=True)
    socketio.emit("system", {"msg": "🎙️ Processing your voice entry..."}, room=sid)

    try:

        # Stream bytes to audio file interface wrappers purely in memory
        audio_stream = io.BytesIO(audio_bytes)
        sound = AudioSegment.from_file(audio_stream)
        
        # Export uncompressed WAV structure parameters for local transcription engines
        wav_stream = io.BytesIO()
        sound.export(wav_stream, format="wav")
        wav_stream.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_stream) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)

        if transcript.strip():
            print(f"[SOCKET] STT Success for {sid}: '{transcript}'", file=sys.stderr)
            
            # 1. Update client UI layout with text transcription
            socketio.emit("voice_transcript", {"text": transcript}, room=sid)
            
            # 2. Process query directly
            user = _get_user_name(sid)
            _execute_ash_pipeline(sid, user, transcript)
        else:
            socketio.emit("system", {"msg": "⚠️ No audible speech detected. Please speak clearly."}, room=sid)

    except Exception as e:
        print(f"[SOCKET STT ERROR]: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        socketio.emit("system", {"msg": "❌ Server voice transcription processing engine failed."}, room=sid)


if __name__ == "__main__":
    print("Starting ASH Socket.IO server on 0.0.0.0:5000", file=sys.stderr)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)