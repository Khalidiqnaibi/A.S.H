from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ASH2.src.py.ash import ASH, USER

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret'

socketio = SocketIO(app, cors_allowed_origins="*")
ash = ASH()


@app.route("/")
def index():
    return render_template("index.html",user=USER)  # your chat page


@socketio.on("connect")
def on_connect():
    print("Client connected")
    emit("system", {"msg": "Ash is up!"})


@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected")


@socketio.on("user_message")
def handle_user_message(data):
    """
    data = {
        "user": "Khalid",
        "message": "hello ash"
    }
    """
    msg = data.get("message", "").strip()
    user = data.get("user") or USER

    if not msg:
        return

    query = f"{user}: {msg}"

    # 🔥 Call your existing logic
    response = ash.run(query)

    # Send back to client
    emit("ash_response", {
        "text": response
    })


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
