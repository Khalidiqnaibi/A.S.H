from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from typing import Dict, Any
from dotenv import load_dotenv

from AgentSystem import ChromaVDB,mistral

load_dotenv(r'A.S.H\.env')

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

# Initialize VDB
LLM = mistral.MistralLLM(mode="openrouter",openrouter_key=os.getenv("OPENROUTER_API_KEY"), temperature=0.7)
vdb = ChromaVDB(llm=LLM)
vdb.init_cloud_db_client()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/embed/text", methods=["POST"])
def embed_text():
    data: Dict[str, Any] = request.json or {}
    text = data.get("text")
    metadata = data.get("metadata", {})

    if not text:
        return jsonify({"error": "Missing text"}), 400

    vdb.embed_text(text, metadata)
    return jsonify({"status": "ok", "type": "text"})

@app.route("/embed/file", methods=["POST"])
def embed_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    metadata = request.form.to_dict()

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    vdb.add_documents(path=path, base_metadata=metadata)

    return jsonify({
        "status": "ok",
        "type": "file",
        "filename": filename
    })

@app.route("/query", methods=["POST"])
def query():
    data = request.json or {}
    query = data.get("query")
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "Missing query"}), 400

    docs = vdb.search(query, top_k=top_k)

    return jsonify({
        "query": query,
        "results": [
            {
                "content": d.page_content[:500],
                "metadata": d.metadata
            }
            for d in docs
        ]
    })

@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify(vdb.get_metadata_map())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
