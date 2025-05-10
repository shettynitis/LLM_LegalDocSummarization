from flask import Blueprint, request, jsonify, render_template, current_app
from .model_utils import summarize_text

main = Blueprint("main", __name__)

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please provide some text."}), 400

    # —— LOG IT ——
    # truncate to, say, first 1000 chars to avoid huge entries
    snippet = text.replace("\n", " ")[:1000]
    current_app.logger.info(f"User submitted: “{snippet}”")
    # ————————

    summary = summarize_text(text)
    return jsonify({"summary": summary})