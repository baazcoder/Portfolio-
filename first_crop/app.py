import os
import pickle
import numpy as np
from flask import Flask, request, flash, redirect, url_for, render_template
from pathlib import Path

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change‑me‑to‑secure‑value")

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.pkl"

try:
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
except Exception as exc:
    app.logger.error("Failed to load model: %s", exc)
    model = None

@app.before_request
def debug_environment():
    tpl = HERE / "templates"
    stc = HERE / "static"
    app.logger.info("cwd: %s", Path.cwd())
    app.logger.info("script dir: %s", HERE)
    app.logger.info("template folder: %s (exists? %s)", tpl, tpl.exists())
    app.logger.info("static folder: %s (exists? %s)", stc, stc.exists())
    app.logger.info("model.pkl path: %s (exists? %s)", MODEL_PATH, MODEL_PATH.exists())

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    field_names = [
        "nitrogen", "phosphorus", "potassium",
        "temperature", "humidity", "ph_value", "rainfall"
    ]
    try:
        values = []
        for field in field_names:
            raw = request.form.get(field, "").strip()
            if raw == "":
                raise ValueError(f"Missing value for {field}.")
            values.append(float(raw))
    except ValueError as err:
        flash(f"Input error: {err}", "error")
        return redirect(url_for("home"))

    if model is None:
        flash("Model unavailable. Check logs.", "error")
        return redirect(url_for("home"))

    try:
        X = np.array([values])
        prediction = model.predict(X)[0]
    except Exception as exc:
        app.logger.error("Prediction failure: %s", exc)
        flash("Model prediction error.", "error")
        return redirect(url_for("home"))

    return render_template("index.html", prediction_text=f"🧪 Predicted crop: {prediction}")

@app.errorhandler(500)
def internal_error(error):
    flash("Internal server error.", "error")
    return render_template("index.html"), 500

if __name__ == "__main__":
    app.run(debug=True)
