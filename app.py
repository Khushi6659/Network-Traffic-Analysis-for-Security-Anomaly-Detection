import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # to save plots without GUI
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, redirect, url_for, send_file
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Flask App ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load trained ML model
MODEL_PATH = "network_intrusion_model.pkl"
model = joblib.load(MODEL_PATH)


# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return " No file part"
    
    file = request.files["file"]
    if file.filename == "":
        return " No file selected"
    
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Read CSV
        df = pd.read_csv(filepath)

        # Ensure 'attack' column exists (else just predict features)
        if "attack" in df.columns:
            y_true = df["attack"]
            X = df.drop("attack", axis=1)
        else:
            y_true = None
            X = df

        # Run predictions
        y_pred = model.predict(X)

        # Add predictions to dataframe
        df["prediction"] = y_pred

        # Save results file
        results_path = os.path.join(app.config["UPLOAD_FOLDER"], "results.csv")
        df.to_csv(results_path, index=False)

        # If true labels exist, generate evaluation
        report_text = None
        cm_path = None
        if y_true is not None:
            report = classification_report(y_true, y_pred, output_dict=True)
            report_text = classification_report(y_true, y_pred)

            # Plot confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            cm_path = os.path.join("static", "confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()

        return render_template("results.html",
                               tables=[df.head(20).to_html(classes="data")],
                               report=report_text,
                               cm_image=cm_path,
                               download_link=url_for("download_file", filename="results.csv"))
    return redirect(url_for("home"))


@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename), as_attachment=True)


# ---------------- Run Flask ----------------
if __name__ == "__main__":
    app.run(debug=True)
