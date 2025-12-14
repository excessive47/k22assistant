from flask import Flask, render_template, request, redirect, url_for, send_file
import sqlite3
import csv
import io

app = Flask(__name__)


def get_db():
    conn = sqlite3.connect("qa_pairs.db")
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/", methods=["GET", "POST"])
def index():
    conn = get_db()
    if request.method == "POST":
        frage = request.form.get("frage", "").strip()
        antwort = request.form.get("antwort", "").strip()
        if frage and antwort:
            conn.execute(
                "INSERT INTO qa (frage, antwort) VALUES (?, ?)", (frage, antwort)
            )
            conn.commit()
        return redirect(url_for("index"))
    rows = conn.execute("SELECT id, frage, antwort FROM qa ORDER BY id DESC").fetchall()
    return render_template("index.html", qa_pairs=rows)


@app.route("/delete/<int:qa_id>")
def delete(qa_id):
    conn = get_db()
    conn.execute("DELETE FROM qa WHERE id=?", (qa_id,))
    conn.commit()
    return redirect(url_for("index"))


@app.route("/edit/<int:qa_id>", methods=["GET", "POST"])
def edit(qa_id):
    conn = get_db()
    if request.method == "POST":
        frage = request.form.get("frage", "").strip()
        antwort = request.form.get("antwort", "").strip()
        if frage and antwort:
            conn.execute(
                "UPDATE qa SET frage=?, antwort=? WHERE id=?", (frage, antwort, qa_id)
            )
            conn.commit()
            return redirect(url_for("index"))
    row = conn.execute(
        "SELECT id, frage, antwort FROM qa WHERE id=?", (qa_id,)
    ).fetchone()
    return render_template("edit.html", qa=row)


@app.route("/export")
def export_csv():
    conn = get_db()
    rows = conn.execute("SELECT frage, antwort FROM qa").fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Frage", "Antwort"])
    for row in rows:
        writer.writerow([row["frage"], row["antwort"]])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="qa_export.csv",
    )


def init_db():
    conn = get_db()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS qa (id INTEGER PRIMARY KEY AUTOINCREMENT, frage TEXT, antwort TEXT)"
    )
    conn.commit()


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
