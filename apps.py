from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
import threading
import main   # your code file (rename to attendance_system.py)

app = Flask(__name__)
app.secret_key = "secret123"  # for flash messages


@app.route("/")
def home():
    students = main.get_students()
    return render_template("index.html", students=students)


@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    if request.method == "POST":
        name = request.form["name"]
        reg_no = request.form["reg_no"]
        threading.Thread(target=main.enroll_student, args=(name, reg_no)).start()
        flash(f"Enrollment started for {name}. Please look at the camera.")
        return redirect(url_for("home"))
    return render_template("enroll.html")


@app.route("/train")
def train():
    threading.Thread(target=main.train_model).start()
    flash("Training started. Check console for progress.")
    return redirect(url_for("home"))


@app.route("/recognize")
def recognize():
    threading.Thread(target=main.recognize_loop).start()
    flash("Recognition started. Press 'q' in camera window to stop.")
    return redirect(url_for("home"))


@app.route("/export")
def export():
    out_file = "attendance_export.csv"
    main.export_attendance_csv(out_file)
    return send_file(out_file, as_attachment=True)


if __name__ == "__main__":
    main.init_db()
    app.run(debug=True)
