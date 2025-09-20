import os
import cv2
import numpy as np
import sqlite3
import argparse
import pickle
from datetime import datetime, date
import pandas as pd

# ---------- CONFIG ----------
DATASET_DIR = "dataset"            # where face images will be saved
TRAINER_FILE = "trainer.yml"       # trained model
LABELS_FILE = "labels.pickle"      # mapping label->name
DB_FILE = "attendance.db"
ATTENDANCE_CSV = "attendance_log.csv"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_PER_PERSON = 30
IMG_WIDTH, IMG_HEIGHT = 200, 200
CONFIDENCE_THRESHOLD = 60.0  # lower -> stricter (0 best). LBPH gives distance-like values; tune for you.
# ----------------------------

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            reg_no TEXT UNIQUE,
            name TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            name TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_student_db(name, reg_no):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO students (reg_no, name) VALUES (?, ?)", (reg_no, name))
        conn.commit()
        student_id = c.lastrowid
    except sqlite3.IntegrityError:
        # reg_no exists -> get id
        c.execute("SELECT id FROM students WHERE reg_no = ?", (reg_no,))
        row = c.fetchone()
        student_id = row[0]
    conn.close()
    return student_id

def get_students():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, reg_no, name FROM students")
    rows = c.fetchall()
    conn.close()
    return rows

def mark_attendance(student_id, name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = datetime.now().isoformat(timespec='seconds')
    # Check last attendance for the student today
    c.execute("SELECT timestamp FROM attendance WHERE student_id = ? ORDER BY timestamp DESC LIMIT 1", (student_id,))
    row = c.fetchone()
    if row:
        last_ts = datetime.fromisoformat(row[0])
        if last_ts.date() == date.today():
            conn.close()
            return False  # already marked today
    c.execute("INSERT INTO attendance (student_id, name, timestamp) VALUES (?, ?, ?)", (student_id, name, now))
    conn.commit()
    conn.close()
    # append to CSV
    df_row = pd.DataFrame([{"student_id": student_id, "name": name, "timestamp": now}])
    if not os.path.exists(ATTENDANCE_CSV):
        df_row.to_csv(ATTENDANCE_CSV, index=False)
    else:
        df_row.to_csv(ATTENDANCE_CSV, mode='a', header=False, index=False)
    return True

def enroll_student(name, reg_no):
    student_id = add_student_db(name, reg_no)
    folder_name = f"{student_id}_{reg_no}_{name.replace(' ', '_')}"
    path = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print(f"[ENROLL] Capturing {IMG_PER_PERSON} face images for {name} (ID {student_id}). Press 'q' to quit early.")
    count = 0
    while count < IMG_PER_PERSON:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
            img_path = os.path.join(path, f"{str(count).zfill(3)}.jpg")
            cv2.imwrite(img_path, face_resized)
            count += 1
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{count}/{IMG_PER_PERSON}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            break
        cv2.imshow("Enroll - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images into {path}. Run `--train` to train the model.")

def train_model():
    print("[TRAIN] Scanning dataset and training LBPH recognizer...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}
    curr_label = 0

    for person_folder in os.listdir(DATASET_DIR) if os.path.exists(DATASET_DIR) else []:
        folder_path = os.path.join(DATASET_DIR, person_folder)
        if not os.path.isdir(folder_path): continue
        try:
            student_id_str, reg_no, name = person_folder.split("_", 2)
            student_id = int(student_id_str)
        except Exception:
            student_id = curr_label + 1
            name = person_folder
        label = student_id
        label_map[label] = f"{name} ({reg_no})" if '_' in person_folder else name
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            faces.append(img)
            labels.append(label)
        curr_label = max(curr_label, label)

    if not faces:
        print("No training data found. Enroll students first.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.write(TRAINER_FILE)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)
    print(f"[TRAIN] Training complete. Saved {TRAINER_FILE} and {LABELS_FILE}.")

def recognize_loop():
    if not os.path.exists(TRAINER_FILE) or not os.path.exists(LABELS_FILE):
        print("No trained model found. Run `--train` first.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    with open(LABELS_FILE, "rb") as f:
        label_map = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("[RECOGNIZE] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
            label, confidence = recognizer.predict(face_resized)

            if confidence < CONFIDENCE_THRESHOLD:
                display = label_map.get(label, f"ID:{label}")
                try:
                    student_id = int(label)
                except Exception:
                    student_id = None

                if student_id is not None:
                    marked = mark_attendance(student_id, display)
                    if marked:
                        note = "Present"
                        color = (0, 255, 0)   # Green
                    else:
                        note = "Already Present"
                        color = (0, 165, 255) # Orange
                else:
                    note = "Present"
                    color = (0, 255, 0)

                text = f"{display} ({confidence:.1f}) - {note}"
            else:
                text = f"Unknown ({confidence:.1f})"
                color = (0, 0, 255)  # Red

            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped. Attendance saved to DB and CSV.")

def export_attendance_csv(out="attendance_export.csv"):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM attendance ORDER BY timestamp DESC", conn)
    conn.close()
    df.to_csv(out, index=False)
    print(f"Exported attendance to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Attendance System (Face Recognition)")
    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument("--enroll", action="store_true", help="Enroll a new student (capture images + add to DB)")
    sub.add_argument("--train", action="store_true", help="Train the face recognizer from dataset")
    sub.add_argument("--recognize", action="store_true", help="Run recognition and mark attendance")
    sub.add_argument("--export", action="store_true", help="Export attendance DB to CSV")
    args = parser.parse_args()

    init_db()
    if args.enroll:
        name = input("Student name: ").strip()
        reg_no = input("Registration number (unique): ").strip()
        enroll_student(name, reg_no)
    elif args.train:
        train_model()
    elif args.recognize:
        recognize_loop()
    elif args.export:
        export_attendance_csv()
