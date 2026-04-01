import os
import sys
import threading
import webbrowser
import sqlite3
import getpass
import time
import subprocess
import psycopg2
from flask import Flask, render_template, request, jsonify, session
from rag_engine import ask_contract

# =====================================================
# PATH BASE
# =====================================================

if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DB_NAME = os.path.join(BASE_PATH, "contracts.db")
LOG_DB = os.path.join(BASE_PATH, "logs.db")
TEMPLATES_PATH = os.path.join(BASE_PATH, "templates")

# =====================================================
# FLASK
# =====================================================

app = Flask(__name__, template_folder=TEMPLATES_PATH)
app.secret_key = "super_secret_key"


# =====================================================
def get_connection():
    db_url = os.getenv("DATABASE_URL")

    if db_url:
        return psycopg2.connect(db_url)
    else:
        return sqlite3.connect(DB_NAME)

# =====================================================
# INIT LOGS
# =====================================================
def kill_port(port):
    try:
        result = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()

        lines = result.strip().split("\n")
        pids = set()

        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                pid = parts[-1]
                if pid.isdigit():
                    pids.add(pid)

        for pid in pids:
            print(f"🔪 Matando PID {pid} en puerto {port}")
            subprocess.call(f'taskkill /PID {pid} /F', shell=True)

    except subprocess.CalledProcessError:
        print(f"✅ Puerto {port} libre")

def init_logs_db():
    conn = sqlite3.connect(LOG_DB, timeout=10)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            contract_id INTEGER,
            contract_name TEXT,
            question TEXT,
            query_type TEXT,
            response_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

init_logs_db()

# =====================================================
# SEARCH CORE
# =====================================================

def search_contracts(keyword):
    conn = get_connection()
    cursor = conn.cursor()

    if os.getenv("DATABASE_URL"):
        # Supabase (PostgreSQL)
        cursor.execute("""
            SELECT id, filename
            FROM contracts
            WHERE filename ILIKE %s
            ORDER BY filename ASC
        """, (f"%{keyword}%",))
    else:
        # SQLite (como lo tenías)
        cursor.execute("""
            SELECT id, filename
            FROM contracts
            WHERE LOWER(filename) LIKE ?
            ORDER BY filename ASC
        """, (f"%{keyword.lower()}%",))

    results = cursor.fetchall()
    conn.close()

    return [{"id": r[0], "filename": r[1]} for r in results]

# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def index():
    return render_template("index.html")

# 🔥 FIX CRÍTICO AQUÍ
@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json(silent=True) or {}
        keyword = data.get("keyword", "").strip()

        print("KEYWORD RECIBIDO:", keyword)

        results = search_contracts(keyword)

        return jsonify(results)

    except Exception as e:
        print("🔥 ERROR SEARCH REAL:", str(e))  # <-- CLAVE
        return jsonify({"error": str(e)}), 500

@app.route("/select", methods=["POST"])
def select_contract():
    try:
        data = request.get_json(silent=True) or {}

        contract_id = data.get("id")
        filename = data.get("filename")

        session["contract_id"] = contract_id
        session["contract_name"] = filename

        return jsonify({"status": "ok"})

    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})

@app.route("/ask", methods=["POST"])
def ask():

    start_time = time.time()

    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()

        contract_id = session.get("contract_id")
        contract_name = session.get("contract_name")

        if not contract_id:
            return jsonify({"answer": "Debe seleccionar un contrato primero."})

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT path, filetype FROM contracts WHERE id = %s",
            (contract_id,)
        )

        result = cursor.fetchone()
        conn.close()

        if not result:
            return jsonify({"answer": "Contrato no encontrado."})

        path, filetype = result

        from config import CONTRACTS_FOLDER

        # 🔥 NORMALIZAR PATH (Windows → Render)
        filename_only = os.path.basename(path)

        full_path = os.path.join(CONTRACTS_FOLDER, filename_only)
   

        answer = ask_contract(
            question,
            contract_id,
            full_path,
            filetype
        )

        response_time = round(time.time() - start_time, 2)

        query_type = "resumen" if "resumen ejecutivo" in question.lower() else "pregunta"

        username = getpass.getuser()

        conn = sqlite3.connect(LOG_DB, timeout=10)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO logs
            (username, contract_id, contract_name, question, query_type, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            username,
            contract_id,
            contract_name,
            question,
            query_type,
            response_time
        ))

        conn.commit()
        conn.close()

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": f"Error IA: {str(e)}"})

# =====================================================
# AUTO OPEN
# =====================================================

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    import os

    port = int(os.environ.get("PORT", 5000))

    # SOLO en local (Windows)
    if os.name == "nt":
        try:
            kill_port(port)
            threading.Timer(1.5, open_browser).start()
        except:
            pass

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )