import os
import sqlite3
import time
from datetime import datetime

# ========= CONFIG =========
from config import CONTRACTS_FOLDER, DB_NAME
# ==========================


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            path TEXT UNIQUE,
            filetype TEXT,
            modified_time REAL,
            indexed_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def index_contracts():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    total_files = 0
    new_files = 0
    updated_files = 0

    for root, dirs, files in os.walk(CONTRACTS_FOLDER):
        for file in files:
            if file.lower().endswith((".pdf", ".docx")):

                total_files += 1
                full_path = os.path.join(root, file)
                modified_time = os.path.getmtime(full_path)
                filetype = file.split(".")[-1].lower()

                cursor.execute("SELECT modified_time FROM contracts WHERE path = ?", (full_path,))
                result = cursor.fetchone()

                if result is None:
                    # Nuevo archivo
                    cursor.execute("""
                        INSERT INTO contracts (filename, path, filetype, modified_time, indexed_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        file,
                        full_path,
                        filetype,
                        modified_time,
                        datetime.now().isoformat()
                    ))
                    new_files += 1

                else:
                    # Verificar si fue modificado
                    if result[0] != modified_time:
                        cursor.execute("""
                            UPDATE contracts
                            SET filename = ?, filetype = ?, modified_time = ?, indexed_at = ?
                            WHERE path = ?
                        """, (
                            file,
                            filetype,
                            modified_time,
                            datetime.now().isoformat(),
                            full_path
                        ))
                        updated_files += 1

    conn.commit()
    conn.close()

    print("===== INDEXACIÓN COMPLETADA =====")
    print(f"Total encontrados: {total_files}")
    print(f"Nuevos: {new_files}")
    print(f"Actualizados: {updated_files}")
    print("==================================")


if __name__ == "__main__":
    start = time.time()
    init_db()
    index_contracts()
    print(f"Tiempo ejecución: {round(time.time() - start, 2)} segundos")