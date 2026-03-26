import sqlite3
import os

conn = sqlite3.connect("contracts.db")
cursor = conn.cursor()

# Crear tabla
cursor.execute("""
CREATE TABLE IF NOT EXISTS contracts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    path TEXT,
    filetype TEXT
)
""")

# Insertar archivos desde /contracts
for file in os.listdir("contracts"):
    if file.lower().endswith((".pdf", ".docx")):
        cursor.execute(
            "INSERT INTO contracts (filename, path, filetype) VALUES (?, ?, ?)",
            (file, file, file.split(".")[-1])
        )

conn.commit()
conn.close()

print("OK - contratos cargados")