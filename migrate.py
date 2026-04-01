import sqlite3
import psycopg2
import json

# -------- CONFIG --------
SQLITE_DB = "contracts.db"

POSTGRES_URL = "postgresql://postgres:ZSQ69RqjSLePl1TP@db.jencrgqchkvpwrunitcw.supabase.co:5432/postgres"

# -------- CONEXIONES --------
sqlite_conn = sqlite3.connect(SQLITE_DB)
sqlite_cur = sqlite_conn.cursor()

pg_conn = psycopg2.connect(POSTGRES_URL)
pg_cur = pg_conn.cursor()

# -------- MIGRAR CONTRACTS --------
print("Migrando contracts...")

sqlite_cur.execute("SELECT id, filename, path, filetype FROM contracts")
rows = sqlite_cur.fetchall()

for r in rows:
    pg_cur.execute("""
        INSERT INTO contracts (id, filename, path, filetype)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """, r)

pg_conn.commit()

# -------- MIGRAR LOGS --------
print("Migrando logs...")

sqlite_cur.execute("""
SELECT username, contract_id, contract_name, question, query_type, response_time
FROM logs
""")

rows = sqlite_cur.fetchall()

for r in rows:
    pg_cur.execute("""
        INSERT INTO logs (username, contract_id, contract_name, question, query_type, response_time)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, r)

pg_conn.commit()

# -------- MIGRAR EMBEDDINGS --------
print("Migrando embeddings...")

sqlite_cur.execute("""
SELECT contract_id, chunk_text, embedding
FROM contract_embeddings
""")

rows = sqlite_cur.fetchall()

for r in rows:
    pg_cur.execute("""
        INSERT INTO contract_embeddings (contract_id, chunk_text, embedding)
        VALUES (%s, %s, %s)
    """, (
        r[0],
        r[1],
        json.loads(r[2])
    ))

pg_conn.commit()

# -------- CIERRE --------
sqlite_conn.close()
pg_conn.close()

print("✅ MIGRACIÓN COMPLETA")