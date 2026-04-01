import sqlite3

conn = sqlite3.connect("contracts.db")
cursor = conn.cursor()

cursor.execute("""
UPDATE contracts
SET path = REPLACE(
    path,
    'C:\Users\CFBAYOLO\OneDrive - Telefonica\CONTRATOS FIRMADOS\',
    ''
)
""")

conn.commit()
conn.close()

print("Rutas corregidas correctamente")