import sqlite3
import os
from config import DB_NAME




def search_contracts(keyword):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    query = """
        SELECT id, filename
        FROM contracts
        WHERE LOWER(filename) LIKE ?
        ORDER BY filename ASC
    """

    cursor.execute(query, (f"%{keyword.lower()}%",))
    results = cursor.fetchall()

    conn.close()
    return results


def get_contract_by_id(contract_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT filename, path, filetype
        FROM contracts
        WHERE id = ?
    """, (contract_id,))

    result = cursor.fetchone()
    conn.close()
    return result


if __name__ == "__main__":
    print("===== BUSCADOR DE CONTRATOS =====")
    keyword = input("Ingrese palabra clave: ")

    results = search_contracts(keyword)

    if not results:
        print("\nNo se encontraron contratos.")
        exit()

    print(f"\nSe encontraron {len(results)} resultados:\n")

    for r in results:
        print(f"[{r[0]}] {r[1]}")

    try:
        selected_id = int(input("\nSeleccione ID del contrato: "))
    except ValueError:
        print("ID inválido.")
        exit()

    contract = get_contract_by_id(selected_id)

    if contract:
        filename, path, filetype = contract
        print("\n===== CONTRATO SELECCIONADO =====")
        print(f"Nombre: {filename}")
        print(f"Ruta: {path}")
        print(f"Tipo: {filetype}")
    else:
        print("No se encontró el contrato.")