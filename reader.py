import sqlite3
from pypdf import PdfReader
from docx import Document
from config import DB_NAME




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


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def extract_text_from_docx(path):
    doc = Document(path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


if __name__ == "__main__":
    print("===== LECTOR DE CONTRATOS =====")
    contract_id = int(input("Ingrese ID del contrato: "))

    contract = get_contract_by_id(contract_id)

    if not contract:
        print("Contrato no encontrado.")
        exit()

    filename, path, filetype = contract
    print(f"\nLeyendo: {filename}\n")

    if filetype == "pdf":
        text = extract_text_from_pdf(path)
    elif filetype == "docx":
        text = extract_text_from_docx(path)
    else:
        print("Tipo no soportado.")
        exit()

    if not text.strip():
        print("⚠ No se pudo extraer texto (posible PDF escaneado).")
    else:
        print("===== TEXTO EXTRAÍDO (primeros 1000 caracteres) =====\n")
        print(text[:1000])