import re


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    from reader import get_contract_by_id, extract_text_from_pdf, extract_text_from_docx

    contract_id = int(input("Ingrese ID del contrato: "))
    contract = get_contract_by_id(contract_id)

    if not contract:
        print("Contrato no encontrado.")
        exit()

    filename, path, filetype = contract

    if filetype == "pdf":
        text = extract_text_from_pdf(path)
    else:
        text = extract_text_from_docx(path)

    text = clean_text(text)
    chunks = chunk_text(text)

    print(f"\nTotal chunks generados: {len(chunks)}\n")
    print("Primer chunk:\n")
    print(chunks[0])