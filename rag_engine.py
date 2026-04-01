import os
import sys
import sqlite3
import json
import numpy as np
import re
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from reader import extract_text_from_pdf, extract_text_from_docx
from chunker import clean_text, chunk_text

# =====================================================
# CACHE EN MEMORIA
# =====================================================

EMBEDDINGS_CACHE = {}
ANSWER_CACHE = {}

# =====================================================
# CARGAR VARIABLES DE ENTORNO
# =====================================================

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not XAI_API_KEY:
    raise Exception("Falta la variable de entorno XAI_API_KEY")

llm_client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1"
)

embedder = SentenceTransformer(EMBEDDING_MODEL)

# =====================================================
# RUTA BASE
# =====================================================

if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DB_NAME = os.path.join(BASE_PATH, "contracts.db")

# =====================================================
# TABLA EMBEDDINGS
# =====================================================

def init_embeddings_table():
    conn = sqlite3.connect(DB_NAME, timeout=10)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS contract_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        contract_id INTEGER,
        chunk_text TEXT,
        embedding TEXT
    )
    """)

    conn.commit()
    conn.close()

# =====================================================
# LEER CONTRATO
# =====================================================

def load_contract_text(path, filetype):
    if filetype == "pdf":
        text = extract_text_from_pdf(path)
    elif filetype == "docx":
        text = extract_text_from_docx(path)
    else:
        raise Exception("Tipo de archivo no soportado")

    text = clean_text(text)
    return text

# =====================================================
# EMBEDDINGS LOCALES
# =====================================================

def embed_texts(texts):
    if isinstance(texts, str):
        texts = [texts]

    vectors = embedder.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]

# =====================================================
# GUARDAR EMBEDDINGS
# =====================================================

def save_embeddings(contract_id, chunks, embeddings):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    cur = conn.cursor()

    for chunk, emb in zip(chunks, embeddings):
        cur.execute("""
        INSERT INTO contract_embeddings
        (contract_id, chunk_text, embedding)
        VALUES (?, ?, ?)
        """, (
            contract_id,
            chunk,
            json.dumps(emb)
        ))

    conn.commit()
    conn.close()

# =====================================================
# CARGAR EMBEDDINGS
# =====================================================

def load_embeddings(contract_id):
    if contract_id in EMBEDDINGS_CACHE:
        return EMBEDDINGS_CACHE[contract_id]

    conn = sqlite3.connect(DB_NAME, timeout=5)
    cur = conn.cursor()

    cur.execute("""
    SELECT chunk_text, embedding
    FROM contract_embeddings
    WHERE contract_id = ?
    """, (contract_id,))

    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None, None

    chunks = []
    embeddings = []

    for chunk_text, emb in rows:
        chunks.append(chunk_text)
        embeddings.append(json.loads(emb))

    EMBEDDINGS_CACHE[contract_id] = (chunks, embeddings)
    return chunks, embeddings

# =====================================================
# BORRAR EMBEDDINGS ANTIGUOS
# =====================================================

def delete_embeddings(contract_id):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    cur = conn.cursor()
    cur.execute("DELETE FROM contract_embeddings WHERE contract_id = ?", (contract_id,))
    conn.commit()
    conn.close()

    if contract_id in EMBEDDINGS_CACHE:
        del EMBEDDINGS_CACHE[contract_id]

# =====================================================
# COSINE SIMILARITY
# =====================================================

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =====================================================
# BUSQUEDA SEMANTICA
# =====================================================

def search_similar(question, chunks, chunk_embeddings, top_k=6):
    question_embedding = embed_texts(question)[0]
    scores = []

    for i, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(question_embedding, emb)
        scores.append((score, chunks[i]))

    scores.sort(reverse=True)
    return [chunk for score, chunk in scores[:top_k]]

# =====================================================
# EXTRAER FECHAS
# =====================================================

def extract_dates(text):
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{1,2}-\d{1,2}-\d{2,4}',
        r'\d{1,2} de [A-Za-záéíóúÁÉÍÓÚñÑ]+ de \d{4}',
        r'\d{4}-\d{2}-\d{2}'
    ]

    matches = []

    for p in patterns:
        matches += re.findall(p, text)

    return list(set(matches))

# =====================================================
# DETECTAR CLÁUSULAS CRÍTICAS
# =====================================================

def detect_critical_clauses(text):
    results = {
        "multas": [],
        "sla": [],
        "terminacion": [],
        "responsabilidad": [],
        "pagos": []
    }

    patterns = {
        "multas": r"(multa|penalidad|penalización).*?(\.|;)",
        "sla": r"(nivel de servicio|sla|disponibilidad).*?(\.|;)",
        "terminacion": r"(terminación|termino anticipado|resciliación).*?(\.|;)",
        "responsabilidad": r"(responsabilidad|limitación).*?(\.|;)",
        "pagos": r"(pago|facturación|precio).*?(\.|;)"
    }

    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        results[key] = [m[0] for m in matches[:5]]

    return results

# =====================================================
# PROMPT LEGAL
# =====================================================

def build_prompt(context, question):
    if "resumen ejecutivo" in question.lower():
        return f"""
Actúa como un abogado corporativo senior especializado en revisión de contratos.

Tu tarea es generar un RESUMEN EJECUTIVO COMPLETO, PRECISO y ESTRICTO basado EXCLUSIVAMENTE en el contenido del contrato.

========================
REGLAS OBLIGATORIAS
========================
- NO inventar información
- NO asumir datos no explícitos
- NO generalizar si el contrato contiene datos específicos
- SIEMPRE privilegiar datos textuales del contrato
- Si algo no existe, indicar exactamente:
"No se identifica esta información en el contrato"

========================
EXTRACCIÓN OBLIGATORIA DE FECHAS
========================
Debes identificar y reportar TODAS las fechas presentes en el contrato, incluyendo:
- Fecha de firma
- Fecha de inicio
- Fecha de término
- Plazos contractuales
- Plazos de pago
- Plazos de aviso
- Renovaciones
- Cualquier otra referencia temporal

Si NO hay fechas explícitas, indicar:
"No se identifican fechas explícitas en el contrato"

========================
FORMATO DE RESPUESTA
========================
RESUMEN EJECUTIVO DEL CONTRATO

FECHAS RELEVANTES
- Fecha de firma:
- Inicio de vigencia:
- Término:
- Renovación:
- Otros plazos relevantes:

1. Objeto del contrato
2. Partes involucradas
3. Alcance del servicio
4. Vigencia
5. Renovación
6. Terminación
7. Condiciones económicas
8. Obligaciones principales
9. Riesgos contractuales

CONTRATO:
{context}
"""

    return f"""
Actúa como un abogado corporativo experto en análisis de contratos.

Responde la pregunta usando EXCLUSIVAMENTE el contenido del contrato.

REGLAS:
- No inventar información
- No asumir
- No completar con conocimiento externo
- Si no está en el contrato responder EXACTAMENTE:
"No se encontró esa información en el contrato"

Si la pregunta involucra fechas:
- Debes buscar fechas explícitas en el contrato
- No responder con estimaciones
- No omitir fechas si existen

CONTRATO:
{context}

PREGUNTA:
{question}

RESPUESTA:
"""

# =====================================================
# CONSULTA LLM
# =====================================================

def ask_llm(context, question):
    prompt = build_prompt(context, question)

    extra_instructions = ""

    if "resumen ejecutivo" in question.lower():
        extra_instructions = """
Además del resumen, debes analizar e incluir:

1. CLÁUSULAS CRÍTICAS:
- Multas o penalidades (montos si existen)
- SLA / niveles de servicio (porcentajes, disponibilidad)
- Terminación anticipada
- Límites de responsabilidad
- Condiciones de pago

2. EVALUACIÓN DE RIESGO CONTRACTUAL:
Debes clasificar el contrato como:
- ALTO
- MEDIO
- BAJO

Y justificar en 2-3 líneas basado en:
- multas
- responsabilidad
- obligaciones
- condiciones económicas

3. FORMATO ADICIONAL OBLIGATORIO:

CLÁUSULAS CRÍTICAS
- Multas:
- SLA:
- Terminación:
- Responsabilidad:
- Pagos:

EVALUACIÓN DE RIESGO
Nivel:
Justificación:
"""

    final_prompt = extra_instructions + "\n\n" + prompt

    response = llm_client.chat.completions.create(
        model=GROK_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "Eres un abogado corporativo experto en contratos."},
            {"role": "user", "content": final_prompt}
        ]
    )

    return response.choices[0].message.content

# =====================================================
# MOTOR PRINCIPAL
# =====================================================

def ask_contract(question, contract_id, path, filetype, force_reembed=False):
    cache_key = f"{contract_id}:{question}"

    if cache_key in ANSWER_CACHE and not force_reembed:
        return ANSWER_CACHE[cache_key]

    init_embeddings_table()

    if not os.path.isabs(path):
        path = os.path.join(BASE_PATH, path)

    if force_reembed:
        delete_embeddings(contract_id)

    chunks, embeddings = load_embeddings(contract_id)

    if chunks is None or "resumen ejecutivo" in question.lower():
        text = load_contract_text(path, filetype)

        if not text:
            return "No se pudo extraer texto del contrato."

        if "resumen ejecutivo" in question.lower():
            context = text[:30000]
            answer = ask_llm(context, question)
            ANSWER_CACHE[cache_key] = answer
            return answer

    chunks, embeddings = load_embeddings(contract_id)

    if chunks is None:
        chunks = chunk_text(text, chunk_size=2000, overlap=300)
        embeddings = embed_texts(chunks)
        save_embeddings(contract_id, chunks, embeddings)

    relevant_chunks = search_similar(question, chunks, embeddings)
    context = "\n\n".join(relevant_chunks)

    answer = ask_llm(context, question)
    ANSWER_CACHE[cache_key] = answer
    return answer