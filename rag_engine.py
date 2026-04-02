import os
import sqlite3
import json
import numpy as np
import re
from dotenv import load_dotenv
from openai import OpenAI

from reader import extract_text_from_pdf, extract_text_from_docx
from chunker import clean_text, chunk_text



# =====================================================
# CACHE EN MEMORIA
# =====================================================

EMBEDDINGS_CACHE = {}
ANSWER_CACHE = {}

# =====================================================
# CARGAR VARIABLES DE ENTORNO (.env)
# =====================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =====================================================
# RUTA BASE
# =====================================================

import sys

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
# EMBEDDINGS OPENAI
# =====================================================

def embed_texts(texts):

    if isinstance(texts, str):
        texts = [texts]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [item.embedding for item in response.data]

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

    # 🔥 cache primero
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

    # 🔥 guardar en cache
    EMBEDDINGS_CACHE[contract_id] = (chunks, embeddings)

    return chunks, embeddings

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

def search_similar(question, chunks, chunk_embeddings, top_k=10):

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
        r'\d{1,2} de [A-Za-z]+ de \d{4}',
        r'\d{4}-\d{2}-\d{2}'
    ]

    matches = []

    for p in patterns:
        matches += re.findall(p, text)

    return list(set(matches))

# =====================================================
# 🔥 NUEVO: DETECTAR CLÁUSULAS CRÍTICAS
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
⚖️ REGLAS OBLIGATORIAS
========================

- NO inventar información
- NO asumir datos no explícitos
- NO generalizar si el contrato contiene datos específicos
- SIEMPRE privilegiar datos textuales del contrato
- Si algo no existe, indicar exactamente:
"No se identifica esta información en el contrato"

========================
📅 EXTRACCIÓN OBLIGATORIA DE FECHAS (CRÍTICO)
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

⚠️ PROHIBIDO:
- Omitir fechas si existen
- Reemplazar fechas por frases genéricas

Si NO hay fechas explícitas, indicar:
"No se identifican fechas explícitas en el contrato"

========================
📌 FORMATO DE RESPUESTA (OBLIGATORIO)
========================

**RESUMEN EJECUTIVO DEL CONTRATO**

**FECHAS RELEVANTES**
- Fecha de firma:
- Inicio de vigencia:
- Término:
- Renovación:
- Otros plazos relevantes:

1. **Objeto del contrato** 
2. **Partes involucradas** 
3. **Alcance del servicio** 
4. **Vigencia (incluir fechas si existen)** 
5. **Renovación (incluir plazos exactos)** 
6. **Terminación (causales y condiciones)** 
7. **Condiciones económicas (montos, moneda, plazos de pago)** 
8. **Obligaciones principales (por parte)** 
9. **Riesgos contractuales (límites de responsabilidad, multas, etc.)**

========================
✏️ NIVEL DE DETALLE MÍNIMO (OBLIGATORIO)
========================

- Para CADA ítem numerado del 1 al 9 debes redactar un párrafo desarrollado, no una sola frase.
- Cada ítem debe contener, como referencia, entre 6 y 10 oraciones completas.
- Siempre que existan en el contrato, incluye datos concretos: montos, porcentajes, plazos, nombres de cláusulas o secciones, y citas textuales breves entre comillas.
- Evita frases genéricas como "El contrato regula el servicio"; describe qué servicio, con qué alcances y bajo qué condiciones según el texto.
- Si en un ítem no se encuentra información suficiente en el contrato, indícalo explícitamente y explica qué buscaste (por ejemplo: "No se identifican cláusulas específicas sobre X tras revisar las secciones de objeto, obligaciones y condiciones económicas").

El resumen ejecutivo completo debe ser extenso y detallado: evita respuestas telegráficas o con menos de una página equivalente de texto cuando la información del contrato lo permita.

========================
📄 CONTRATO
========================
{context}
"""

    return f"""
Actúa como un abogado corporativo experto en análisis de contratos.

Responde la pregunta usando EXCLUSIVAMENTE el contenido del contrato.

========================
⚖️ REGLAS
========================

- No inventar información
- No asumir
- Escribir por lo menos 6-7 líneas con información referente a la consulta
- No completar con conocimiento externo
- Si no está en el contrato responder EXACTAMENTE:
"No se encontró esa información en el contrato"

========================
📅 FECHAS (IMPORTANTE)
========================

Si la pregunta involucra fechas:
- Debes buscar fechas explícitas en el contrato
- No responder con estimaciones
- No omitir fechas si existen

========================
📄 CONTRATO
========================
{context}

========================
❓ PREGUNTA
========================
{question}

========================
🧾 RESPUESTA
========================
"""
# =====================================================
# CACHE RESPUESTAS
# =====================================================

ANSWER_CACHE = {}

# =====================================================
# CONSULTA LLM
# =====================================================

def ask_llm(context, question):

    prompt = build_prompt(context, question)

    # =====================================================
    # 🔥 INSTRUCCIONES ADICIONALES (NO MODIFICA TU PROMPT)
    # =====================================================

    extra_instructions = ""
    max_output_tokens = 2500

    if "resumen ejecutivo" in question.lower():

        extra_instructions = """
========================
⚠️ ANÁLISIS LEGAL AVANZADO (OBLIGATORIO)
========================

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

========================
✏️ DETALLE DE ANÁLISIS (OBLIGATORIO)
========================

- En CLÁUSULAS CRÍTICAS, bajo cada viñeta (Multas, SLA, etc.), redacta al menos 4 a 6 oraciones explicando:
  - Dónde aparece en el contrato (cláusula o sección si está indicada).
  - El contenido preciso (montos, porcentajes, plazos, condiciones específicas).
  - Qué implicancias tiene para las partes.
- En EVALUACIÓN DE RIESGO, la justificación debe tener al menos 5 oraciones, conectando explícitamente:
  - Multas y penalidades.
  - Límites de responsabilidad.
  - Condiciones de pago y obligaciones de servicio.
  - Cualquier otro factor contractual relevante identificado en el texto.


**CLÁUSULAS CRÍTICAS**
- Multas:
- SLA:
- Terminación:
- Responsabilidad:
- Pagos:

**EVALUACIÓN DE RIESGO**
Nivel:
Justificación:

⚠️ NO omitir estas secciones aunque no haya información.
"""

    final_prompt = extra_instructions + "\n\n" + prompt

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=4000,  # <<< NUEVO
        messages=[
            {"role": "system", "content": "Eres un abogado corporativo experto en contratos."},
            {"role": "user", "content": final_prompt}
        ]
    )

    return response.choices[0].message.content


# =====================================================
# MOTOR PRINCIPAL
# =====================================================

def ask_contract(question, contract_id, path, filetype):
    cache_key = f"{contract_id}:{question}"

    if cache_key in ANSWER_CACHE:
        return ANSWER_CACHE[cache_key]

    init_embeddings_table()

    # convertir ruta relativa en absoluta
    if not os.path.isabs(path):
        path = os.path.join(BASE_PATH, path)

    chunks, embeddings = load_embeddings(contract_id)

    # 🔥 SOLO cargar texto si no hay embeddings o es resumen
    if chunks is None or "resumen ejecutivo" in question.lower():

        text = load_contract_text(path, filetype)

        if not text:
            return "No se pudo extraer texto del contrato."


    # -------------------------------------------------
    # RESUMEN EJECUTIVO → TEXTO COMPLETO
    # -------------------------------------------------

    if "resumen ejecutivo" in question.lower():

        context = text[:30000]

        return ask_llm(context, question)

    # -------------------------------------------------
    # PREGUNTAS NORMALES → RAG
    # -------------------------------------------------

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
