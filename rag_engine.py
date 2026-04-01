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
# CACHE
# =====================================================

EMBEDDINGS_CACHE = {}
ANSWER_CACHE = {}

# =====================================================
# CONFIG
# =====================================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 6
SIM_THRESHOLD = 0.75
MAX_CONTEXT_CHARS = 8000

# =====================================================
# ENV
# =====================================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =====================================================
# PATH
# =====================================================

import sys

if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DB_NAME = os.path.join(BASE_PATH, "contracts.db")

# =====================================================
# DB
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
# LOAD CONTRACT
# =====================================================

def load_contract_text(path, filetype):

    if filetype == "pdf":
        text = extract_text_from_pdf(path)
    elif filetype == "docx":
        text = extract_text_from_docx(path)
    else:
        raise Exception("Tipo de archivo no soportado")

    return clean_text(text)

# =====================================================
# EMBEDDINGS
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
# SAVE / LOAD
# =====================================================

def save_embeddings(contract_id, chunks, embeddings):

    conn = sqlite3.connect(DB_NAME, timeout=10)
    cur = conn.cursor()

    for chunk, emb in zip(chunks, embeddings):
        cur.execute("""
            INSERT INTO contract_embeddings
            (contract_id, chunk_text, embedding)
            VALUES (?, ?, ?)
        """, (contract_id, chunk, json.dumps(emb)))

    conn.commit()
    conn.close()

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

    chunks, embeddings = [], []

    for chunk_text, emb in rows:
        chunks.append(chunk_text)
        embeddings.append(json.loads(emb))

    EMBEDDINGS_CACHE[contract_id] = (chunks, embeddings)

    return chunks, embeddings

# =====================================================
# SIMILARITY
# =====================================================

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =====================================================
# KEYWORD BOOST
# =====================================================

def keyword_score(text, question):
    words = question.lower().split()
    return sum(1 for w in words if w in text.lower())

# =====================================================
# SEARCH
# =====================================================

def search_similar(question, chunks, chunk_embeddings):

    question_embedding = embed_texts(question)[0]

    scores = []

    for i, emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(question_embedding, emb)
        kw = keyword_score(chunks[i], question)
        score = sim + (kw * 0.05)
        scores.append((score, chunks[i]))

    scores.sort(reverse=True)

    filtered = [chunk for score, chunk in scores if score >= SIM_THRESHOLD]

    return filtered[:TOP_K]

# =====================================================
# CONTEXT
# =====================================================

def build_context(chunks):

    context = ""

    for i, chunk in enumerate(chunks):
        tagged = f"[CHUNK_{i}]\n{chunk}\n"
        
        if len(context) + len(tagged) > MAX_CONTEXT_CHARS:
            break

        context += tagged + "\n"

    return context

# =====================================================
# 🔎 PROMPT QA (PREGUNTAS) — EXACTO
# =====================================================

def build_qa_prompt(context, question):

    return f"""
Actúa como un abogado corporativo senior especialista en contratos.

Debes responder la pregunta utilizando EXCLUSIVAMENTE el contenido del contrato.

========================
⚖️ REGLAS ESTRICTAS
========================

- Prohibido inventar información
- Prohibido usar conocimiento externo
- Prohibido inferir información no explícita
- Solo puedes usar texto contenido en el contrato

Si la información no está explícitamente presente, responde EXACTAMENTE:
"No se encontró esa información en el contrato"

========================
📌 METODOLOGÍA (OBLIGATORIA)
========================

1. Identificar el fragmento más relevante del contrato
2. Verificar que contiene la respuesta explícita
3. Responder de forma directa y precisa (sin relleno)

========================
📄 FORMATO DE RESPUESTA (OBLIGATORIO)
========================

**RESPUESTA:**
Respuesta concreta y específica a la pregunta.

**EVIDENCIA:**

Debes citar el texto EXACTO junto con su identificador de origen.

Formato obligatorio:
[CHUNK_X]
"Texto exacto del contrato..."

⚠️ No se permite citar sin indicar el CHUNK correspondiente

Las citas deben ser sustantivas y contener información relevante.

- Prohibido citar frases genéricas o introductorias
- Prohibido citar fragmentos incompletos sin contexto
- La cita debe incluir datos concretos (fechas, montos, condiciones, obligaciones)

Ejemplo incorrecto:
"Por el presente acto..."

Ejemplo correcto:
"La duración del presente Contrato será de 1 año, contado a partir de la fecha del presente instrumento, renovable automática e indefinidamente..."

Si no existe una cita sustantiva, indicar:
"No se identifica evidencia suficiente en el contrato"

========================
📄 CONTRATO
========================
{context}

========================
❓ PREGUNTA
========================
{question}
"""

# =====================================================
# 🧾 PROMPT RESUMEN — EXACTO
# =====================================================

def build_summary_prompt(context):

    return f"""
Actúa como un abogado corporativo senior especializado en análisis de contratos complejos.

Tu objetivo es elaborar un RESUMEN EJECUTIVO completo, preciso y estructurado,
basado EXCLUSIVAMENTE en el contenido del contrato.

========================
⚖️ REGLAS ESTRICTAS
========================

- Prohibido inventar información
- Prohibido inferir o asumir datos no explícitos
- No omitir información relevante si está presente
- Si una sección no tiene información, indicar explícitamente:
  "No se identifica esta información en el contrato"

========================
📅 EXTRACCIÓN OBLIGATORIA DE FECHAS
========================

Debes identificar TODAS las fechas explícitas, incluyendo:

- Firma
- Inicio de vigencia
- Término
- Renovación
- Plazos de pago
- Avisos
- Multas o SLA asociados a tiempo

⚠️ Prohibido reemplazar fechas por descripciones vagas

========================
📌 METODOLOGÍA
========================

1. Analizar múltiples fragmentos del contrato
2. Consolidar información sin perder precisión
3. Priorizar datos concretos (fechas, montos, porcentajes)
4. Mantener lenguaje jurídico claro y directo (sin relleno)

========================
📄 FORMATO OBLIGATORIO
========================

**RESUMEN EJECUTIVO DEL CONTRATO**

**FECHAS RELEVANTES**
- Firma:
- Inicio:
- Término:
- Renovación:
- Otros plazos:

1. **Objeto del contrato**
2. **Partes involucradas**
3. **Alcance del servicio**

4. **Condiciones económicas**
- Precio:
- Moneda:
- Plazos de pago:

5. **Vigencia y renovación**
- Detalle completo con fechas

6. **Terminación**
- Causales
- Condiciones

7. **Cláusulas críticas**
- Multas / penalidades
- SLA / niveles de servicio
- Responsabilidad / limitaciones

8. **Riesgos contractuales**
- Identificación clara de riesgos
- Justificación breve basada en el contrato

========================
📌 EVIDENCIA (OBLIGATORIA)
========================
Debes citar el texto EXACTO junto con su identificador de origen.

Formato obligatorio:
[CHUNK_X]
"Texto exacto del contrato..."

⚠️ No se permite citar sin indicar el CHUNK correspondiente

Las citas deben ser sustantivas y contener información relevante.

- Prohibido citar frases genéricas o introductorias
- Prohibido citar fragmentos incompletos sin contexto
- La cita debe incluir datos concretos (fechas, montos, condiciones, obligaciones)

Ejemplo incorrecto:
"Por el presente acto..."

Ejemplo correcto:
"La duración del presente Contrato será de 1 año, contado a partir de la fecha del presente instrumento, renovable automática e indefinidamente..."

Si no existe una cita sustantiva, indicar:
"No se identifica evidencia suficiente en el contrato"

========================
📄 CONTRATO
========================
{context}
"""

# =====================================================
# FECHAS + CLÁUSULAS
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
# RERANK
# =====================================================

def rerank_chunks(question, chunks):

    joined = "\n\n".join([f"[{i}] {c[:300]}" for i, c in enumerate(chunks)])

    prompt = f"""
Selecciona los 3 fragmentos MÁS relevantes para responder la pregunta.

Pregunta:
{question}

Fragmentos:
{joined}

Responde SOLO con los números separados por coma.
Ejemplo: 0,2,4
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content

    try:
        idxs = [int(x.strip()) for x in content.split(",")]
        return [chunks[i] for i in idxs if i < len(chunks)]
    except:
        return chunks[:3]

# =====================================================
# VALIDATION
# =====================================================

def validate_answer(answer):

    if "EVIDENCIA" not in answer:
        return False

    if "[CHUNK_" not in answer:
        return False

    lines = answer.split("\n")
    evidence_lines = [l for l in lines if '"' in l]

    if not evidence_lines:
        return False

    for line in evidence_lines:
        if len(line.strip()) < 40:
            return False

    return True

# =====================================================
# LLM
# =====================================================

def ask_llm(prompt):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Abogado experto en contratos"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# =====================================================
# MAIN
# =====================================================

def ask_contract(question, contract_id, path, filetype):

    cache_key = f"{contract_id}:{hash(question)}"

    if cache_key in ANSWER_CACHE:
        return ANSWER_CACHE[cache_key]

    init_embeddings_table()

    if not os.path.isabs(path):
        path = os.path.join(BASE_PATH, path)

    chunks, embeddings = load_embeddings(contract_id)

    if chunks is None:
        text = load_contract_text(path, filetype)

        if not text:
            return "No se pudo extraer texto del contrato."

        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        embeddings = embed_texts(chunks)
        save_embeddings(contract_id, chunks, embeddings)

    # =========================
    # RESUMEN
    # =========================

    if "resumen ejecutivo" in question.lower():

        text = load_contract_text(path, filetype)

        fechas = extract_dates(text)
        clausulas = detect_critical_clauses(text)

        extra = "\n\n=== DATOS EXTRAÍDOS AUTOMÁTICAMENTE ===\n"

        extra += "\nFECHAS DETECTADAS:\n"
        if fechas:
            extra += "\n".join(f"- {f}" for f in fechas)
        else:
            extra += "No se detectaron fechas"

        extra += "\n\nCLÁUSULAS CRÍTICAS:\n"
        for k, v in clausulas.items():
            extra += f"\n{k.upper()}:\n"
            if v:
                for item in v:
                    extra += f"- {item}\n"
            else:
                extra += "- No detectado\n"

        context = text[:25000] + extra

        prompt = build_summary_prompt(context)
        answer = ask_llm(prompt)

        ANSWER_CACHE[cache_key] = answer
        return answer

    # =========================
    # QA
    # =========================

    relevant_chunks = search_similar(question, chunks, embeddings)
    relevant_chunks = rerank_chunks(question, relevant_chunks)

    if not relevant_chunks:
        return "No se encontró información relevante en el contrato."

    context = build_context(relevant_chunks)

    prompt = build_qa_prompt(context, question)
    answer = ask_llm(prompt)

    if not validate_answer(answer):
        answer = ask_llm(prompt)

        if not validate_answer(answer):
            return "No se pudo generar una respuesta con evidencia suficiente del contrato."

    ANSWER_CACHE[cache_key] = answer

    return answer