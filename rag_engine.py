# -*- coding: utf-8 -*-
"""
rag_engine.py — Motor RAG para análisis de contratos (PDF/DOCX)

Objetivos (según requerimiento):
- Separar flujo de "pregunta individual" vs "resumen ejecutivo".
- Responder SOLO con base en fragmentos recuperados (RAG).
- Exigir evidencia textual literal desde los fragmentos (no inventar).
- Pregunta individual: Formato A (respuesta + evidencia al final).
- Resumen ejecutivo: Formato estructurado con evidencia en cada ítem.
- Persistir embeddings en SQLite por contract_id.
"""

import os
import sys
import json
import sqlite3
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from reader import extract_text_from_pdf, extract_text_from_docx
from chunker import clean_text, chunk_text


# =====================================================
# CONFIG
# =====================================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

# Temperatura baja para minimizar desviaciones / alucinaciones.
CHAT_TEMPERATURE = 0.1

# Recuperación
DEFAULT_TOP_K_INDIVIDUAL = 10
DEFAULT_TOP_K_MULTIQUERY_PER_Q = 6

# Umbral mínimo de similitud: si no supera, se considera "sin evidencia suficiente"
DEFAULT_MIN_SCORE = 0.62

# Límite de contexto por caracteres (evita prompts excesivamente largos)
MAX_CONTEXT_CHARS_INDIVIDUAL = 18000
MAX_CONTEXT_CHARS_SUMMARY = 38000

# Cache en memoria
EMBEDDINGS_CACHE: Dict[int, Tuple[List[str], List[List[float]]]] = {}
ANSWER_CACHE: Dict[str, str] = {}

# Rutas
if getattr(sys, "frozen", False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DB_NAME = os.path.join(BASE_PATH, "contracts.db")


# =====================================================
# BASE DE DATOS
# =====================================================

def init_embeddings_table() -> None:
    conn = sqlite3.connect(DB_NAME, timeout=10)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS contract_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    # Índice recomendado para acelerar consultas por contract_id
    cur.execute("CREATE INDEX IF NOT EXISTS idx_contract_embeddings_contract_id ON contract_embeddings(contract_id)")
    conn.commit()
    conn.close()


def save_embeddings(contract_id: int, chunks: List[str], embeddings: List[List[float]]) -> None:
    conn = sqlite3.connect(DB_NAME, timeout=10)
    cur = conn.cursor()

    rows = [(contract_id, chunk, json.dumps(emb)) for chunk, emb in zip(chunks, embeddings)]
    cur.executemany(
        """
        INSERT INTO contract_embeddings (contract_id, chunk_text, embedding)
        VALUES (?, ?, ?)
        """,
        rows
    )
    conn.commit()
    conn.close()

    EMBEDDINGS_CACHE[contract_id] = (chunks, embeddings)


def load_embeddings(contract_id: int) -> Tuple[Optional[List[str]], Optional[List[List[float]]]]:
    # Cache primero
    if contract_id in EMBEDDINGS_CACHE:
        return EMBEDDINGS_CACHE[contract_id]

    conn = sqlite3.connect(DB_NAME, timeout=5)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_text, embedding
        FROM contract_embeddings
        WHERE contract_id = ?
        """,
        (contract_id,)
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None, None

    chunks = []
    embeddings = []
    for chunk_text, emb_json in rows:
        chunks.append(chunk_text)
        embeddings.append(json.loads(emb_json))

    EMBEDDINGS_CACHE[contract_id] = (chunks, embeddings)
    return chunks, embeddings


# =====================================================
# CARGA Y PREPROCESO DEL CONTRATO
# =====================================================

def load_contract_text(path: str, filetype: str) -> str:
    if filetype == "pdf":
        text = extract_text_from_pdf(path)
    elif filetype == "docx":
        text = extract_text_from_docx(path)
    else:
        raise ValueError("Tipo de archivo no soportado. Use 'pdf' o 'docx'.")

    return clean_text(text or "")


# =====================================================
# EMBEDDINGS
# =====================================================

def embed_texts(texts: List[str], batch_size: int = 96) -> List[List[float]]:
    """
    Genera embeddings en lotes para evitar límites del API.
    """
    if isinstance(texts, str):
        texts = [texts]

    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        res = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in res.data])

    return all_embeddings


# =====================================================
# SIMILITUD Y BÚSQUEDA (RAG)
# =====================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def search_similar(
    query: str,
    chunks: List[str],
    chunk_embeddings: List[List[float]],
    top_k: int = DEFAULT_TOP_K_INDIVIDUAL,
    min_score: float = DEFAULT_MIN_SCORE
) -> List[Tuple[float, str]]:
    """
    Retorna lista de (score, chunk_text) filtrada por min_score, ordenada desc.
    """
    q_emb = embed_texts([query])[0]
    scored: List[Tuple[float, str]] = []

    for ch, emb in zip(chunks, chunk_embeddings):
        s = cosine_similarity(q_emb, emb)
        if s >= min_score:
            scored.append((s, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def multiconsulta_chunks(
    chunks: List[str],
    embeddings: List[List[float]],
    min_score: float = DEFAULT_MIN_SCORE
) -> List[str]:
    """
    Multi-query para resumen ejecutivo: recupera muchos fragmentos cubriendo aspectos clave.
    Devuelve fragmentos únicos (manteniendo orden de aparición en la recolección).
    """
    queries = [
        "fechas contrato vigencia inicio termino plazo duración renovación",
        "partes contrato proveedor cliente empresa representante domicilio",
        "objeto del contrato servicios productos descripción alcance entregables",
        "condiciones económicas pagos precio facturación moneda impuestos plazos de pago",
        "terminación contrato causales aviso anticipado término anticipado rescisión",
        "responsabilidad limite responsabilidad daños indirectos multa penalidad indemnización",
        "SLA nivel de servicio disponibilidad tiempos de respuesta continuidad",
        "confidencialidad datos personales privacidad seguridad de la información",
        "garantías propiedad intelectual licencias cesión derechos",
    ]

    collected: List[str] = []
    seen = set()

    for q in queries:
        top = search_similar(q, chunks, embeddings, top_k=DEFAULT_TOP_K_MULTIQUERY_PER_Q, min_score=min_score)
        for _, ch in top:
            if ch not in seen:
                collected.append(ch)
                seen.add(ch)

    return collected


def clip_context_by_chars(fragments: List[str], max_chars: int) -> List[str]:
    """
    Recorta lista de fragmentos para no exceder max_chars en el contexto.
    Mantiene el orden.
    """
    out = []
    total = 0
    for fr in fragments:
        if total + len(fr) + 10 > max_chars:
            break
        out.append(fr)
        total += len(fr) + 10
    return out


# =====================================================
# PROMPTS (OBLIGATORIOS) — EXACTAMENTE LOS DOS QUE PEDISTE
# =====================================================

SUMMARY_PROMPT_REQUIRED = """Usted actuará como un abogado corporativo senior, especializado en contratación comercial y análisis de riesgo contractual.

A partir de los FRAGMENTOS DE CONTRATO proporcionados, deberá elaborar un RESUMEN EJECUTIVO exhaustivo, con un enfoque claro, preciso y estrictamente basado en el texto disponible. 

No puede inferir, suponer, completar información ni utilizar conocimientos externos. Si un dato no está textual o explícitamente disponible en los fragmentos, la única respuesta permitida es: 
"Esta información no se encuentra en los fragmentos proporcionados".

INSTRUCCIÓN CRÍTICA SOBRE EVIDENCIA:
Para cada ítem solicitado, usted debe proporcionar evidencia textual literal tomada exclusivamente de los fragmentos. No puede modificarla, parafrasearla ni sintetizarla. Si no existe evidencia aplicable, debe indicar:
"No existe evidencia textual disponible para este ítem en los fragmentos entregados".

FORMATO OBLIGATORIO DEL RESUMEN EJECUTIVO

I. Fechas Relevantes
   - Fecha de firma:
     Evidencia:
   - Inicio de vigencia:
     Evidencia:
   - Término:
     Evidencia:
   - Renovación:
     Evidencia:
   - Otros plazos relevantes:
     Evidencia:

II. Objeto del Contrato
    Evidencia:

III. Partes que Intervienen
     Evidencia:

IV. Alcance y Servicios/Obligaciones del Proveedor
     Evidencia:

V. Vigencia Contractual
    Evidencia:

VI. Mecanismos de Renovación
    Evidencia:

VII. Causales y Condiciones de Terminación
     Evidencia:

VIII. Condiciones Económicas
      Evidencia:

IX. Obligaciones Principales
    Evidencia:

X. Aspectos de Riesgo Contractual (responsabilidad, penalidades, SLA, limitaciones)
   Evidencia:

Cualquier sección sin sustento textual válido debe indicarse como no identificada.
"""

INDIVIDUAL_PROMPT_REQUIRED = """Usted actuará como un abogado corporativo senior especializado en análisis contractual.

Debe responder exclusivamente en función de los fragmentos de contrato proporcionados. Toda afirmación debe estar respaldada por una cita textual literal del fragmento correspondiente. No está permitido inferir, interpretar más allá del texto, ni incorporar información que no conste en los fragmentos.

Si la información solicitada no está contenida en los fragmentos del contrato, debe responder:
"No se encontró esta información en los fragmentos proporcionados".

FORMATO OBLIGATORIO DE RESPUESTA:
1. Respuesta directa, clara y precisa a la pregunta.
2. Evidencia textual literal del contrato que la respalde.
3. Si no existe evidencia, indíquelo expresamente.

Debe garantizar un estándar jurídico profesional en la precisión y redacción.
"""


def build_prompt(fragments: List[str], question: str, modo_resumen: bool) -> str:
    """
    Ensambla el prompt final incorporando:
    - El prompt requerido (resumen o individual) EXACTO.
    - El bloque de FRAGMENTOS (enumerado) que el modelo puede citar.
    - La pregunta.
    """
    fragments_block = "\n\n".join(
        f"[FRAGMENTO {i+1}]\n{txt}"
        for i, txt in enumerate(fragments)
    )

    # “Candado” adicional: obligar a citar solo desde fragmentos enumerados (sin cambiar los prompts requeridos)
    evidence_lock = (
        "\n\n"
        "REGLA OPERATIVA (OBLIGATORIA, NO NEGOCIABLE):\n"
        "- Usted solo puede citar evidencia textual literal desde los FRAGMENTOS enumerados a continuación.\n"
        "- Está prohibido inventar citas o redactar evidencia que no sea una cita literal.\n"
        "- Si no existe evidencia literal aplicable para un ítem, use exactamente la frase indicada en las instrucciones.\n"
    )

    if modo_resumen:
        return (
            SUMMARY_PROMPT_REQUIRED.strip()
            + evidence_lock
            + "\n\nFRAGMENTOS DE CONTRATO (ENUMERADOS):\n"
            + fragments_block
            + "\n\nPREGUNTA SOLICITADA:\n"
            + question
            + "\n"
        )

    return (
        INDIVIDUAL_PROMPT_REQUIRED.strip()
        + evidence_lock
        + "\n\nFRAGMENTOS DE CONTRATO (ENUMERADOS):\n"
        + fragments_block
        + "\n\nPREGUNTA:\n"
        + question
        + "\n\nRESPUESTA:\n"
    )


# =====================================================
# LLM
# =====================================================

def ask_llm(prompt: str) -> str:
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un abogado corporativo senior. "
                    "No inventes información. "
                    "No uses conocimiento externo. "
                    "Usa solo los fragmentos entregados. "
                    "La evidencia debe ser cita literal."
                )
            },
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content


# =====================================================
# MOTOR PRINCIPAL
# =====================================================

def ask_contract(question: str, contract_id: int, path: str, filetype: str) -> str:
    """
    Entrada principal:
    - question: pregunta del usuario
    - contract_id: id del contrato
    - path: ruta al archivo (pdf/docx)
    - filetype: 'pdf' o 'docx'
    """
    init_embeddings_table()

    # Normalizar ruta
    if not os.path.isabs(path):
        path = os.path.join(BASE_PATH, path)

    # Cargar texto
    text = load_contract_text(path, filetype)
    if not text.strip():
        return "No se pudo extraer texto del contrato."

    # Recuperar embeddings persistidos o generarlos
    chunks, embeddings = load_embeddings(contract_id)
    if chunks is None or embeddings is None:
        # Chunking (una vez)
        chunks = chunk_text(text, chunk_size=500, overlap=80)
        if not chunks:
            return "No se pudo segmentar el contrato en fragmentos utilizables."

        embeddings = embed_texts(chunks)
        save_embeddings(contract_id, chunks, embeddings)

    # Determinar modo
    modo_resumen = "resumen ejecutivo" in (question or "").lower()

    # Recuperación
    if modo_resumen:
        selected = multiconsulta_chunks(chunks, embeddings, min_score=DEFAULT_MIN_SCORE)
        selected = clip_context_by_chars(selected, MAX_CONTEXT_CHARS_SUMMARY)
        if not selected:
            return "No se encontró información relevante en los fragmentos proporcionados."
        prompt = build_prompt(selected, question, modo_resumen=True)
        return ask_llm(prompt)

    # Pregunta individual
    scored = search_similar(
        question,
        chunks,
        embeddings,
        top_k=DEFAULT_TOP_K_INDIVIDUAL,
        min_score=DEFAULT_MIN_SCORE
    )
    selected = [ch for _, ch in scored]
    selected = clip_context_by_chars(selected, MAX_CONTEXT_CHARS_INDIVIDUAL)

    if not selected:
        # No llamamos al LLM si no hay evidencia suficiente
        return "No se encontró esta información en los fragmentos proporcionados."

    prompt = build_prompt(selected, question, modo_resumen=False)
    return ask_llm(prompt)