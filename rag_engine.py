# -*- coding: utf-8 -*-
"""
rag_engine.py — Motor RAG para análisis contractual (PDF/DOCX)

Corrige fallas críticas:
1) Resumen ejecutivo con retrieval por ítem (I..X), no contexto mezclado.
2) Evidencia con fragment_id + quote y VALIDACIÓN literal (quote debe existir en el fragmento).
3) Salida estructurada (JSON) y render controlado para evitar "evidencia inventada".
4) Mantiene dos modos: Resumen ejecutivo vs Pregunta individual.
5) Incluye EXACTAMENTE los dos prompts provistos por el usuario (sin alterarlos).
"""

import os
import sys
import json
import sqlite3
import re
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from reader import extract_text_from_pdf, extract_text_from_docx
from chunker import clean_text, chunk_text


# =====================================================
# CONFIGURACION
# =====================================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TEMPERATURE = 0.0  # mas estricto

DEFAULT_MIN_SCORE = 0.62
TOPK_INDIVIDUAL = 10
TOPK_PER_ITEM = 8

# Limites por caracteres para no explotar el prompt
MAX_CHARS_INDIVIDUAL = 22000
MAX_CHARS_PER_ITEM = 12000

# Frases exactas (para normalizar salidas y evitar que se usen como "evidencia")
NOT_FOUND_SUMMARY = "Esta información no se encuentra en los fragmentos proporcionados"
NOT_FOUND_INDIVIDUAL = "No se encontró esta información en los fragmentos proporcionados"
NO_EVIDENCE_PHRASE = "No existe evidencia textual disponible para este ítem en los fragmentos entregados"

# Cache
EMBEDDINGS_CACHE: Dict[int, Tuple[List[str], List[List[float]]]] = {}

# Rutas
if getattr(sys, "frozen", False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DB_NAME = os.path.join(BASE_PATH, "contracts.db")


# =====================================================
# PROMPTS OBLIGATORIOS (INCLUIDOS TAL CUAL)
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


# =====================================================
# BASE DE DATOS (EMBEDDINGS)
# =====================================================

def init_embeddings_table() -> None:
    conn = sqlite3.connect(DB_NAME, timeout=10)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS contract_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ce_contract_id ON contract_embeddings(contract_id)")
    conn.commit()
    conn.close()


def save_embeddings(contract_id: int, chunks: List[str], embeddings: List[List[float]]) -> None:
    conn = sqlite3.connect(DB_NAME, timeout=10)
    cur = conn.cursor()
    rows = [(contract_id, ch, json.dumps(emb)) for ch, emb in zip(chunks, embeddings)]
    cur.executemany(
        "INSERT INTO contract_embeddings (contract_id, chunk_text, embedding) VALUES (?, ?, ?)",
        rows
    )
    conn.commit()
    conn.close()
    EMBEDDINGS_CACHE[contract_id] = (chunks, embeddings)


def load_embeddings(contract_id: int) -> Tuple[Optional[List[str]], Optional[List[List[float]]]]:
    if contract_id in EMBEDDINGS_CACHE:
        return EMBEDDINGS_CACHE[contract_id]

    conn = sqlite3.connect(DB_NAME, timeout=5)
    cur = conn.cursor()
    cur.execute("SELECT chunk_text, embedding FROM contract_embeddings WHERE contract_id=?", (contract_id,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None, None

    chunks: List[str] = []
    embeddings: List[List[float]] = []
    for chunk_text, emb_json in rows:
        chunks.append(chunk_text)
        embeddings.append(json.loads(emb_json))

    EMBEDDINGS_CACHE[contract_id] = (chunks, embeddings)
    return chunks, embeddings


# =====================================================
# LECTURA DEL CONTRATO
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
    if isinstance(texts, str):
        texts = [texts]
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        res = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend([d.embedding for d in res.data])
    return out


# =====================================================
# SIMILITUD Y RETRIEVAL
# =====================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def search_similar(
    query: str,
    chunks: List[str],
    embeddings: List[List[float]],
    top_k: int,
    min_score: float
) -> List[Tuple[int, float]]:
    """Devuelve lista de (idx, score)"""
    q_emb = embed_texts([query])[0]
    scored: List[Tuple[int, float]] = []
    for i, emb in enumerate(embeddings):
        s = cosine_similarity(q_emb, emb)
        if s >= min_score:
            scored.append((i, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def select_fragments_by_indices(chunks: List[str], indices_scores: List[Tuple[int, float]]) -> Listreturn [chunks[i] for i, _ in indices_scores]


def clip_fragments_by_chars(fragments: List[str], max_chars: int) -> Listout: List[str] = []
    total = 0
    for fr in fragments:
        add = len(fr) + 2
        if total + add > max_chars:
            break
        out.append(fr)
        total += add
    return out


# =====================================================
# RETRIEVAL POR ITEM (RESUMEN EJECUTIVO)
# =====================================================

SUMMARY_ITEMS: List[Tuple[str, str]] = [
    ("I_Fechas", "fechas firma suscripción inicio vigencia término renovacion prórroga plazos"),
    ("II_Objeto", "objeto del contrato propósito finalidad servicios productos alcance"),
    ("III_Partes", "partes intervinientes contratante contratista proveedor cliente representante domicilio"),
    ("IV_Alcance", "alcance del servicio obligaciones del proveedor entregables responsabilidades del proveedor"),
    ("V_Vigencia", "vigencia duración plazo inicio término vigencia contractual"),
    ("VI_Renovacion", "renovación prórroga automática periodos iguales aviso renovación"),
    ("VII_Terminacion", "terminación anticipada rescisión causales aviso previo incumplimiento resolución"),
    ("VIII_Economico", "precio pagos facturación moneda tarifas honorarios impuestos plazos de pago"),
    ("IX_Obligaciones", "obligaciones principales confidencialidad deberes obligaciones de las partes"),
    ("X_Riesgos", "responsabilidad penalidades multas SLA nivel de servicio limitaciones indemnización daños")
]


def retrieve_per_summary_item(
    chunks: List[str],
    embeddings: List[List[float]],
    min_score: float = DEFAULT_MIN_SCORE,
    top_k_per_item: int = TOPK_PER_ITEM
) -> Dict[str, List[Tuple[int, float]]]:
    per: Dict[str, List[Tuple[int, float]]] = {}
    for key, q in SUMMARY_ITEMS:
        per[key] = search_similar(q, chunks, embeddings, top_k=top_k_per_item, min_score=min_score)
    return per


# =====================================================
# VALIDACION DE EVIDENCIA (ANTI-INVENTO)
# =====================================================

def normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def is_literal_quote_in_fragment(quote: str, fragment: str) -> bool:
    if not quote or not fragment:
        return False
    q = normalize_ws(quote)
    f = normalize_ws(fragment)
    return q in f


def evidence_is_banned(quote: str) -> bool:
    """Evita que el modelo use frases de control como si fueran evidencia."""
    q = normalize_ws(quote).strip().strip('"').strip("'")
    if not q:
        return True
    if q == NOT_FOUND_SUMMARY:
        return True
    if q == NOT_FOUND_INDIVIDUAL:
        return True
    if q == NO_EVIDENCE_PHRASE:
        return True
    return False


# =====================================================
# PROMPT BUILDER (CON JSON + BLOQUE FRAGMENTOS)
# =====================================================

def format_fragments_enumerated(fragments: List[str]) -> str:
    return "\n\n".join(f"[FRAGMENTO {i+1}]\n{fr}" for i, fr in enumerate(fragments))


def build_prompt_individual(question: str, fragments: List[str]) -> str:
    fragments_block = format_fragments_enumerated(fragments)

    # Instrucciones operativas adicionales (sin alterar el prompt requerido)
    operational = f"""
INSTRUCCIONES OPERATIVAS ADICIONALES (OBLIGATORIAS):
- Debe responder en JSON estrictamente valido.
- Debe incluir evidencia como un objeto con: fragment_id (entero) y quote (string literal).
- El quote debe ser una subcadena literal existente en el fragmento indicado.
- Si no existe evidencia, use exactamente la frase:
  "{NOT_FOUND_INDIVIDUAL}"
  y deje evidence como null.

FORMATO JSON OBLIGATORIO:
{{
  "answer": "texto de la respuesta",
  "evidence": {{
     "fragment_id": 3,
     "quote": "cita literal exacta"
  }},
  "not_found": false
}}

Si no hay información:
{{
  "answer": "{NOT_FOUND_INDIVIDUAL}",
  "evidence": null,
  "not_found": true
}}

FRAGMENTOS (SOLO ESTOS PUEDE CITAR):
{fragments_block}

PREGUNTA:
{question}
""".strip()

    return INDIVIDUAL_PROMPT_REQUIRED.strip() + "\n\n" + operational


def build_prompt_summary(question: str, fragments_per_item: Dict[str, List[str]]) -> str:
    # Construye un bloque por item (para evitar mezclar contexto)
    blocks: List[str] = []
    order_keys = [k for k, _ in SUMMARY_ITEMS]
    labels = {
        "I_Fechas": "I. Fechas Relevantes",
        "II_Objeto": "II. Objeto del Contrato",
        "III_Partes": "III. Partes que Intervienen",
        "IV_Alcance": "IV. Alcance y Servicios/Obligaciones del Proveedor",
        "V_Vigencia": "V. Vigencia Contractual",
        "VI_Renovacion": "VI. Mecanismos de Renovación",
        "VII_Terminacion": "VII. Causales y Condiciones de Terminación",
        "VIII_Economico": "VIII. Condiciones Económicas",
        "IX_Obligaciones": "IX. Obligaciones Principales",
        "X_Riesgos": "X. Aspectos de Riesgo Contractual (responsabilidad, penalidades, SLA, limitaciones)"
    }

    for key in order_keys:
        frs = fragments_per_item.get(key, [])
        frs = clip_fragments_by_chars(frs, MAX_CHARS_PER_ITEM)
        blocks.append(
            f"{labels.get(key, key)}\n"
            f"{format_fragments_enumerated(frs) if frs else '[SIN FRAGMENTOS PARA ESTE ITEM]'}"
        )

    per_item_block = "\n\n" + ("\n\n" + ("-" * 60) + "\n\n").join(blocks) + "\n\n"

    operational = f"""
INSTRUCCIONES OPERATIVAS ADICIONALES (OBLIGATORIAS):
- Debe responder en JSON estrictamente valido.
- Para cada item I..X debe producir: conclusion (string) y evidence (objeto o null).
- evidence, si existe, debe contener: fragment_id (entero) y quote (string literal).
- El quote debe ser literal y existir dentro del fragmento correspondiente.
- Si un item no tiene sustento textual, la conclusion debe ser EXACTAMENTE:
  "{NOT_FOUND_SUMMARY}"
  y evidence debe ser null.
- Si existe sustento, la evidence no puede ser la frase "{NOT_FOUND_SUMMARY}".

FORMATO JSON OBLIGATORIO:
{{
  "I_Fechas": {{
     "fields": {{
        "fecha_firma": {{ "conclusion": "...", "evidence": {{\"fragment_id\":1,\"quote\":\"...\"}} | null }},
        "inicio_vigencia": {{ ... }},
        "termino": {{ ... }},
        "renovacion": {{ ... }},
        "otros_plazos": {{ ... }}
     }}
  }},
  "II_Objeto": {{ "conclusion":"...", "evidence": {{...}} | null }},
  "III_Partes": {{ "conclusion":"...", "evidence": {{...}} | null }},
  "IV_Alcance": {{ "conclusion":"...", "evidence": {{...}} | null }},
  "V_Vigencia": {{ "conclusion":"...", "evidence": {{...}} | null }},
  "VI_Renovacion": {{ "conclusion":"...", "evidence": {{...}} | null }},
  "VII_Terminacion": {{ "conclusion":"...", "evidence": {{...}} | null }},
  "VIII_Economico": {{ "conclusion":"...", "evidence": {{...}} | null }},
  "IX_Obligaciones": {{ "conclusion":"...", "evidence": {{...}} | null }},
  "X_Riesgos": {{ "conclusion":"...", "evidence": {{...}} | null }}
}}

IMPORTANTE:
- Use fragment_id segun la numeracion dentro de cada bloque de item (FRAGMENTO 1..N de ese item).
- No referencie fragmentos fuera del item correspondiente.

PREGUNTA SOLICITADA:
{question}

FRAGMENTOS POR ITEM:
{per_item_block}
""".strip()

    return SUMMARY_PROMPT_REQUIRED.strip() + "\n\n" + operational


# =====================================================
# LLM + PARSER JSON ROBUSTO
# =====================================================

def call_llm(prompt: str) -> str:
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": (
                    "Abogado corporativo senior. Prohibido inventar. "
                    "Debe responder SOLO con JSON valido y evidencia literal."
                )
            },
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content


def extract_json_object(text: str) -> str:
    """
    Intenta extraer el primer objeto JSON del texto.
    """
    if not text:
        raise ValueError("Respuesta vacia del modelo.")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No se encontro JSON en la respuesta.")
    return text[start:end + 1]


def parse_json_response(text: str) -> Any:
    raw = extract_json_object(text)
    return json.loads(raw)


# =====================================================
# VALIDACION DE RESULTADOS (INDIVIDUAL)
# =====================================================

def validate_individual_json(data: dict, fragments: List[str]) -> dict:
    """
    Valida que:
    - Si evidence existe: fragment_id valido (1..N) y quote literal en fragmento.
    - No permite usar frases de control como evidencia.
    """
    if not isinstance(data, dict):
        return {"answer": NOT_FOUND_INDIVIDUAL, "evidence": None, "not_found": True}

    answer = data.get("answer")
    evidence = data.get("evidence")
    not_found = bool(data.get("not_found", False))

    if not answer:
        answer = NOT_FOUND_INDIVIDUAL
        not_found = True

    if not_found:
        return {"answer": NOT_FOUND_INDIVIDUAL, "evidence": None, "not_found": True}

    if evidence is None:
        # si no hay evidencia, debe declararlo con la frase exacta
        if normalize_ws(answer).strip() != NOT_FOUND_INDIVIDUAL:
            # Permitimos respuesta, pero sin evidencia es riesgoso: forzamos not_found
            return {"answer": NOT_FOUND_INDIVIDUAL, "evidence": None, "not_found": True}
        return {"answer": NOT_FOUND_INDIVIDUAL, "evidence": None, "not_found": True}

    try:
        fid = int(evidence.get("fragment_id"))
        quote = str(evidence.get("quote") or "")
    except Exception:
        return {"answer": NOT_FOUND_INDIVIDUAL, "evidence": None, "not_found": True}

    if fid < 1 or fid > len(fragments):
        return {"answer": NOT_FOUND_INDIVIDUAL, "evidence": None, "not_found": True}

    if evidence_is_banned(quote):
        return {"answer": NOT_FOUND_INDIVIDUAL, "evidence": None, "not_found": True}

    frag = fragments[fid - 1]
    if not is_literal_quote_in_fragment(quote, frag):
        return {"answer": NOT_FOUND_INDIVIDUAL, "evidence": None, "not_found": True}

    return {"answer": answer, "evidence": {"fragment_id": fid, "quote": quote}, "not_found": False}


def render_individual(validated: dict) -> str:
    if validated.get("not_found"):
        return NOT_FOUND_INDIVIDUAL

    ans = validated["answer"].strip()
    ev = validated.get("evidence") or {}
    fid = ev.get("fragment_id")
    quote = ev.get("quote")

    # Formato A: respuesta + evidencia al final
    return (
        f"{ans}\n\n"
        f"Evidencia:\n"
        f"[FRAGMENTO {fid}]\n"
        f"\"{quote}\""
    )


# =====================================================
# VALIDACION DE RESULTADOS (RESUMEN)
# =====================================================

def validate_evidence_obj(evidence: Any, fragments: List[str]) -> Optionalif evidence is None:
        return None
    if not isinstance(evidence, dict):
        return None
    try:
        fid = int(evidence.get("fragment_id"))
        quote = str(evidence.get("quote") or "")
    except Exception:
        return None

    if fid < 1 or fid > len(fragments):
        return None
    if evidence_is_banned(quote):
        return None
    if not is_literal_quote_in_fragment(quote, fragments[fid - 1]):
        return None
    return {"fragment_id": fid, "quote": quote}


def validate_summary_json(data: dict, fragments_per_item: Dict[str, List[str]]) -> dict:
    """
    Valida item por item. Si evidencia invalida o conclusion vacia -> marca NOT_FOUND_SUMMARY y evidence null.
    Para I_Fechas valida fields internos.
    """
    out: dict = {}
    if not isinstance(data, dict):
        # si el JSON no es correcto, devolvemos todo no encontrado
        for key, _ in SUMMARY_ITEMS:
            if key == "I_Fechas":
                out[key] = {
                    "fields": {
                        "fecha_firma": {"conclusion": NOT_FOUND_SUMMARY, "evidence": None},
                        "inicio_vigencia": {"conclusion": NOT_FOUND_SUMMARY, "evidence": None},
                        "termino": {"conclusion": NOT_FOUND_SUMMARY, "evidence": None},
                        "renovacion": {"conclusion": NOT_FOUND_SUMMARY, "evidence": None},
                        "otros_plazos": {"conclusion": NOT_FOUND_SUMMARY, "evidence": None},
                    }
                }
            else:
                out[key] = {"conclusion": NOT_FOUND_SUMMARY, "evidence": None}
        return out

    for key, _ in SUMMARY_ITEMS:
        frs = fragments_per_item.get(key, [])
        frs = clip_fragments_by_chars(frs, MAX_CHARS_PER_ITEM)

        if key == "I_Fechas":
            node = data.get(key, {})
            fields = (node.get("fields") if isinstance(node, dict) else {}) or {}
            out_fields = {}
            for fkey in ["fecha_firma", "inicio_vigencia", "termino", "renovacion", "otros_plazos"]:
                fnode = fields.get(fkey, {}) if isinstance(fields, dict) else {}
                concl = (fnode.get("conclusion") if isinstance(fnode, dict) else None) or NOT_FOUND_SUMMARY
                evid = (fnode.get("evidence") if isinstance(fnode, dict) else None)

                ve = validate_evidence_obj(evid, frs) if frs else None

                # si no hay evidencia valida, forzar conclusion a NOT_FOUND_SUMMARY (evita relleno)
                if ve is None:
                    out_fields[fkey] = {"conclusion": NOT_FOUND_SUMMARY, "evidence": None}
                else:
                    out_fields[fkey] = {"conclusion": concl, "evidence": ve}

            out[key] = {"fields": out_fields}
        else:
            node = data.get(key, {}) if isinstance(data.get(key, {}), dict) else {}
            concl = (node.get("conclusion") if isinstance(node, dict) else None) or NOT_FOUND_SUMMARY
            evid = node.get("evidence") if isinstance(node, dict) else None

            ve = validate_evidence_obj(evid, frs) if frs else None

            if ve is None:
                out[key] = {"conclusion": NOT_FOUND_SUMMARY, "evidence": None}
            else:
                out[key] = {"conclusion": concl, "evidence": ve}

    return out


def render_summary(validated: dict, fragments_per_item: Dict[str, List[str]]) -> str:
    labels = {
        "I_Fechas": "I. Fechas Relevantes",
        "II_Objeto": "II. Objeto del Contrato",
        "III_Partes": "III. Partes que Intervienen",
        "IV_Alcance": "IV. Alcance y Servicios/Obligaciones del Proveedor",
        "V_Vigencia": "V. Vigencia Contractual",
        "VI_Renovacion": "VI. Mecanismos de Renovación",
        "VII_Terminacion": "VII. Causales y Condiciones de Terminación",
        "VIII_Economico": "VIII. Condiciones Económicas",
        "IX_Obligaciones": "IX. Obligaciones Principales",
        "X_Riesgos": "X. Aspectos de Riesgo Contractual (responsabilidad, penalidades, SLA, limitaciones)"
    }

    lines: List[str] = []
    # I Fechas (subcampos)
    lines.append(labels["I_Fechas"])
    fields = validated["I_Fechas"]["fields"]
    pretty = {
        "fecha_firma": "Fecha de firma",
        "inicio_vigencia": "Inicio de vigencia",
        "termino": "Término",
        "renovacion": "Renovación",
        "otros_plazos": "Otros plazos relevantes"
    }
    frs_I = clip_fragments_by_chars(fragments_per_item.get("I_Fechas", []), MAX_CHARS_PER_ITEM)

    for fkey, title in pretty.items():
        node = fields.get(fkey, {})
        concl = node.get("conclusion", NOT_FOUND_SUMMARY)
        ev = node.get("evidence")
        lines.append(f"- {title}: {concl}")
        if ev is None:
            lines.append(f"  Evidencia: {NO_EVIDENCE_PHRASE}")
        else:
            fid = ev["fragment_id"]
            quote = ev["quote"]
            lines.append(f"  Evidencia: [FRAGMENTO {fid}] \"{quote}\"")

    lines.append("")  # blank

    # Items II..X
    for key in [k for k, _ in SUMMARY_ITEMS if k != "I_Fechas"]:
        lines.append(labels.get(key, key))
        node = validated.get(key, {})
        concl = node.get("conclusion", NOT_FOUND_SUMMARY)
        ev = node.get("evidence")
        lines.append(f"Conclusión: {concl}")
        if ev is None:
            lines.append(f"Evidencia: {NO_EVIDENCE_PHRASE}")
        else:
            fid = ev["fragment_id"]
            quote = ev["quote"]
            lines.append(f"Evidencia: [FRAGMENTO {fid}] \"{quote}\"")
        lines.append("")

    return "\n".join(lines).strip()


# =====================================================
# MOTOR PRINCIPAL
# =====================================================

def get_or_create_embeddings(contract_id: int, text: str) -> Tuple[List[str], List[List[float]]]:
    chunks, embeddings = load_embeddings(contract_id)
    if chunks is not None and embeddings is not None:
        return chunks, embeddings

    chunks = chunk_text(text, chunk_size=500, overlap=80)
    embeddings = embed_texts(chunks)
    save_embeddings(contract_id, chunks, embeddings)
    return chunks, embeddings


def ask_contract(question: str, contract_id: int, path: str, filetype: str) -> str:
    init_embeddings_table()

    if not os.path.isabs(path):
        path = os.path.join(BASE_PATH, path)

    text = load_contract_text(path, filetype)
    if not text.strip():
        return "No se pudo extraer texto del contrato."

    chunks, embeddings = get_or_create_embeddings(contract_id, text)

    is_summary = "resumen ejecutivo" in (question or "").lower()

    if not is_summary:
        # Pregunta individual
        idx_scores = search_similar(question, chunks, embeddings, top_k=TOPK_INDIVIDUAL, min_score=DEFAULT_MIN_SCORE)
        fragments = select_fragments_by_indices(chunks, idx_scores)
        fragments = clip_fragments_by_chars(fragments, MAX_CHARS_INDIVIDUAL)

        if not fragments:
            return NOT_FOUND_INDIVIDUAL

        prompt = build_prompt_individual(question, fragments)
        raw = call_llm(prompt)

        try:
            data = parse_json_response(raw)
        except Exception:
            # fallback duro
            return NOT_FOUND_INDIVIDUAL

        validated = validate_individual_json(data, fragments)
        return render_individual(validated)

    # Resumen ejecutivo: retrieval por item
    per_item_idx = retrieve_per_summary_item(chunks, embeddings, min_score=DEFAULT_MIN_SCORE, top_k_per_item=TOPK_PER_ITEM)

    # Construye fragmentos por item (cada item independiente)
    fragments_per_item: Dict[str, List[str]] = {}
    for key, _ in SUMMARY_ITEMS:
        idx_scores = per_item_idx.get(key, [])
        frs = select_fragments_by_indices(chunks, idx_scores)
        fragments_per_item[key] = clip_fragments_by_chars(frs, MAX_CHARS_PER_ITEM)

    prompt = build_prompt_summary(question, fragments_per_item)
    raw = call_llm(prompt)

    try:
        data = parse_json_response(raw)
    except Exception:
        # fallback duro: todo no encontrado
        validated = validate_summary_json({}, fragments_per_item)
        return render_summary(validated, fragments_per_item)

    validated = validate_summary_json(data, fragments_per_item)
    return render_summary(validated, fragments_per_item)