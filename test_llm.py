from openai import OpenAI

client = OpenAI()

context = """
La duración del contrato será de 1 año contado desde la fecha de firma,
renovable automáticamente por períodos iguales, salvo aviso previo de 60 días.
"""

question = "¿Cuál es la duración del contrato? incluye evidencia"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    messages=[
        {"role": "system", "content": "Responde usando el texto y cita evidencia textual"},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta:\n{question}"}
    ]
)

print(response.choices[0].message.content)