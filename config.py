import os
import getpass

# Usuario Windows actual
username = getpass.getuser()

# Ruta base en OneDrive
BASE_DIR = fr"C:\Users\{username}\OneDrive - Telefonica\CONTRATOS FIRMADOS"

# 🔥 FIX: usar carpeta correcta
CONTRACTS_FOLDER = os.path.join(BASE_DIR, "contracts")

# Carpeta IA (si la usas)
AI_FOLDER = os.path.join(BASE_DIR, "CONTRACT_AI")

# Bases de datos
DB_NAME = os.path.join(BASE_DIR, "contracts.db")
LOG_DB = os.path.join(BASE_DIR, "logs.db")

# Templates Flask
TEMPLATES_PATH = os.path.join(BASE_DIR, "templates")