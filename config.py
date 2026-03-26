import os
import sys

if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# 🔥 SIEMPRE usar rutas relativas al proyecto
CONTRACTS_FOLDER = os.path.join(BASE_PATH, "contracts")

# Bases de datos
DB_NAME = os.path.join(BASE_PATH, "contracts.db")
LOG_DB = os.path.join(BASE_PATH, "logs.db")

# Templates
TEMPLATES_PATH = os.path.join(BASE_PATH, "templates")