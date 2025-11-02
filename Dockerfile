# ---- Base Python ----
FROM python:3.12-slim

# ---- Définir le répertoire de travail ----
WORKDIR /app

# ---- Copier le code ----
COPY . .

# ---- Installer les dépendances ----
RUN pip install --no-cache-dir uv && \
    uv pip install --system .

# ---- Commande de démarrage ----
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]