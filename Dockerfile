FROM python:3.10.10-slim

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Exponer puerto (solo documentación)
EXPOSE 8000

# Usar variable de entorno PORT de Cloud Run
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
