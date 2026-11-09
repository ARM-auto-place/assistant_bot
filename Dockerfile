FROM python:3.11-slim-bookworm
WORKDIR /service
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements-core.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt \
    && useradd --uid 10001 --create-home service
COPY *.py ./
USER service
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
