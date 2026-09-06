# Fitness-Dashboard – Container für Fly.io (oder jeden anderen Docker-Host)
#
# Läuft serve_v2.py (v2-Oberfläche + Mobilansicht) auf Port 8080 mit dem
# unveränderten Rechenkern app.py. Die Datenbank-CSVs liegen auf einem
# persistenten Volume unter /app/datenbanken; beim allerersten Start wird das
# Volume aus den mitgelieferten Seed-Daten befüllt (siehe entrypoint.sh).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Europe/Berlin \
    V2_HOST=0.0.0.0 \
    V2_PORT=8080 \
    V2_DEBUG=0 \
    V2_NO_BROWSER=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Das Datenverzeichnis ist ein Mountpunkt für das Fly-Volume und bleibt leer.
# Liegen ausnahmsweise Seed-Daten im Build-Kontext, wandern sie zur Seite und
# werden beim allerersten Start ins leere Volume kopiert (s. entrypoint.sh).
RUN if [ -d /app/datenbanken ] && [ -n "$(ls -A /app/datenbanken 2>/dev/null)" ]; then \
        mv /app/datenbanken /app/seed_datenbanken; \
    fi \
 && mkdir -p /app/datenbanken \
 && chmod +x /app/entrypoint.sh

EXPOSE 8080
CMD ["/app/entrypoint.sh"]
