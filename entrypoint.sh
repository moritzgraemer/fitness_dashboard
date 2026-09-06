#!/bin/sh
# Container-Start: Datenverzeichnis bereitstellen, dann den Server starten.
#
# Zugangsdaten kommen NICHT aus einer Datei, sondern aus dem Fly-Secret
# APP_CONFIG_JSON; app.py und backend/sync_garmin_csv.py lesen es direkt aus
# der Umgebung. So liegt nie ein Passwort im Image oder im Git-Repository.
set -e

mkdir -p /app/datenbanken

if [ -z "$APP_CONFIG_JSON" ] && [ ! -f /app/config.json ]; then
  echo "⚠  Kein APP_CONFIG_JSON gesetzt – Garmin-Sync und KI-Funktionen bleiben aus."
  echo "   fly secrets set APP_CONFIG_JSON=\"\$(cat config.json)\""
fi

if [ -z "$DASH_PASSWORD" ]; then
  echo "⚠  Kein DASH_PASSWORD gesetzt – das Dashboard wäre öffentlich erreichbar!"
  echo "   fly secrets set DASH_PASSWORD='ein-langes-passwort'"
fi

# Erststart: Seed-Daten (nur vorhanden, wenn sie bewusst ins Image gelegt
# wurden) in das leere Volume kopieren. Im öffentlichen Repo gibt es keine.
if [ -d /app/seed_datenbanken ] && [ -z "$(ls -A /app/datenbanken 2>/dev/null)" ]; then
  cp -a /app/seed_datenbanken/. /app/datenbanken/
  echo "Seed-Daten in das Volume kopiert."
fi

exec python3 /app/serve_v2.py
