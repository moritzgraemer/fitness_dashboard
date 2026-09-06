#!/bin/sh
# ════════════════════════════════════════════════════════════════════════════
#  Persönliche Daten auf das Fly-Volume schieben (Erstbefüllung oder Umzug)
#
#  Die Daten liegen bewusst NICHT im Repository. Dieses Skript packt das
#  lokale Datenverzeichnis und entpackt es auf dem Server im Volume.
#
#    scripts/upload_data.sh <app-name> [pfad-zum-lokalen-datenverzeichnis]
#
#  Ohne zweites Argument wird ../datenbanken relativ zum Skript genommen.
# ════════════════════════════════════════════════════════════════════════════
set -e

APP="$1"
SRC="${2:-$(cd "$(dirname "$0")/.." && pwd)/datenbanken}"

if [ -z "$APP" ]; then
  echo "Aufruf: $0 <app-name> [datenverzeichnis]"; exit 1
fi
if [ ! -d "$SRC" ]; then
  echo "✗ Verzeichnis nicht gefunden: $SRC"; exit 1
fi

TMP="$(mktemp -d)"
ARCHIVE="$TMP/datenbanken.tar.gz"

echo "Packe $SRC …"
# streams_raw (Roh-Cache, ~75 MB) und Logs bleiben draußen; .garmin_tokens
# kommt mit, damit der Server ohne erneutes Passwort-Login syncen kann.
tar -czf "$ARCHIVE" -C "$SRC" \
    --exclude='streams_raw' --exclude='*.log' --exclude='*.bak' .
echo "  $(du -h "$ARCHIVE" | cut -f1)"

echo "Übertrage auf $APP …"
fly ssh sftp shell --app "$APP" <<SFTP
put $ARCHIVE /app/datenbanken.tar.gz
SFTP

echo "Entpacke im Volume …"
fly ssh console --app "$APP" -C "sh -c 'cd /app/datenbanken && tar -xzf /app/datenbanken.tar.gz && rm /app/datenbanken.tar.gz && ls | head'"

rm -rf "$TMP"
echo "✓ Fertig. Die App zeigt die Daten nach einem Reload."
