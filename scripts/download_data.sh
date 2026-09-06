#!/bin/sh
# ════════════════════════════════════════════════════════════════════════════
#  Daten vom Fly-Volume zurückholen (Backup oder lokal weiterarbeiten)
#
#    scripts/download_data.sh <app-name> [zielverzeichnis]
# ════════════════════════════════════════════════════════════════════════════
set -e

APP="$1"
DEST="${2:-./datenbanken-backup-$(date +%Y%m%d)}"

if [ -z "$APP" ]; then
  echo "Aufruf: $0 <app-name> [zielverzeichnis]"; exit 1
fi

mkdir -p "$DEST"
echo "Packe auf dem Server …"
fly ssh console --app "$APP" -C "sh -c 'cd /app/datenbanken && tar -czf /tmp/backup.tar.gz --exclude=streams_raw .'"

echo "Hole Archiv …"
fly ssh sftp get /tmp/backup.tar.gz "$DEST/backup.tar.gz" --app "$APP"

echo "Entpacke nach $DEST …"
tar -xzf "$DEST/backup.tar.gz" -C "$DEST" && rm "$DEST/backup.tar.gz"
echo "✓ Fertig: $DEST"
