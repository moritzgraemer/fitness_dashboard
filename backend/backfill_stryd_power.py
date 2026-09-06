#!/usr/bin/env python3
"""
Stryd-Laufpower für bestehende Läufe nachtragen
===============================================

Einmaliger Backfill: holt für alle Läufe ohne avg_power_w die Garmin-
Aktivitätsdetails, liest die Stryd-Leistung (Connect-IQ-Feld) und schreibt
Ø/Max/NP in GarminConnectData_Aktivities.csv, die Rundenmittel in
GarminConnectData_Laps.csv und avg_watts/np_watts in activities.csv.

    python3 backend/backfill_stryd_power.py                # Läufe ab 2026-01-01
    python3 backend/backfill_stryd_power.py --since 2025-06-01
    python3 backend/backfill_stryd_power.py --max 20       # nur die 20 jüngsten

Läufe ohne Stryd-Feld (vor dem Pod) bleiben unverändert. Danach die
Einheiten-Ansicht neu laden; gecachte Streams laden sich beim Öffnen einmalig
neu (Stream-Format v3).
"""
import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import stryd_power as sp                     # noqa: E402
from garminconnect import Garmin             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--since', default='2026-01-01')
ap.add_argument('--max', type=int, default=None)
args = ap.parse_args()

with open(BASE_DIR.parent / 'config.json', encoding='utf-8') as f:
    cfg = json.load(f).get('garmin', {})
tokenstore = str(BASE_DIR.parent / 'datenbanken' / '.garmin_tokens')
api = Garmin(cfg.get('username', ''), cfg.get('password', ''))
try:
    api.login(tokenstore)
except Exception:
    api.login()

print(f'Backfill Stryd-Power für Läufe ab {args.since} …')
st = sp.enrich_stryd_power(api, since=args.since, max_calls=args.max)
print(f"\nFertig: {st['filled']} Läufe ergänzt, {st['checked']} geprüft, "
      f"{st['no_stryd']} ohne Stryd-Feld, {st['errors']} Fehler.")
