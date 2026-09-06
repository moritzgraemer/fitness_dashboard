#!/usr/bin/env python3
"""
Dashboard_Claude – Backend
Fetches Strava activities → CSV → computes CTL/ATL/TSB → serves dashboard.

Usage:
  pip install flask pandas requests urllib3
  python app.py
  → http://localhost:5000

First run:  POST /api/sync  (fetches all Strava activities, saves activities.csv)
"""

import os
import sys
import re
import json
import math
import subprocess
import threading
import webbrowser
import requests
import urllib3
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Zentrale Zugangsdaten ─────────────────────────────────────────────────────
# Alle Secrets liegen in config.json (NICHT in Git – siehe .gitignore).
# Vorlage: config.example.json.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')


def _load_config():
    """Zugangsdaten laden – aus der Umgebung oder aus config.json.

    Auf einem Server (Fly.io) stehen die Zugangsdaten als Secret in der
    Umgebungsvariable APP_CONFIG_JSON; dann existiert gar keine Datei und es
    landet auch nie eine im Image. Lokal bleibt config.json der Weg.
    Fehlt beides, startet die App trotzdem: die Oberfläche und alle bereits
    vorhandenen Daten funktionieren, nur Garmin-Sync und KI-Funktionen nicht.
    """
    raw = os.environ.get('APP_CONFIG_JSON', '').strip()
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f'⚠  APP_CONFIG_JSON ist kein gültiges JSON ({e}) – wird ignoriert.')
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f'\n⚠  Keine Zugangsdaten gefunden (weder APP_CONFIG_JSON noch {CONFIG_PATH}).\n'
              '   Garmin-Sync und KI-Funktionen bleiben aus; alles andere läuft.\n')
        return {}
    except json.JSONDecodeError as e:
        print(f'\n⚠  config.json ist kein gültiges JSON ({e}) – wird ignoriert.\n')
        return {}


CONFIG = _load_config()

GROQ_API_KEY      = os.environ.get('GROQ_API_KEY') or CONFIG.get('groq', {}).get('api_key', '')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')

try:
    from groq import Groq as _GroqClient
    _groq = _GroqClient(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    _groq = None

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
TRAINER_PROFILE_PATH = os.path.join(BASE_DIR, 'trainer_profile.md')
DB_DIR         = os.path.join(BASE_DIR, 'datenbanken')
# Gemeinsamer Garmin-Token-Store (garth-Session), von app.py/sync_garmin_csv.py/
# compute_best_efforts.py geteilt, damit ein Sync-Vorgang nicht mehrfach das
# ratenlimitierte Login-Endpoint trifft (429 "IP rate limited by Garmin").
GARMIN_TOKENSTORE = os.path.join(DB_DIR, '.garmin_tokens')
FRONTEND_DIR   = os.path.join(BASE_DIR, 'frontend')
BACKEND_DIR    = os.path.join(BASE_DIR, 'backend')
CSV_PATH       = os.path.join(DB_DIR, 'activities.csv')
INJURIES_PATH  = os.path.join(DB_DIR, 'injuries.csv')
STREAMS_DIR    = os.path.join(DB_DIR, 'streams')
GOALS_PATH     = os.path.join(DB_DIR, 'goals.csv')
PLANNED_PATH   = os.path.join(DB_DIR, 'planned_sessions.csv')
NON_CARDIO_PATH = os.path.join(DB_DIR, 'non_cardio_activities.csv')
DELETED_IDS_PATH = os.path.join(DB_DIR, 'deleted_activities.csv')
MAKROPLAN_PATH        = os.path.join(DB_DIR, 'makroplan.csv')
INJURY_UPDATES_PATH   = os.path.join(DB_DIR, 'injury_updates.csv')
MORNING_PATH          = os.path.join(DB_DIR, 'morning_checkins.csv')
GARMIN_BODY_PATH      = os.path.join(DB_DIR, 'GarminConnectData_Koerperdaten.csv')
GARMIN_AKTIVITIES_PATH = os.path.join(DB_DIR, 'GarminConnectData_Aktivities.csv')
GARMIN_LAPS_PATH       = os.path.join(DB_DIR, 'GarminConnectData_Laps.csv')
WIDGET_DATA_PATH      = os.path.expanduser('~/widget_data.csv')

# Trainings-Zustandsmodell (energiesystem-spezifische Fitness) – reines Modul.
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
import training_model as tm
import unit_load as ul       # L_aer / L_mech pro Runde
import readiness as rd       # Zustand (CTL/ATL/TSB, z-Scores, R) + Kaskade
import stryd_power as sp     # Laufpower aus Stryd-Connect-IQ-Feldern

def _load_trainer_profile() -> str:
    """Read trainer_profile.md; returns empty string if file is missing."""
    try:
        with open(TRAINER_PROFILE_PATH, encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return ''
HTML_FILE             = 'dashboard_claude.html'

# ── Strava (deaktiviert) ──────────────────────────────────────────────────────
# Die Strava-API-App wurde serverseitig auf "Inactive" gesetzt und liefert keine
# Daten mehr. Aktivitäten, Streams und Routen kommen jetzt aus Garmin Connect
# (siehe merge_garmin_activities_into_csv / _get_garmin_stream weiter unten).
# Bereits geladene Strava-Aktivitäten in activities.csv bleiben erhalten.

# ── Training config (adjust to your values) ───────────────────────────────────
LTHR            = 168   # Lactate Threshold Heart Rate (bpm)
FTP             = 248   # Functional Threshold Power (Watts, for cycling TSS)
RUN_FTP         = 0     # Laufleistungs-Schwelle (Stryd, Watt). 0 = kein Laufpower
HR_MAX          = 198   # Maximum Heart Rate (bpm)
CTL_TC          = 42    # Fitness time constant (days)
ATL_TC          = 7     # Fatigue time constant (days)
# RUN_MULTIPLIER (früher ×1,2 auf Läufe) ist entfallen: der exzentrische
# Mehraufwand des Laufens steckt jetzt explizit in L_mech statt als pauschaler
# Aufschlag in der aeroben Last. Siehe backend/unit_load.py.

# ── Strength / climbing load (session-RPE method, Foster) ─────────────────────
# Kraft & Bouldern erzeugen kaum kardiovaskulären, aber realen neuromuskulären
# Stress – HF-TSS würde ihn falsch verbuchen. Stattdessen sRPE-Last = RPE×Dauer.
# Kalibrierung an den Fallback-Anker (50 TSS/h ≙ moderat, RPE 5 über 60 min):
#   RPE 5 × 60 min = 300 AU  ≙  50 TSS  →  k = 50/300 ≈ 0.167
STRENGTH_TYPES   = {'WeightTraining', 'Workout', 'Crossfit', 'StrengthTraining',
                    'RockClimbing', 'Bouldering', 'Climbing',
                    # Deutsche Werte aus dem manuellen Eingabe-Formular (Frontend)
                    'Krafttraining', 'Bouldern'}
SRPE_TO_TSS      = 50.0 / 300.0   # AU → TSS-Äquivalent
DEFAULT_RPE      = 6              # Fallback, wenn keine RPE geloggt wurde

# Emoji je Aktivitätstyp – inkl. der deutschen Typen aus dem manuellen
# Eingabe-Formular (Krafttraining / Bouldern) und der englischen Strava-Typen.
ACTIVITY_ICONS = {
    'Run': '🏃', 'TrailRun': '⛰️', 'VirtualRun': '🏃',
    'Ride': '🚴', 'VirtualRide': '🚴', 'GravelRide': '🚴',
    'EBikeRide': '🚴', 'MountainBikeRide': '🚵‍♂️',
    'WeightTraining': '🏋️', 'Workout': '🏋️', 'StrengthTraining': '🏋️',
    'Crossfit': '🏋️', 'Krafttraining': '🏋️',
    'RockClimbing': '🧗', 'Bouldering': '🧗', 'Climbing': '🧗', 'Bouldern': '🧗',
    'Swim': '🏊', 'Walk': '🚶', 'Hike': '🥾',
    'RowingV2': '🚣', 'Rowing': '🚣', 'Kayaking': '🛶',
    'Yoga': '🧘', 'Pilates': '🧘', 'InlineSkate': '🛼',
    'HighIntensityIntervalTraining': '🔥',
}

# HR zones: 5-zone model (% of HR_MAX), non-overlapping integer bpm ranges
# Lower bound of zone n = floor(HR_MAX * lower_pct) + 1  (except zone 1 starts at floor(50%))
# Upper bound of zone n = floor(HR_MAX * upper_pct)  (zone 5 ends at HR_MAX)
_Z = [math.floor(HR_MAX * p) for p in (0.50, 0.60, 0.70, 0.80, 0.90)]
ZONE_BPM = {
    1: (_Z[0],      _Z[1]),        # 99–118 bpm
    2: (_Z[1] + 1,  _Z[2]),        # 119–138 bpm
    3: (_Z[2] + 1,  _Z[3]),        # 139–158 bpm
    4: (_Z[3] + 1,  _Z[4]),        # 159–178 bpm
    5: (_Z[4] + 1,  HR_MAX),       # 179–198 bpm
}

app = Flask(__name__, static_folder=FRONTEND_DIR)


# ═════════════════════════════════════════════════════════════════════════════
# Tombstones – lokal gelöschte Aktivitäten dauerhaft ausblenden
# ═════════════════════════════════════════════════════════════════════════════

def _load_deleted_ids() -> set:
    """IDs von Aktivitäten, die lokal gelöscht wurden. Diese Liste überlebt jeden
    Sync und verhindert, dass Strava-Aktivitäten beim nächsten Sync wieder
    auftauchen (in Strava selbst bleiben sie unangetastet)."""
    if not os.path.exists(DELETED_IDS_PATH):
        return set()
    try:
        df = pd.read_csv(DELETED_IDS_PATH)
        return set(pd.to_numeric(df['id'], errors='coerce').dropna().astype('int64'))
    except (OSError, KeyError, ValueError):
        return set()


def _add_deleted_id(activity_id: int) -> None:
    """Markiert eine Aktivitäts-ID dauerhaft als gelöscht (Tombstone)."""
    ids = _load_deleted_ids()
    ids.add(int(activity_id))
    pd.DataFrame({'id': sorted(ids)}).to_csv(DELETED_IDS_PATH, index=False)


# ═════════════════════════════════════════════════════════════════════════════
# CSV – save & load
# ═════════════════════════════════════════════════════════════════════════════

def load_csv():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, parse_dates=['date'])
        # Ensure text columns are always object dtype (never float64 / NaN-only columns)
        for col in ('comment', 'ai_analysis', 'intention', 'intention_met'):
            if col not in df.columns:
                df[col] = ''
            df[col] = df[col].fillna('').astype(str)
        if 'rpe' not in df.columns:
            df['rpe'] = ''
        return df.sort_values('date').reset_index(drop=True)
    return pd.DataFrame()


def _parse_hhmm(val) -> int:
    """'1:30' -> 5400 Sekunden. Akzeptiert h:mm oder h:mm:ss. Leer/ungültig -> 0."""
    try:
        s = str(val).strip()
        if not s or s in ('—', 'nan', 'None'):
            return 0
        parts = s.split(':')
        if len(parts) == 2:
            h, m = parts
            return int(h) * 3600 + int(m) * 60
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + int(sec)
        return 0
    except (ValueError, TypeError):
        return 0


def _clean_nc_str(val) -> str:
    """NaN/None/'nan' (entsteht durch pandas-Inferenz bei leeren CSV-Zellen) -> ''."""
    if val is None:
        return ''
    if isinstance(val, float) and math.isnan(val):
        return ''
    s = str(val).strip()
    return '' if s.lower() in ('nan', 'none') else s


def _non_cardio_to_activity_rows() -> pd.DataFrame:
    """Manuelle Kraft-/Boulder-Einträge (non_cardio_activities.csv) in das
    Aktivitäten-Schema überführen, damit sie durch dieselbe TSS-/Trainingslast-
    Pipeline laufen wie Strava-Aktivitäten (sonst bleiben CTL/ATL/Kraft-Score
    von manuell geloggten Einheiten komplett unberührt).
    Negative IDs verhindern Kollisionen mit echten (immer positiven) Strava-IDs."""
    nc = _load_non_cardio()
    if nc.empty:
        return pd.DataFrame()
    rows = []
    for _, r in nc.iterrows():
        rows.append({
            'id':            -int(r['id']),
            'name':          r.get('name') or r.get('type', ''),
            'type':          r.get('type', ''),
            'date':          r.get('date', ''),
            'moving_time':   _parse_hhmm(r.get('duration')),
            'distance_km':   0.0,
            'elevation_m':   0,
            'avg_hr':        0,
            'max_hr':        0,
            'avg_watts':     0,
            'np_watts':      0,
            'avg_speed_kmh': 0.0,
            'polyline':      '',
            'comment':       _clean_nc_str(r.get('comment')),
            'ai_analysis':   '',
            'intention':     _clean_nc_str(r.get('intention')),
            'intention_met': _clean_nc_str(r.get('intention_met')),
            'rpe':           r.get('rpe', ''),
        })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.dropna(subset=['date'])


def load_activities():
    """load_csv() + manuelle Kraft-/Boulder-Einträge zusammengeführt.
    Für ALLES, was TSS/CTL/ATL/System-Scores berechnet oder Aktivitäten
    anzeigt. NICHT für Code-Pfade, die nach load_csv() wieder auf CSV_PATH
    zurückschreiben (sonst würden die negativen Manual-IDs in activities.csv
    landen) – die bleiben bei load_csv()."""
    df    = load_csv()
    extra = _non_cardio_to_activity_rows()
    if extra.empty:
        return df
    merged = pd.concat([df, extra], ignore_index=True) if not df.empty else extra
    return merged.sort_values('date').reset_index(drop=True)


def _backfill_activity_intentions():
    """Carry the intention of a planned session over to the completed activity on
    the same day. Best-effort match by date — writes back to activities.csv only
    when something actually changed. Returns True if the CSV was updated."""
    if not os.path.exists(CSV_PATH):
        return False
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=['date'])
    except Exception:
        return False
    if df.empty:
        return False
    if 'intention' not in df.columns:
        df['intention'] = ''
    df['intention'] = df['intention'].fillna('').astype(str)

    planned = _load_planned()
    if planned.empty:
        return False
    # date (YYYY-MM-DD) -> first non-empty planned intention for that day
    by_date = {}
    for _, p in planned.iterrows():
        intent = str(p.get('intention', '') or '').strip()
        day    = str(p.get('date', ''))[:10]
        if intent and day and day not in by_date:
            by_date[day] = intent
    if not by_date:
        return False

    changed = False
    for idx, r in df.iterrows():
        if str(r.get('intention', '') or '').strip():
            continue
        day = r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])[:10]
        if day in by_date:
            df.at[idx, 'intention'] = by_date[day]
            changed = True
    if changed:
        df.to_csv(CSV_PATH, index=False)
    return changed


# ═════════════════════════════════════════════════════════════════════════════
# Last pro Einheit – L_aer (aerob) und L_mech (mechanisch)
#
# Die Rechenkerne stehen in backend/unit_load.py; hier wird nur das Beschaffen
# der Runden (Laps) und der Modalitäts-Sonderfall Kraft/Bouldern erledigt.
#
#   L_aer  = Σ_i (t_i/60) · IF_i² · 100        (1 h an der Schwelle = 100)
#   L_mech = Σ_i φ_i · n_i / 1000              (Lauf) bzw. λ · t_h · IF² (Rad)
#
# Weil IF² konvex ist (Jensen), wird L_aer RUNDENWEISE gebildet, sobald Garmin
# Runden geliefert hat – die alte Rechnung über die Ø-HF der ganzen Aktivität
# unterschätzt Intervalleinheiten systematisch. Ohne Runden wird die Aktivität
# als eine einzige Runde behandelt; dann ist das Ergebnis identisch zur alten
# Formel (ohne den entfallenen ×1,2-Laufaufschlag, s. RUN_MULTIPLIER oben).
#
# Kraft/Bouldern bleiben bei der sRPE-Last (RPE × Dauer × k): dort ist weder HF
# noch Schrittzahl ein sinnvoller Treiber.
# ═════════════════════════════════════════════════════════════════════════════

def is_strength_type(t) -> bool:
    """True für Kraft-/Boulder-Modalitäten (sRPE-Last statt HF-TSS)."""
    return str(t) in STRENGTH_TYPES


# ── Runden-Beschaffung (gecacht) ──────────────────────────────────────────────
# calc_tss läuft zeilenweise über bis zu 1400 Aktivitäten und wird pro Request
# mehrfach aufgerufen – die Laps-CSV darf dabei nicht je Zeile gelesen werden.
# Beide Caches invalidieren über die mtime ihrer Quelldatei.
_LAPS_CACHE   = {'mtime': None, 'index': {}}
_GARMIN_MAP   = {'key': None, 'map': {}}


def _laps_index() -> dict:
    """{garmin_activity_id: [Roh-Runde, ...]} aus GarminConnectData_Laps.csv."""
    try:
        mtime = os.path.getmtime(GARMIN_LAPS_PATH)
    except OSError:
        return {}
    if _LAPS_CACHE['mtime'] == mtime:
        return _LAPS_CACHE['index']

    index = {}
    try:
        df = pd.read_csv(GARMIN_LAPS_PATH)
        for col in ('avg_run_cadence', 'avg_bike_cadence', 'moving_duration_s'):
            if col not in df.columns:
                df[col] = float('nan')
        df = df.sort_values(['activity_id', 'lap_number'])
        for act_id, grp in df.groupby('activity_id'):
            laps = []
            for _, r in grp.iterrows():
                cad = r['avg_run_cadence']
                if pd.isna(cad):
                    cad = r['avg_bike_cadence']
                # Netto- statt Bruttozeit: duration_s enthält Ampeln, Foto- und
                # Trinkpausen (Median +3 s, aber bis zu +32 min pro Runde). Der
                # Rest der App rechnet durchgehend mit moving_time – mit der
                # Bruttozeit wären L_aer und L_mech systematisch zu hoch, und
                # die Schritte einer Standzeit würden als Aufprall verbucht.
                dur = r['moving_duration_s']
                if pd.isna(dur) or float(dur) <= 0:
                    dur = r.get('duration_s')
                laps.append({
                    'lap_number':   r.get('lap_number'),
                    'duration_s':   dur,
                    'distance_m':   r.get('distance_m'),
                    # avg_speed_ms von Garmin bezieht sich auf die Bruttozeit;
                    # das Tempo wird deshalb in normalize_lap aus Distanz/Nettozeit
                    # rekonstruiert (φ soll das gelaufene Tempo abbilden).
                    'avg_speed_ms': None,
                    'avg_hr':       r.get('avg_hr'),
                    'avg_power_w':  r.get('avg_power_w'),
                    'ascent_m':     r.get('ascent_m'),
                    'descent_m':    r.get('descent_m'),
                    'cadence_spm':  cad,
                })
            index[int(act_id)] = laps
    except Exception:
        index = {}
    _LAPS_CACHE.update(mtime=mtime, index=index)
    return index


def _garmin_id_map() -> dict:
    """{activities.csv-id: garmin_activity_id} für alle Aktivitäten auf einmal.

    Garmin-IDs stehen direkt drin, seit der Sync auf Garmin umgestellt wurde;
    für den Übergangszeitraum mit Strava-IDs wird wie in
    _find_garmin_activity_id über Datum + Distanz gematcht – hier aber gebündelt
    statt pro Aktivität, weil sonst jede Zeile zwei CSVs laden würde.
    """
    try:
        key = (os.path.getmtime(GARMIN_AKTIVITIES_PATH), os.path.getmtime(CSV_PATH))
    except OSError:
        return {}
    if _GARMIN_MAP['key'] == key:
        return _GARMIN_MAP['map']

    mapping = {}
    try:
        acts = load_csv()
        gar  = pd.read_csv(GARMIN_AKTIVITIES_PATH)
        gar['start_time'] = pd.to_datetime(gar['start_time'], errors='coerce')
        gar['day'] = gar['start_time'].dt.date
        known = set(gar['activity_id'].dropna().astype('int64'))

        by_day = {d: g for d, g in gar.groupby('day')}
        for _, r in acts.iterrows():
            try:
                aid = int(r['id'])
            except (TypeError, ValueError):
                continue
            if aid in known:                     # bereits eine Garmin-ID
                mapping[aid] = aid
                continue
            day = r['date'].date() if hasattr(r['date'], 'date') else None
            same_day = by_day.get(day)
            if same_day is None or same_day.empty:
                continue
            try:
                target_km = float(r['distance_km'])
            except (TypeError, ValueError):
                continue
            diffs = (same_day['distance_km'] - target_km).abs()
            best  = diffs.idxmin()
            if diffs.loc[best] <= max(0.15, target_km * 0.03):
                mapping[aid] = int(same_day.loc[best, 'activity_id'])
    except Exception:
        mapping = {}
    _GARMIN_MAP.update(key=key, map=mapping)
    return mapping


def activity_laps(row) -> list:
    """Runden einer Aktivität – echte Garmin-Laps, sonst eine synthetische
    Runde über die ganze Aktivität (dann degeneriert L_aer zur alten Formel)."""
    try:
        aid = int(row['id'])
    except (TypeError, ValueError, KeyError):
        aid = None

    if aid is not None and aid > 0:
        gid  = _garmin_id_map().get(aid, aid)
        laps = _laps_index().get(gid)
        if laps:
            return laps

    # ── Synthetische Ganz-Aktivitäts-Runde ────────────────────────────────
    dur   = float(row.get('moving_time') or 0)
    dist  = float(row.get('distance_km') or 0) * 1000.0
    watts = row.get('np_watts') or 0
    if not (float(watts or 0) > 0):
        watts = row.get('avg_watts') or 0
    # Ohne Runden liegt nur der Gesamt-Höhenmeter-Wert vor; er wird je zur
    # Hälfte auf An- und Abstieg verteilt (Rundkurs-Annahme), weil
    # activities.csv Auf- und Abstieg nicht getrennt führt.
    elev = float(row.get('elevation_m') or 0)
    return [{
        'lap_number':   1,
        'duration_s':   dur,
        'distance_m':   dist,
        'avg_speed_ms': (dist / dur) if dur > 0 else 0.0,
        'avg_hr':       row.get('avg_hr'),
        'avg_power_w':  watts,
        'ascent_m':     elev,
        'descent_m':    elev,
        'cadence_spm':  None,
    }]


def calc_strength_load(row) -> float:
    """sRPE-Last für Kraft/Bouldern: RPE(0–10) × Dauer_min × k."""
    dur_h = float(row.get('moving_time') or 0) / 3600
    try:
        rpe = float(row.get('rpe') or 0)
    except (TypeError, ValueError):
        rpe = 0.0
    if not (rpe > 0):
        rpe = DEFAULT_RPE
    return round(dur_h * 60 * rpe * SRPE_TO_TSS, 1)


# Memo über die Rundenrechnung: dieselbe Aktivität wird pro Request mehrfach
# bewertet (erst df['tss'], dann compute_load, dann System-Split). Der Schlüssel
# enthält alle Felder, die ins Ergebnis eingehen, sodass eine editierte RPE oder
# ein Nachtrag aus dem Sync die Zeile automatisch neu rechnet.
_LOAD_MEMO = {}
_LOAD_MEMO_KEY = None


def _load_cache_key(row):
    return (row.get('id'), row.get('type'), row.get('moving_time'),
            row.get('distance_km'), row.get('elevation_m'), row.get('avg_hr'),
            row.get('avg_watts'), row.get('np_watts'), str(row.get('rpe', '')))


def calc_unit_load(row) -> dict:
    """L_aer und L_mech einer Aktivität inkl. Runden-Aufschlüsselung."""
    global _LOAD_MEMO_KEY
    try:
        laps_mtime = os.path.getmtime(GARMIN_LAPS_PATH)
    except OSError:
        laps_mtime = 0
    if _LOAD_MEMO_KEY != laps_mtime:          # Laps neu gesynct → Memo verwerfen
        _LOAD_MEMO.clear()
        _LOAD_MEMO_KEY = laps_mtime

    try:
        key = _load_cache_key(row)
        hit = _LOAD_MEMO.get(key)
        if hit is not None:
            return hit
    except TypeError:                          # unhashbare Zelle – dann eben ohne Memo
        key = None

    res = _calc_unit_load_uncached(row)
    if key is not None:
        _LOAD_MEMO[key] = res
    return res


def _calc_unit_load_uncached(row) -> dict:
    t = str(row.get('type', ''))
    if is_strength_type(t):
        load = calc_strength_load(row)
        return {'l_aer': load, 'l_mech': 0.0, 'n_laps': 0,
                'aer': {'l_aer': load, 'source': 'srpe', 'laps': [],
                        'jensen_gain': 0.0, 'if_mean': 0.0},
                'mech': {'l_mech': 0.0, 'model': 'none'}}

    res = ul.unit_load(activity_laps(row), t, LTHR, ftp=FTP, run_ftp=RUN_FTP)

    # Fallback wie bisher: ohne HF und ohne Power 50 Lasteinheiten pro Stunde.
    if res['l_aer'] <= 0:
        dur_h = float(row.get('moving_time') or 0) / 3600
        res['l_aer'] = round(dur_h * 50, 1)
        res['aer']['l_aer'] = res['l_aer']
        res['aer']['source'] = 'fallback'
    return res


def calc_tss(row):
    """Aerobe Last L_aer der Einheit (früher: TSS).

    Name und Signatur bleiben, weil `df['tss']` quer durch app.py und
    fable-dashboard/serve.py als Lastspalte verwendet wird.
    """
    return calc_unit_load(row)['l_aer']


def calc_mech(row):
    """Mechanische Last L_mech der Einheit."""
    return calc_unit_load(row)['l_mech']


# ═════════════════════════════════════════════════════════════════════════════
# Zustand – CTL / ATL / TSB / Mech-Last  (Bannister-Impuls-Antwort)
#
#   CTL_t = CTL_{t-1} + α_c·(L_t − CTL_{t-1})      α_c = 1/42   "Fitness"
#   ATL_t = ATL_{t-1} + α_a·(L_t − ATL_{t-1})      α_a = 1/7    "Ermüdung"
#   TSB_t = CTL_{t-1} − ATL_{t-1}                               "Form"
#
# L_t ist die aerobe Tageslast (Summe der L_aer aller Einheiten des Tages).
#
# Zwei bewusste Abweichungen von der vorherigen Implementierung:
#   • α = 1/τ statt k = 2/(τ+1). Die Impuls-Antwort ist über die Zeitkonstante
#     definiert (Bannister/TrainingPeaks), nicht über die Chart-EMA-Konvention;
#     das alte k reagierte rund doppelt so schnell wie τ = 42 es vorsieht.
#   • TSB nutzt die VORTAGESWERTE. Die Form, mit der man in den Tag geht, kann
#     nicht von der Einheit abhängen, die an diesem Tag erst noch kommt.
#
# Zusätzlich wird die mechanische Last mitgeführt:
#   Mech-Last_k = rollender 28-Tage-Mittelwert der täglichen L_mech × 7
#                 (= geglättete Wochenmenge, Grundlage des Budgets in der Kaskade)
#
# ── ATL/CTL Ratio  =  "Load Ratio" / "Monotony risk"
#    Ratio < 1.1  → well recovered
#    Ratio 1.1–1.3 → moderate load, monitor
#    Ratio > 1.3  → elevated injury risk (increase was too fast)
#    Ratio > 1.5  → deload immediately
#
# Reference: Coggan, A. (2003). "Training and Racing Using a Power Meter."
#            Bannister et al. (1975). "Systems model of training for athletic performance."
# ═════════════════════════════════════════════════════════════════════════════

def compute_load(df):
    """Tagesreihe CTL / ATL / TSB / L_mech / Mech-Last über die letzten 730 Tage.

    Rückgabe-Spalten: date, tss (= L_aer), l_mech, ctl, atl, tsb, mech_week.
    """
    df = df.copy()
    loads = df.apply(calc_unit_load, axis=1) if not df.empty else []
    df['tss']    = [l['l_aer'] for l in loads] if len(loads) else []
    df['l_mech'] = [l['l_mech'] for l in loads] if len(loads) else []

    # Pro Kalendertag summieren (mehrere Einheiten am selben Tag sind additiv)
    daily = (df.groupby('date')[['tss', 'l_mech']].sum().reset_index()
             if not df.empty else pd.DataFrame(columns=['date', 'tss', 'l_mech']))

    # Lückenloser Tagesindex, damit Ruhetage (Last 0) in die EMA eingehen –
    # ohne sie wäre die EMA systematisch zu optimistisch.
    today    = pd.Timestamp(datetime.now().date())
    all_days = pd.DataFrame({'date': pd.date_range(today - pd.Timedelta(days=730), today)})
    merged   = all_days.merge(daily, on='date', how='left').fillna(0)

    state = rd.ema_state(merged['tss'])
    merged['ctl'] = state['ctl'].values
    merged['atl'] = state['atl'].values
    merged['tsb'] = state['tsb'].values
    merged['mech_week'] = rd.mech_weekly(merged['l_mech']).values
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# Recovery Index  (Z-Score Methodik)
#
# R = clip(50 + 15 × (0.45·z_hrv + 0.25·z_rhr + 0.20·S − 0.10·z_atl), 0, 100)
#
#   z_hrv   = (RMSSD_heute − μ_7d) / σ_7d          [45 %]  ↑ gut
#   z_rhr   = −(RHR_heute  − μ_7d) / σ_7d           [25 %]  ↓ gut
#   S       = clip((Schlaf_h / 8) × Qualität, 0, 1)  [20 %]  ↑ gut
#   z_atl   = (ATL_heute − μ_7d_atl) / σ_7d_atl     [10 %]  ↑ schlecht
#
# Zonen: [0,33) Rot · [33,66) Gelb · [66,100] Grün
# ═════════════════════════════════════════════════════════════════════════════

def compute_recovery_index(garmin_row: dict, past_garmin_df=None,
                           atl_today: float = None, atl_7d=None) -> dict:
    """
    Returns dict with:
      recovery_index  float 0-100
      sub             dict of component scores
      raw             raw values used
    """
    import numpy as np

    def _f(key, default=None):
        try:
            v = float(garmin_row.get(key) or default)
            return None if math.isnan(v) else v
        except (TypeError, ValueError):
            return default

    # ── 1. HRV z-score (45 %) ────────────────────────────────────────────────
    # Priority: hrv_weekly_avg (most reliable from Garmin) → hrv_5min_high → hrv_last_night
    hrv = _f('hrv_weekly_avg') or _f('hrv_5min_high') or _f('hrv_last_night')
    hrv_field = ('hrv_weekly_avg' if _f('hrv_weekly_avg') is not None
                 else 'hrv_5min_high' if _f('hrv_5min_high') is not None
                 else 'hrv_last_night')
    if hrv is not None and past_garmin_df is not None and not past_garmin_df.empty:
        hist_hrv = pd.to_numeric(past_garmin_df[hrv_field], errors='coerce').dropna().tail(7)
        if len(hist_hrv) >= 3:
            z_hrv = (hrv - hist_hrv.mean()) / (hist_hrv.std() + 1e-6)
        else:
            z_hrv = 0.0
    elif hrv is not None:
        # single-point fallback: compare to Garmin baseline midpoint
        hrv_low  = _f('hrv_baseline_low')
        hrv_high = _f('hrv_baseline_high')
        if hrv_low and hrv_high:
            mid    = (hrv_low + hrv_high) / 2
            spread = (hrv_high - hrv_low) / 4 + 1e-6
            z_hrv  = (hrv - mid) / spread
        else:
            z_hrv = 0.0
    else:
        z_hrv = 0.0

    # ── 2. Resting HR z-score (25 %) ─────────────────────────────────────────
    rhr = _f('resting_hr')
    if rhr is not None and past_garmin_df is not None and not past_garmin_df.empty:
        hist_rhr = pd.to_numeric(past_garmin_df['resting_hr'], errors='coerce').dropna().tail(7)
        if len(hist_rhr) >= 3:
            z_rhr = -(rhr - hist_rhr.mean()) / (hist_rhr.std() + 1e-6)
        else:
            z_rhr = 0.0
    else:
        z_rhr = 0.0

    # ── 3. Sleep score S ∈ [0,1] (20 %) ─────────────────────────────────────
    sleep_total_s = _f('sleep_total_s')
    sleep_deep_s  = _f('sleep_deep_s')
    sleep_score_raw = _f('sleep_score')

    if sleep_total_s and sleep_total_s > 0:
        sleep_h = sleep_total_s / 3600.0
        if sleep_deep_s:
            deep_pct = sleep_deep_s / sleep_total_s
            # quality 0-1: 20 % deep sleep = perfect
            quality = min(deep_pct / 0.20, 1.0)
        else:
            quality = 0.85  # assume reasonable quality without data
        S = max(0.0, min(1.0, (sleep_h / 8.0) * quality))
    elif sleep_score_raw is not None:
        S = max(0.0, min(1.0, sleep_score_raw / 100.0))
    else:
        S = 0.5

    # ── 4. ATL z-score (10 %, optional) ──────────────────────────────────────
    if atl_today is not None and atl_7d is not None and len(atl_7d) >= 3:
        arr   = np.array(atl_7d, dtype=float)
        z_atl = (atl_today - arr.mean()) / (arr.std() + 1e-6)
    else:
        z_atl = 0.0

    # ── Combine ───────────────────────────────────────────────────────────────
    raw_score = 0.45 * z_hrv + 0.25 * z_rhr + 0.20 * S - 0.10 * z_atl
    ri = float(np.clip(50.0 + 15.0 * raw_score, 0.0, 100.0))

    return {
        'recovery_index': round(ri, 1),
        'sub': {
            'z_hrv':  round(z_hrv,  2),
            'z_rhr':  round(z_rhr,  2),
            'sleep_s': round(S,     2),
            'z_atl':  round(z_atl,  2),
        },
        'raw': {
            'hrv_used':       hrv,
            'hrv_field':      hrv_field,
            'resting_hr':     rhr,
            'sleep_total_h':  round(sleep_total_s / 3600, 1) if sleep_total_s else None,
            'sleep_score':    sleep_score_raw,
            'atl_today':      atl_today,
        },
    }


def _ri_label(ri: float) -> str:
    if ri >= 66: return 'Bereit'
    if ri >= 33: return 'Moderat'
    return 'Erholen'


def _get_garmin_row_for_date(date_str: str):
    """Return (garmin_row_dict, past_df) for a given YYYY-MM-DD date, or (None, None)."""
    if not os.path.exists(GARMIN_BODY_PATH):
        return None, None
    try:
        body_df  = pd.read_csv(GARMIN_BODY_PATH, dtype=str)
        day_rows = body_df[body_df['date'].astype(str).str[:10] == date_str]
        if day_rows.empty:
            return None, None
        past_df = body_df[body_df['date'].astype(str).str[:10] < date_str]
        return day_rows.iloc[0].to_dict(), past_df
    except Exception:
        return None, None


def _get_recent_garmin_row(max_back=5):
    """Return (row, past_df, date_str) for today or the most recent prior day with data."""
    base = datetime.now().date()
    for back in range(0, max_back + 1):
        d   = (base - pd.Timedelta(days=back)).strftime('%Y-%m-%d')
        row, past = _get_garmin_row_for_date(d)
        if row:
            return row, past, d
    return None, None, None


def _update_widget_csv():
    """Write ~/widget_data.csv — called after every sync so the Übersicht widget stays current."""
    try:
        # Heutige Garmin-Daten bevorzugen; sonst den letzten verfügbaren Tag nehmen,
        # damit das Widget nicht „—" zeigt, nur weil heute noch nicht gesynct wurde.
        garmin_row, past_df, ri_date = _get_recent_garmin_row()
        atl_today, atl_7d   = _atl_for_date(ri_date) if ri_date else (None, None)
        ri = None
        if garmin_row:
            ri = compute_recovery_index(garmin_row, past_df, atl_today, atl_7d)['recovery_index']

        weekly_tss, weekly_km = 0.0, 0.0
        df = load_activities()
        if not df.empty:
            df['tss']  = df.apply(calc_tss, axis=1)
            cutoff     = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=6)
            week_df    = df[df['date'] >= cutoff]
            run_types  = ('Run', 'TrailRun', 'VirtualRun')   # Wandern zählt nicht als Lauf
            weekly_tss = round(float(week_df['tss'].sum()), 1)
            weekly_km  = round(float(week_df[week_df['type'].isin(run_types)]['distance_km'].sum()), 1)

        with open(WIDGET_DATA_PATH, 'w') as f:
            f.write('recovery,tss,km\n')
            f.write(f"{ri if ri is not None else ''},{weekly_tss},{weekly_km}\n")
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Flask routes
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    # Kein Caching des Dashboards: verhindert, dass der Browser eine veraltete
    # HTML/JS-Version festhält (sonst „reagieren keine Klicks mehr" nach Updates).
    resp = send_from_directory(FRONTEND_DIR, HTML_FILE)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma']        = 'no-cache'
    resp.headers['Expires']       = '0'
    return resp


@app.route('/favicon.png')
def favicon():
    return send_from_directory(BASE_DIR, 'runner_icon.png')


# ── App-Lifecycle: Server beim Schließen des Browser-Tabs beenden ──────────────
# Token wird bei jedem Server-Neustart neu erzeugt → Frontend erkennt "frischer
# Start" (zurück zum Dashboard) gegenüber einem reinen Browser-Refresh.
SERVER_TOKEN     = os.urandom(8).hex()
_shutdown_timer  = None
_shutdown_lock   = threading.Lock()


def _arm_shutdown(seconds):
    global _shutdown_timer
    with _shutdown_lock:
        if _shutdown_timer:
            _shutdown_timer.cancel()
        t = threading.Timer(seconds, lambda: os._exit(0))
        t.daemon = True
        t.start()
        _shutdown_timer = t


@app.route('/api/server_token')
def api_server_token():
    return jsonify({'token': SERVER_TOKEN})


@app.route('/api/client_alive', methods=['GET', 'POST'])
def api_client_alive():
    # Heartbeat eines offenen Tabs: geplanten Shutdown verschieben.
    _arm_shutdown(6.0)
    return ('', 204)


@app.route('/api/client_gone', methods=['POST'])
def api_client_gone():
    # Tab wird geschlossen/neu geladen: Shutdown bald auslösen (ein Reload meldet
    # sich über /api/client_alive zurück und verschiebt ihn wieder).
    _arm_shutdown(4.0)
    return ('', 204)


def _atl_for_date(date_str: str):
    """Return (atl_today, atl_7d_list) for a given date, or (None, None)."""
    try:
        df = load_activities()
        if df.empty:
            return None, None
        df['tss'] = df.apply(calc_tss, axis=1)
        load_df   = compute_load(df)
        load_df['date_str'] = load_df['date'].astype(str).str[:10]
        idx = load_df[load_df['date_str'] == date_str].index
        if len(idx) == 0:
            return None, None
        pos       = load_df.index.get_loc(idx[0])
        atl_today = float(load_df.iloc[pos]['atl'])
        start     = max(0, pos - 7)
        atl_7d    = load_df.iloc[start:pos]['atl'].tolist()
        return atl_today, atl_7d
    except Exception:
        return None, None


@app.route('/api/recovery_index')
@app.route('/api/recovery_index/<date_str>')
def api_recovery_index(date_str=None):
    """Return Recovery Index for a given date (default: today)."""
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    garmin_row, past_df = _get_garmin_row_for_date(date_str)
    if garmin_row is None:
        return jsonify({'date': date_str, 'recovery_index': None, 'sub': {}, 'raw': {},
                        'label': 'Keine Daten'})
    atl_today, atl_7d = _atl_for_date(date_str)
    result = compute_recovery_index(garmin_row, past_df, atl_today, atl_7d)
    result['date']  = date_str
    result['label'] = _ri_label(result['recovery_index'])
    return jsonify(result)


@app.route('/api/widget_data')
def api_widget_data():
    """Compute today's widget values and write ~/widget_data.csv."""
    _update_widget_csv()
    try:
        with open(WIDGET_DATA_PATH) as f:
            lines = f.read().strip().split('\n')
        parts = lines[1].split(',') if len(lines) > 1 else ['', '0', '0']
        return jsonify({
            'status':   'ok',
            'recovery': float(parts[0]) if parts[0] else None,
            'tss':      float(parts[1]),
            'km':       float(parts[2]),
            'date':     datetime.now().strftime('%Y-%m-%d'),
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/garmin_calories')
def api_garmin_calories():
    """Return daily calorie data from Garmin CSV for the last N days."""
    days = min(int(request.args.get('days', 30)), 90)
    if not os.path.exists(GARMIN_BODY_PATH):
        return jsonify({'status': 'error', 'message': 'Garmin CSV nicht gefunden'}), 404
    try:
        df     = pd.read_csv(GARMIN_BODY_PATH)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df     = df.dropna(subset=['date']).sort_values('date')
        cutoff = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=days - 1)
        df     = df[df['date'] >= cutoff]

        def _iv(v):
            try:
                x = float(v)
                return 0 if math.isnan(x) else int(x)
            except Exception:
                return 0

        records = []
        for _, r in df.iterrows():
            records.append({
                'date':             str(r['date'])[:10],
                'active_calories':  _iv(r.get('active_calories', 0)),
                'bmr_calories':     _iv(r.get('bmr_calories', 0)),
                'total_calories':   _iv(r.get('total_calories', 0)),
            })

        today_str = datetime.now().date().isoformat()
        today_rec = next((x for x in records if x['date'] == today_str),
                         {'active_calories': 0, 'bmr_calories': 0, 'total_calories': 0})

        # 7-day rolling average of total_calories (for trend line)
        for i, rec in enumerate(records):
            window = [records[j]['total_calories'] for j in range(max(0, i - 6), i + 1)
                      if records[j]['total_calories'] > 0]
            rec['avg7'] = round(sum(window) / len(window)) if window else 0

        return jsonify({'status': 'ok', 'data': records, 'today': today_rec})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/sleep_data')
def api_sleep_data():
    """Return daily sleep data from Garmin CSV for the last N days."""
    days = min(int(request.args.get('days', 365)), 400)
    if not os.path.exists(GARMIN_BODY_PATH):
        return jsonify({'status': 'error', 'message': 'Garmin CSV nicht gefunden'}), 404
    try:
        df = pd.read_csv(GARMIN_BODY_PATH)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')
        cutoff = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=days - 1)
        df = df[df['date'] >= cutoff]

        def _fv(v):
            try:
                x = float(v)
                return None if math.isnan(x) else x
            except Exception:
                return None

        def _sv(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            return str(v)

        records = []
        for _, r in df.iterrows():
            records.append({
                'date':                 str(r['date'])[:10],
                'sleep_score':          _fv(r.get('sleep_score')),
                'sleep_total_s':        _fv(r.get('sleep_total_s')),
                'sleep_deep_s':         _fv(r.get('sleep_deep_s')),
                'sleep_light_s':        _fv(r.get('sleep_light_s')),
                'sleep_rem_s':          _fv(r.get('sleep_rem_s')),
                'sleep_awake_s':        _fv(r.get('sleep_awake_s')),
                'sleep_start_local':    _sv(r.get('sleep_start_local')),
                'sleep_end_local':      _sv(r.get('sleep_end_local')),
                'sleep_need_baseline_s': _fv(r.get('sleep_need_baseline_s')),
                'sleep_need_actual_s':  _fv(r.get('sleep_need_actual_s')),
                'sleep_debt_s':         _fv(r.get('sleep_debt_s')),
                'hrv_last_night':       _fv(r.get('hrv_last_night')),
                'hrv_weekly_avg':       _fv(r.get('hrv_weekly_avg')),
                'resting_hr':           _fv(r.get('resting_hr')),
                'body_battery_highest': _fv(r.get('body_battery_highest')),
                'body_battery_lowest':  _fv(r.get('body_battery_lowest')),
            })

        return jsonify({'status': 'ok', 'data': records})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/sleep_edit', methods=['POST'])
def api_sleep_edit():
    """Manually update sleep_start_local and sleep_end_local for a given date."""
    data        = request.get_json(force=True) or {}
    date_str    = data.get('date')         # YYYY-MM-DD
    sleep_start = data.get('sleep_start')  # ISO datetime e.g. "2026-06-30T23:45:00"
    sleep_end   = data.get('sleep_end')    # ISO datetime e.g. "2026-07-01T07:30:00"

    if not all([date_str, sleep_start, sleep_end]):
        return jsonify({'status': 'error', 'message': 'Fehlende Felder: date, sleep_start, sleep_end'}), 400
    if not os.path.exists(GARMIN_BODY_PATH):
        return jsonify({'status': 'error', 'message': 'Garmin CSV nicht gefunden'}), 404

    try:
        df = pd.read_csv(GARMIN_BODY_PATH)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        mask = df['date'].dt.strftime('%Y-%m-%d') == date_str
        if not mask.any():
            return jsonify({'status': 'error', 'message': f'Datum {date_str} nicht gefunden'}), 404

        df.loc[mask, 'sleep_start_local'] = sleep_start
        df.loc[mask, 'sleep_end_local']   = sleep_end

        try:
            start_dt = datetime.fromisoformat(sleep_start)
            end_dt   = datetime.fromisoformat(sleep_end)
            total_s  = (end_dt - start_dt).total_seconds()
            if 0 < total_s < 86400:
                df.loc[mask, 'sleep_total_s'] = total_s
        except Exception:
            pass

        df.to_csv(GARMIN_BODY_PATH, index=False)
        return jsonify({'status': 'ok', 'date': date_str})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/recipe_suggestion', methods=['POST'])
def api_recipe_suggestion():
    """AI-generated recipe suggestion tuned to current training load + recovery."""
    if _groq is None:
        return jsonify({'status': 'error', 'message': 'Groq nicht verfügbar'}), 503

    today_str = datetime.now().date().isoformat()
    df        = load_activities()

    # ── Training load ─────────────────────────────────────────────────────────
    if not df.empty:
        df['tss']  = df.apply(calc_tss, axis=1)
        load_df    = compute_load(df)
        today_row  = load_df[load_df['date'].astype(str).str[:10] == today_str]
        ctl   = float(today_row['ctl'].iloc[-1]) if not today_row.empty else 0.0
        atl   = float(today_row['atl'].iloc[-1]) if not today_row.empty else 0.0
        tsb   = float(today_row['tsb'].iloc[-1]) if not today_row.empty else 0.0
        # TSS trained today
        today_acts = df[df['date'].astype(str).str[:10] == today_str]
        tss_today  = int(today_acts['tss'].sum()) if not today_acts.empty else 0
    else:
        ctl = atl = tsb = tss_today = 0.0

    # ── Recovery Index ────────────────────────────────────────────────────────
    garmin_row, past_df = _get_garmin_row_for_date(today_str)
    if garmin_row:
        atl_today_ri, atl_7d_ri = _atl_for_date(today_str)
        ri_data = compute_recovery_index(garmin_row, past_df, atl_today_ri, atl_7d_ri)
        ri_val  = ri_data['recovery_index']
        ri_str  = f"{ri_val:.0f}/100 ({_ri_label(ri_val)})"
    else:
        ri_str  = 'Keine Garmin-Daten'

    # ── Calories burned today ─────────────────────────────────────────────────
    act_kcal = 0
    if garmin_row:
        try:
            act_kcal = int(float(garmin_row.get('active_calories') or 0))
        except Exception:
            pass

    trainer_profile = _load_trainer_profile()
    profile_section = f"\n\n{trainer_profile}" if trainer_profile else ""

    prompt = f"""Du bist mein Ernährungsberater und kennst meinen Trainingskontext.{profile_section}

**HEUTE – {today_str}**
- CTL (Fitness): {ctl:.0f} | ATL (Müdigkeit): {atl:.0f} | TSB (Form): {tsb:+.0f}
- TSS heute: {tss_today} | Recovery Index: {ri_str}
- Aktive Kalorien verbrannt heute: {act_kcal} kcal

Empfiehl mir ein konkretes, einfach kochbares Rezept das optimal zu meinem aktuellen Trainingszustand passt. Antworte ausschließlich als gültiges JSON, kein Text davor oder danach:
{{
  "name": "Rezeptname",
  "type": "Frühstück oder Mittagessen oder Abendessen oder Snack",
  "prep_time": "X min",
  "macros": {{"kcal": <Zahl>, "protein_g": <Zahl>, "carbs_g": <Zahl>, "fat_g": <Zahl>}},
  "focus": "proteinreich oder kohlenhydratreich oder kalorienreich oder ausgewogen",
  "ingredients": ["Zutat 1 mit Menge", "Zutat 2 mit Menge"],
  "steps": "Kurze Zubereitung in 2-3 Sätzen.",
  "why": "Ein Satz warum dieses Rezept jetzt physiologisch sinnvoll ist."
}}"""

    try:
        response = _groq.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model='llama-3.3-70b-versatile',
            max_tokens=400,
            temperature=0.6,
        )
        raw = response.choices[0].message.content.strip()
        m   = re.search(r'\{.*\}', raw, re.DOTALL)
        rec = json.loads(m.group() if m else raw)
        return jsonify({'status': 'ok', 'recipe': rec,
                        'context': {'ctl': ctl, 'atl': atl, 'tsb': tsb,
                                    'tss_today': tss_today, 'ri_str': ri_str}})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/wellness_suggestion', methods=['POST'])
def api_wellness_suggestion():
    """AI wellness/meditation suggestion tuned to recovery state and training load."""
    if _groq is None:
        return jsonify({'status': 'error', 'message': 'Groq nicht verfügbar'}), 503

    today_str = datetime.now().date().isoformat()
    df        = load_activities()

    if not df.empty:
        df['tss'] = df.apply(calc_tss, axis=1)
        load_df   = compute_load(df)
        today_row = load_df[load_df['date'].astype(str).str[:10] == today_str]
        ctl  = float(today_row['ctl'].iloc[-1]) if not today_row.empty else 0.0
        atl  = float(today_row['atl'].iloc[-1]) if not today_row.empty else 0.0
        tsb  = float(today_row['tsb'].iloc[-1]) if not today_row.empty else 0.0
    else:
        ctl = atl = tsb = 0.0

    garmin_row, past_df = _get_garmin_row_for_date(today_str)
    if garmin_row:
        atl_today_ri, atl_7d_ri = _atl_for_date(today_str)
        ri_data = compute_recovery_index(garmin_row, past_df, atl_today_ri, atl_7d_ri)
        ri_val  = ri_data['recovery_index']
        ri_str  = f"{ri_val:.0f}/100 ({_ri_label(ri_val)})"
        sub     = ri_data['sub']
        sleep_s = sub.get('sleep', 50)
        hrv_s   = sub.get('hrv', 50)
    else:
        ri_str = 'Keine Garmin-Daten'
        sleep_s = hrv_s = 50

    trainer_profile = _load_trainer_profile()
    profile_section = f"\n\n{trainer_profile}" if trainer_profile else ""

    prompt = f"""Du bist mein Wellness-Coach.{profile_section}

**MEIN HEUTIGER ZUSTAND – {today_str}**
- Recovery Index: {ri_str}
- Schlaf-Score: {sleep_s:.0f}/100 | HRV-Score: {hrv_s:.0f}/100
- CTL: {ctl:.0f} | ATL: {atl:.0f} | TSB: {tsb:+.0f}

Empfiehl mir eine konkrete Wellness-/Regenerationsaktivität für heute die zu meinem Erholungszustand passt. Mögliche Typen: Meditation, Atemübung, Yoga, Foam Rolling, Kältebad, Sauna, Spaziergang, Stretching, Körperwahrnehmung.

Antworte ausschließlich als gültiges JSON, kein Text davor oder danach:
{{
  "name": "Name der Aktivität",
  "type": "z.B. Meditation oder Atemübung oder Yoga",
  "emoji": "passendes Emoji",
  "duration": "X Minuten",
  "intensity": "sanft oder moderat oder aktiv",
  "instructions": "Konkrete Schritt-für-Schritt Anleitung in 3-4 Sätzen.",
  "why": "Ein Satz warum diese Aktivität jetzt für meine Erholung optimal ist."
}}"""

    try:
        response = _groq.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model='llama-3.3-70b-versatile',
            max_tokens=300,
            temperature=0.65,
        )
        raw = response.choices[0].message.content.strip()
        m   = re.search(r'\{.*\}', raw, re.DOTALL)
        sug = json.loads(m.group() if m else raw)
        return jsonify({'status': 'ok', 'suggestion': sug,
                        'context': {'ri_str': ri_str, 'tsb': tsb}})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/sync', methods=['POST'])
def api_sync():
    """Aktivitäten aus den (bereits gesyncten) Garmin-Daten in activities.csv
    einspeisen. Die frühere Strava-Anbindung ist deaktiviert (App-Status Inactive);
    neue Aktivitäten kommen ausschließlich aus Garmin. Für den vollständigen
    Abruf frischer Garmin-Daten dient /api/garmin_sync."""
    try:
        df = merge_garmin_activities_into_csv()
        _update_widget_csv()
        return jsonify({'status': 'ok', 'activities': len(df), 'mode': 'garmin'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/activity_days')
def api_activity_days():
    """Map date_iso → 'run'|'bike'|'rest' für ALLE Aktivitäten (für den Streak-Kalender)."""
    df = load_csv()
    out = {}
    if not df.empty:
        run_types  = ('Run', 'TrailRun', 'VirtualRun')   # Wandern/Gehen → 'rest' (Sonstige)
        ride_types = ('Ride', 'VirtualRide', 'GravelRide', 'EBikeRide', 'MountainBikeRide')
        for _, r in df.sort_values('date').iterrows():
            d  = r['date']
            ds = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
            t  = str(r.get('type', ''))
            out[ds] = 'bike' if t in ride_types else ('run' if t in run_types else 'rest')
    return jsonify(out)


@app.route('/api/recent_volume')
def api_recent_volume():
    """Leichte Aktivitätsliste der letzten 24 Wochen für das Wochenvolumen (Strava-Stil)."""
    df = load_activities()
    out = []
    if not df.empty:
        today      = pd.Timestamp(datetime.now().date())
        this_mon   = today - pd.Timedelta(days=today.weekday())
        cutoff     = this_mon - pd.Timedelta(weeks=23)
        recent     = df[df['date'] >= cutoff]
        for _, r in recent.sort_values('date').iterrows():
            d = r['date']
            out.append({
                'date': d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10],
                'type': str(r.get('type', '')),
                'dist': round(float(r.get('distance_km', 0) or 0), 2),
                'time': int(r.get('moving_time', 0) or 0),
                'elev': int(float(r.get('elevation_m', 0) or 0)),
            })
    return jsonify(out)


@app.route('/api/metrics')
def api_metrics():
    """
    Return current training state as JSON:
    - metrics: CTL, ATL, TSB, ratio, CTL delta
    - sparkbars: 7-day history for each metric (0–100 normalised)
    - chart: 10 weekly buckets with TSS, CTL, ATL
    - activities: last 8 workouts
    - last_sync: timestamp of CSV file
    """
    df = load_activities()
    if df.empty:
        return jsonify({
            'error':   'no_data',
            'message': 'Keine Daten. Bitte POST /api/sync aufrufen.',
        })

    df['tss'] = df.apply(calc_tss, axis=1)
    load      = compute_load(df)
    today_row = load.iloc[-1]
    week_ago  = load.iloc[-8] if len(load) >= 8 else load.iloc[0]

    ctl   = today_row['ctl']
    atl   = today_row['atl']
    tsb   = today_row['tsb']
    ratio = round(atl / ctl, 2) if ctl > 0 else 0.0

    # Sparkbar: 7 values normalised to 0–100 for bar height
    def sparkvals(series, n=7):
        vals = series.tail(n).tolist()
        mx   = max(abs(v) for v in vals) or 1
        return [round(abs(v) / mx * 100) for v in vals]

    # Chart: 10 weekly buckets (Mon–Sun ISO weeks, most recent last)
    today        = pd.Timestamp(datetime.now().date())
    this_monday  = today - pd.Timedelta(days=today.weekday())
    chart = []
    for i in range(9, -1, -1):
        wk_start = this_monday - pd.Timedelta(weeks=i)
        wk_end   = wk_start + pd.Timedelta(days=6)
        w     = load[(load['date'] >= wk_start) & (load['date'] <= wk_end)]
        w_df  = df[(df['date'] >= wk_start) & (df['date'] <= wk_end)]
        run_types  = ('Run', 'TrailRun', 'VirtualRun')   # Wandern ist eine eigene Kategorie
        hike_types = ('Hike', 'Walk')
        ride_types = ('Ride', 'VirtualRide', 'GravelRide', 'EBikeRide', 'MountainBikeRide')
        run_km  = w_df[w_df['type'].isin(run_types)]['distance_km'].sum()
        hike_km = w_df[w_df['type'].isin(hike_types)]['distance_km'].sum()
        ride_km = w_df[w_df['type'].isin(ride_types)]['distance_km'].sum()
        # Per-workout TSS contributions (chronological) → stacked-bar segments
        _DOW        = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
        w_sorted    = w_df.sort_values('date')
        workout_tss = [round(float(t), 1) for t in w_sorted['tss'].tolist()]
        workout_days = [_DOW[d.weekday()] if hasattr(d, 'weekday') else ''
                        for d in w_sorted['date'].tolist()]
        chart.append({
            'label':      f"KW {wk_end.isocalendar()[1]}",
            'date_label': f"{wk_start.day}.{wk_start.month}.–{wk_end.day}.{wk_end.month}.",
            'tss':   round(float(w['tss'].sum()), 1),
            # Die beiden Lastkanäle der Woche: L_aer ist identisch mit 'tss'
            # (die aerobe Last IST die Lastspalte), L_mech kommt getrennt dazu.
            'l_aer':  round(float(w['tss'].sum()), 1),
            'l_mech': round(float(w['l_mech'].sum()), 1) if 'l_mech' in w else 0.0,
            'mech_week': round(float(w['mech_week'].iloc[-1]), 1) if len(w) and 'mech_week' in w else 0.0,
            'ctl':   round(float(w['ctl'].iloc[-1]), 1) if len(w) else 0,
            'atl':   round(float(w['atl'].iloc[-1]), 1) if len(w) else 0,
            'km':    round(float(w_df['distance_km'].sum()), 1) if len(w_df) else 0,
            'run_km':  round(float(run_km), 1),
            'hike_km': round(float(hike_km), 1),
            'ride_km': round(float(ride_km), 1),
            'workout_tss': workout_tss,
            'workout_days': workout_days,
        })

    # Streak: consecutive weeks (ending this week) with >= 1 activity
    streak = 0
    for i in range(0, 104):
        wk_start = this_monday - pd.Timedelta(weeks=i)
        wk_end   = wk_start + pd.Timedelta(days=6)
        w_df  = df[(df['date'] >= wk_start) & (df['date'] <= wk_end)]
        if len(w_df) > 0:
            streak += 1
        else:
            if i == 0:
                continue  # current week may still be empty, don't break streak yet
            break

    # Day streak: consecutive days (ending today) with >= 1 activity.
    # Today may still be empty (grace day) → it doesn't break the streak,
    # we just start counting from yesterday in that case.
    act_days   = set(pd.to_datetime(df['date']).dt.normalize())
    day_streak = 0
    cur = today if today in act_days else today - pd.Timedelta(days=1)
    while cur in act_days:
        day_streak += 1
        cur -= pd.Timedelta(days=1)

    # Recent activities (last 8)
    TYPE_ICON = dict(ACTIVITY_ICONS)
    recent = df.sort_values('date', ascending=False).head(8)
    activities_out = []
    for _, r in recent.iterrows():
        t    = str(r.get('type', ''))
        pace = ''
        if t in ('Run', 'TrailRun', 'VirtualRun') and r['avg_speed_kmh'] > 0:
            mpm  = 60 / r['avg_speed_kmh']
            pace = f"{int(mpm)}:{int((mpm % 1) * 60):02d} /km"
        elif r['avg_watts'] > 0:
            pace = f"{int(r['avg_watts'])} W"

        mt       = int(r.get('moving_time', 0))
        date_val = r['date']
        activities_out.append({
            'name':      r['name'],
            'type':      t,
            'icon':      TYPE_ICON.get(t, '🏃'),
            'date':      date_val.strftime('%a, %d. %b') if hasattr(date_val, 'strftime') else str(date_val)[:10],
            'date_fmt':  date_val.strftime('%a, %d. %b %Y') if hasattr(date_val, 'strftime') else str(date_val)[:10],
            'date_iso':  date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val)[:10],
            'dist':      round(float(r['distance_km']), 1),
            'duration':  f"{mt // 3600}:{(mt % 3600) // 60:02d}:{mt % 60:02d}",
            'pace':      pace,
            'tss':       int(r['tss']),          # = L_aer, Name bleibt fürs Frontend
            'l_aer':     round(float(r['tss']), 1),
            'l_mech':    round(calc_mech(r), 1),
            'hr':        int(r['avg_hr']) if r.get('avg_hr', 0) > 0 else None,
            'max_hr':    int(r['max_hr']) if r.get('max_hr', 0) > 0 else None,
            'elevation': int(r['elevation_m']) if r.get('elevation_m', 0) > 0 else None,
            'avg_watts': int(r['avg_watts']) if r.get('avg_watts', 0) > 0 else None,
            'np_watts':  int(r['np_watts']) if r.get('np_watts', 0) > 0 else None,
        })

    # Last sync timestamp
    last_sync = '—'
    if os.path.exists(CSV_PATH):
        last_sync = datetime.fromtimestamp(
            os.path.getmtime(CSV_PATH)
        ).strftime('%d.%m.%Y %H:%M')

    # Calendar dots: activity type by date for the last 90 days
    cal_start = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=90)
    cal_df    = df[df['date'] >= cal_start][['date', 'type']].copy()
    _TYPE_CAL = {
        'Run': 'run', 'TrailRun': 'run', 'VirtualRun': 'run',
        'Ride': 'bike', 'VirtualRide': 'bike', 'GravelRide': 'bike',
        'EBikeRide': 'bike', 'MountainBikeRide': 'bike',
    }
    cal_dots = {}
    for _, row in cal_df.iterrows():
        d = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
        cal_dots[d] = _TYPE_CAL.get(str(row['type']), 'other')

    # Planned sessions dots
    planned_df   = _load_planned()
    planned_dots = {}
    for _, row in planned_df.iterrows():
        d = str(row.get('date', ''))[:10]
        if d and len(d) == 10:
            planned_dots[d] = 'plan'

    return jsonify({
        'streak': streak,
        'day_streak': day_streak,
        'metrics': {
            'ctl':       round(float(ctl), 1),
            'ctl_delta': round(float(ctl - week_ago['ctl']), 1),
            'atl':       round(float(atl), 1),
            'tsb':       round(float(tsb), 1),
            'ratio':     ratio,
        },
        'sparkbars': {
            'ctl': sparkvals(load['ctl']),
            'atl': sparkvals(load['atl']),
            'tsb': sparkvals(load['tsb']),
        },
        'chart':        chart,
        'activities':   activities_out,
        'last_sync':    last_sync,
        'calendar_dots': cal_dots,
        'planned_dots':  planned_dots,
    })


def _safe_poly(val) -> str:
    """Return a clean polyline string, or '' if the value is NaN / None / empty."""
    if val is None:
        return ''
    try:
        if isinstance(val, float) and math.isnan(val):
            return ''
    except Exception:
        pass
    s = str(val).strip()
    return '' if s in ('', 'nan', 'None', 'NaN') else s


def _rpe_val(val):
    """Geloggte RPE als 0–10-Zahl, oder None wenn leer/ungültig."""
    if val is None:
        return None
    s = str(val).strip().replace(',', '.')
    if s in ('', 'nan', 'None', 'NaN'):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return round(f, 1) if f > 0 else None


def _no_nan(d):
    """Ersetze float NaN durch None, damit der Browser-JSON.parse nicht scheitert."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


@app.route('/api/activities')
def api_all_activities():
    """Return activities (full fields) for the Einheiten section, newest first.
    ?limit=N (default 50) or ?limit=all für die komplette Historie."""
    _backfill_activity_intentions()
    df = load_activities()
    if df.empty:
        return jsonify([])
    df['tss'] = df.apply(calc_tss, axis=1)
    recent = df.sort_values('date', ascending=False)
    limit  = request.args.get('limit', '50')
    if limit != 'all':
        try:
            recent = recent.head(int(limit))
        except (TypeError, ValueError):
            recent = recent.head(50)
    _ICON = ACTIVITY_ICONS
    result = []
    for _, r in recent.iterrows():
        t    = str(r.get('type', ''))
        pace = ''
        if t in ('Run', 'TrailRun', 'VirtualRun') and r.get('avg_speed_kmh', 0) > 0:
            mpm  = 60 / r['avg_speed_kmh']
            pace = f"{int(mpm)}:{int((mpm % 1) * 60):02d} /km"
        elif r.get('avg_watts', 0) > 0:
            pace = f"{int(r['avg_watts'])} W"
        mt       = int(r.get('moving_time', 0))
        date_val = r['date']
        result.append(_no_nan({
            'id':        int(r['id']) if r.get('id') else None,
            'name':      '' if pd.isna(r.get('name')) else str(r.get('name')),
            'type':      t,
            'icon':      _ICON.get(t, '🏃'),
            'date_fmt':  date_val.strftime('%a, %d. %b %Y') if hasattr(date_val, 'strftime') else str(date_val)[:10],
            'date_iso':  date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val)[:10],
            'dist':      round(float(r.get('distance_km', 0)), 1),
            'duration':  f"{mt // 3600}:{(mt % 3600) // 60:02d}:{mt % 60:02d}",
            'pace':      pace,
            'avg_speed': round(float(r['avg_speed_kmh']), 1) if r.get('avg_speed_kmh', 0) > 0 else None,
            'tss':       int(r['tss']),          # = L_aer, Name bleibt fürs Frontend
            'l_aer':     round(float(r['tss']), 1),
            'l_mech':    round(calc_mech(r), 1),
            'hr':        int(r['avg_hr'])    if r.get('avg_hr', 0)    > 0 else None,
            'max_hr':    int(r['max_hr'])    if r.get('max_hr', 0)    > 0 else None,
            'elevation': int(r['elevation_m']) if r.get('elevation_m', 0) > 0 else None,
            'avg_watts': int(r['avg_watts']) if r.get('avg_watts', 0) > 0 else None,
            'np_watts':  int(r['np_watts'])  if r.get('np_watts', 0)  > 0 else None,
            'polyline':     _safe_poly(r.get('polyline')),
            'comment':      str(r.get('comment', '') or ''),
            'ai_analysis':  str(r.get('ai_analysis', '') or ''),
            'intention':    str(r.get('intention', '') or ''),
            'intention_met': str(r.get('intention_met', '') or ''),
            'rpe':          _rpe_val(r.get('rpe')),
            'is_strength':  is_strength_type(t),
        }))
    return jsonify(result)


@app.route('/api/activities/<activity_id>', methods=['PATCH'])
def api_patch_activity(activity_id):
    """Update editable fields for a single activity.
    Negative IDs are manuelle Kraft-/Boulder-Einträge (load_activities() gibt
    ihnen -entry_id, siehe _non_cardio_to_activity_rows) -> die werden in
    non_cardio_activities.csv statt activities.csv geschrieben."""
    try:
        activity_id = int(activity_id)
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'message': 'Ungültige ID'}), 400

    if activity_id < 0:
        return _patch_non_cardio_activity(-activity_id, request.get_json() or {})

    df = load_csv()
    if df.empty:
        return jsonify({'status': 'error', 'message': 'Keine Aktivitäten'}), 404
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    mask = df['id'] == activity_id
    if not mask.any():
        return jsonify({'status': 'error', 'message': 'Nicht gefunden'}), 404
    data = request.get_json() or {}
    if 'name' in data:
        df.loc[mask, 'name'] = str(data['name']).strip()
    if 'comment' in data:
        if 'comment' not in df.columns:
            df['comment'] = ''
        df.loc[mask, 'comment'] = str(data['comment'])
    if 'intention' in data:
        if 'intention' not in df.columns:
            df['intention'] = ''
        df.loc[mask, 'intention'] = str(data['intention'])
    if 'intention_met' in data:
        if 'intention_met' not in df.columns:
            df['intention_met'] = ''
        df.loc[mask, 'intention_met'] = str(data['intention_met'])
    if 'rpe' in data:
        if 'rpe' not in df.columns:
            df['rpe'] = ''
        # RPE 0–10 als String gespeichert; leer löscht den Wert (Fallback greift)
        raw = str(data['rpe']).strip().replace(',', '.')
        try:
            val = '' if raw == '' else str(round(max(0.0, min(10.0, float(raw))), 1))
        except ValueError:
            val = ''
        df['rpe'] = df['rpe'].astype('object')
        df.loc[mask, 'rpe'] = val
    df.to_csv(CSV_PATH, index=False)
    return jsonify({'status': 'ok'})


@app.route('/api/activities/<activity_id>', methods=['DELETE'])
def api_delete_activity(activity_id):
    """Löscht eine Aktivität endgültig.

    Negative IDs sind manuelle Kraft-/Boulder-Einträge (non_cardio_activities.csv),
    positive IDs sind Strava-/Garmin-Aktivitäten (activities.csv). Da CTL/ATL/TSB/TSS
    bei jedem Request live aus den CSVs berechnet werden, verschwindet die Einheit
    damit automatisch aus allen Trainingsdaten. Gecachte Stream-Dateien werden mit
    entfernt."""
    try:
        activity_id = int(activity_id)
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'message': 'Ungültige ID'}), 400

    # Manuelle Kraft-/Boulder-Einträge (negative IDs) → non_cardio_activities.csv
    if activity_id < 0:
        entry_id = -activity_id
        df   = _load_non_cardio()
        mask = df['id'] == entry_id
        if not mask.any():
            return jsonify({'status': 'error', 'message': 'Nicht gefunden'}), 404
        df = df[~mask]
        df.to_csv(NON_CARDIO_PATH, index=False)
        return jsonify({'status': 'ok'})

    # Normale Aktivitäten → activities.csv
    df = load_csv()
    if df.empty:
        return jsonify({'status': 'error', 'message': 'Keine Aktivitäten'}), 404
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    mask = df['id'] == activity_id
    if not mask.any():
        return jsonify({'status': 'error', 'message': 'Nicht gefunden'}), 404
    df = df[~mask]
    df.to_csv(CSV_PATH, index=False)

    # Tombstone setzen, damit ein späterer Sync die Aktivität nicht erneut von
    # Strava einliest (in Strava selbst bleibt sie bestehen)
    _add_deleted_id(activity_id)

    # Gecachte Stream-Daten (GPS/HF/…) der gelöschten Aktivität aufräumen
    for d in (STREAMS_DIR, os.path.join(DB_DIR, 'streams_raw')):
        try:
            p = os.path.join(d, f'{activity_id}.json')
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass

    return jsonify({'status': 'ok'})


@app.route('/api/activities/<int:activity_id>/ai_analysis', methods=['POST'])
def api_ai_analysis(activity_id):
    """Generate (or return cached) AI trainer analysis for a single workout."""
    if _groq is None:
        return jsonify({'status': 'error',
                        'message': 'GROQ_API_KEY nicht gesetzt (Umgebungsvariable).'}), 503

    df = load_csv()
    if df.empty:
        return jsonify({'status': 'error', 'message': 'Keine Aktivitäten'}), 404

    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    mask = df['id'] == activity_id
    if not mask.any():
        return jsonify({'status': 'error', 'message': 'Aktivität nicht gefunden'}), 404

    row = df[mask].iloc[0]

    body  = request.get_json() or {}
    force = body.get('force', False)
    notes = str(body.get('notes', '') or '').strip()
    existing = str(row.get('ai_analysis', '') or '').strip()
    if existing and not force:
        return jsonify({'status': 'cached', 'analysis': existing})

    # ── Compute training load context ────────────────────────────────────────
    df['tss'] = df.apply(calc_tss, axis=1)
    load_df   = compute_load(df)
    act_date  = str(row['date'])[:10]
    load_row  = load_df[load_df['date'].astype(str).str[:10] == act_date]
    ctl = float(load_row['ctl'].iloc[-1]) if not load_row.empty else 0.0
    atl = float(load_row['atl'].iloc[-1]) if not load_row.empty else 0.0
    tsb = float(load_row['tsb'].iloc[-1]) if not load_row.empty else 0.0

    # ── Recovery Index for that day ──────────────────────────────────────────
    garmin_row, past_df = _get_garmin_row_for_date(act_date)
    if garmin_row:
        atl_today_ri, atl_7d_ri = _atl_for_date(act_date)
        ri_data = compute_recovery_index(garmin_row, past_df, atl_today_ri, atl_7d_ri)
        ri_val  = ri_data['recovery_index']
        ri_str  = f"{ri_val:.0f}/100 ({_ri_label(ri_val)})"
    else:
        ri_str  = 'Keine Garmin-Daten verfügbar'
        ri_val  = None

    # ── Other workouts that day ──────────────────────────────────────────────
    same_day = df[(df['date'].astype(str).str[:10] == act_date) & (df['id'] != activity_id)]
    if same_day.empty:
        other_str = 'Keine weiteren Trainings an diesem Tag.'
    else:
        parts = []
        for _, o in same_day.iterrows():
            mt = int(o.get('moving_time', 0))
            parts.append(
                f"- {o['name']} ({o['type']}): {float(o.get('distance_km', 0)):.1f} km, "
                f"{mt // 60} min, Ø HF {int(o.get('avg_hr', 0)) or '—'} bpm, "
                f"TSS {int(o.get('tss', 0))}"
            )
        other_str = '\n'.join(parts)

    # ── Format activity fields ────────────────────────────────────────────────
    t    = str(row.get('type', ''))
    mt   = int(row.get('moving_time', 0))
    dur  = f"{mt // 3600}h {(mt % 3600) // 60}min" if mt >= 3600 else f"{mt // 60} min"
    tss_val = int(df[mask]['tss'].iloc[0])

    avg_spd = float(row.get('avg_speed_kmh', 0) or 0)
    if t in ('Run', 'TrailRun', 'VirtualRun') and avg_spd > 0:
        mpm      = 60 / avg_spd
        pace_str = f"{int(mpm)}:{int((mpm % 1) * 60):02d} min/km"
    elif float(row.get('avg_watts', 0) or 0) > 0:
        pace_str = f"Ø {int(row['avg_watts'])} W (NP: {int(row.get('np_watts', 0) or 0)} W)"
    else:
        pace_str = '—'

    power_section = ''
    avg_w = float(row.get('avg_watts', 0) or 0)
    np_w  = float(row.get('np_watts', 0) or 0)
    if avg_w > 0:
        intensity_factor = round((np_w or avg_w) / FTP, 2)
        power_section = (
            f"\n- Ø Leistung: {int(avg_w)} W"
            f"\n- Normierte Leistung (NP): {int(np_w)} W"
            f"\n- Intensity Factor (IF): {intensity_factor}"
        )

    trainer_profile = _load_trainer_profile()
    profile_section = f"\n\n**MEIN TRAINERPROFIL**\n{trainer_profile}" if trainer_profile else ""
    notes_section   = f"\n\n**MEINE NOTIZEN ZU DIESER EINHEIT**\n{notes}" if notes else ""

    # ── Makroplan context ─────────────────────────────────────────────────────
    current_block = _get_current_makro_block()
    if current_block and current_block['phase'] != 'Pre':
        block_str = (
            f"{current_block['block_name']} ({current_block['phase']}) – "
            f"Woche {current_block['week_num']}/{current_block['total_weeks']} – "
            f"TSS-Ziel {current_block['tss_target']:.0f}/Woche"
        )
    elif current_block:
        block_str = f"Vor dem Plan – Plan startet {current_block['start_date']}"
    else:
        block_str = 'Kein Makroplan verfügbar'
    makro_summary = _makroplan_summary_str()

    prompt = f"""Du bist mein persönlicher Trainer. Analysiere dieses Training auf Deutsch.{profile_section}

**TRAININGSPLAN – AKTUELLER BLOCK**
{block_str}

**MAKROPLAN ÜBERSICHT**
{makro_summary}

**KONTEXT**
- LTHR: {LTHR} bpm | Max-HF: {HR_MAX} bpm | FTP: {FTP} W
- Zonen: Z1<{ZONE_BPM[1][1]} | Z2 {ZONE_BPM[2][0]}–{ZONE_BPM[2][1]} | Z3 {ZONE_BPM[3][0]}–{ZONE_BPM[3][1]} | Z4 {ZONE_BPM[4][0]}–{ZONE_BPM[4][1]} | Z5>{ZONE_BPM[5][0]} bpm
- CTL {ctl:.0f} | ATL {atl:.0f} | TSB {tsb:+.0f} | Recovery Index: {ri_str}

**TRAINING**
{row['name']} · {t} · {act_date} · {dur} · {float(row.get('distance_km', 0)):.1f} km
Ø HF {int(row.get('avg_hr', 0) or 0) or '—'} bpm · Max {int(row.get('max_hr', 0) or 0) or '—'} bpm · {pace_str} · TSS {tss_val}{power_section}
Weitere Einheiten heute: {other_str}{notes_section}

---
Antworte mit genau vier Sätzen, kein Markdown, kein Titel, einfacher Fließtext:
Satz 1 – Zone & System: In welcher Zone war das, welches Energiesystem wurde trainiert, und passt das zu meinem polarisierten Prinzip und aktuellen Trainingsblock?
Satz 2 – Trainingseffekt: Was adaptiert physiologisch und in welchem Zeitrahmen?
Satz 3 – Erholungskontext: War die Intensität angesichts Recovery Index und TSB sinnvoll oder riskant?
Satz 4 – Empfehlung: Was konkret als nächstes, bezogen auf den aktuellen Trainingsblock?"""

    try:
        response = _groq.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model='llama-3.3-70b-versatile',
            max_tokens=380,
            temperature=0.55,
        )
        analysis = response.choices[0].message.content.strip()

        df2 = load_csv()
        df2['id'] = pd.to_numeric(df2['id'], errors='coerce')
        mask2 = df2['id'] == activity_id
        if mask2.any():
            if 'ai_analysis' not in df2.columns:
                df2['ai_analysis'] = ''
            df2.loc[mask2, 'ai_analysis'] = analysis
            df2.to_csv(CSV_PATH, index=False)

        return jsonify({'status': 'ok', 'analysis': analysis})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/training_recommendation', methods=['POST'])
def api_training_recommendation():
    """AI-powered training recommendation for today based on RI, last 7 workouts, CTL/ATL/TSB."""
    if _groq is None:
        return jsonify({'status': 'error', 'message': 'Groq nicht verfügbar'}), 503

    req_data = request.get_json(silent=True) or {}
    feedback = str(req_data.get('feedback', '') or '').strip()
    previous = req_data.get('previous')

    df = load_activities()
    today_str = datetime.now().date().isoformat()

    # ── Training load for today ───────────────────────────────────────────────
    if not df.empty:
        df['tss'] = df.apply(calc_tss, axis=1)
        load_df   = compute_load(df)
        today_row = load_df[load_df['date'].astype(str).str[:10] == today_str]
        ctl   = float(today_row['ctl'].iloc[-1]) if not today_row.empty else 0.0
        atl   = float(today_row['atl'].iloc[-1]) if not today_row.empty else 0.0
        tsb   = float(today_row['tsb'].iloc[-1]) if not today_row.empty else 0.0
        ratio = round(atl / ctl, 2) if ctl > 0 else 0.0
    else:
        ctl = atl = tsb = ratio = 0.0

    # ── Recovery Index for today ──────────────────────────────────────────────
    garmin_row, past_df = _get_garmin_row_for_date(today_str)
    if garmin_row:
        atl_today_ri, atl_7d_ri = _atl_for_date(today_str)
        ri_data  = compute_recovery_index(garmin_row, past_df, atl_today_ri, atl_7d_ri)
        ri_val   = ri_data['recovery_index']
        ri_str   = f"{ri_val:.0f}/100 ({_ri_label(ri_val)})"
        sub      = ri_data['sub']
        sub_str  = (f"HRV {sub.get('hrv',0):.0f} | Schlaf {sub.get('sleep',0):.0f} | "
                    f"Body Battery {sub.get('body_battery',0):.0f} | RHR {sub.get('rhr',0):.0f} | "
                    f"Stress {sub.get('stress',0):.0f}")
    else:
        ri_val  = None
        ri_str  = 'Keine Garmin-Daten für heute'
        sub_str = '—'

    # ── Aktive Verletzungen ────────────────────────────────────────────────────
    inj_df = _load_injuries()
    if not inj_df.empty:
        active_inj = inj_df[inj_df['status'].astype(str).str.lower() != 'resolved']
    else:
        active_inj = inj_df
    if active_inj.empty:
        injuries_str = 'Keine aktiven Verletzungen.'
    else:
        lines = []
        for _, i in active_inj.iterrows():
            lines.append(
                f"  {i.get('date','?')} | {i.get('body_part','?')} | {i.get('title','')} "
                f"| Schweregrad {i.get('severity','?')}/5 | {i.get('description','')}"
            )
        injuries_str = '\n'.join(lines)

    # ── Last 7 workouts ───────────────────────────────────────────────────────
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        last7 = df.sort_values('date').tail(7)
        workouts_lines = []
        for _, w in last7.iterrows():
            mt   = int(w.get('moving_time', 0) or 0)
            dist = float(w.get('distance_km', 0) or 0)
            hr   = int(w.get('avg_hr', 0) or 0)
            tss_w = int(w.get('tss', 0) or 0)
            wdate = str(w['date'])[:10]
            workouts_lines.append(
                f"  {wdate} | {w.get('type','?'):12s} | {w.get('name','')[:28]:28s} | "
                f"{dist:5.1f}km | {mt//60:3d}min | HF {hr or '—':>3} bpm | TSS {tss_w}"
            )
        workouts_str = '\n'.join(workouts_lines)
    else:
        workouts_str = '  Keine Einträge vorhanden.'

    trainer_profile = _load_trainer_profile()
    profile_section = f"\n\n{trainer_profile}" if trainer_profile else ""

    # ── Makroplan context ─────────────────────────────────────────────────────
    current_block = _get_current_makro_block()
    if current_block and current_block['phase'] != 'Pre':
        block_str = (
            f"{current_block['block_name']} ({current_block['phase']}) – "
            f"Woche {current_block['week_num']}/{current_block['total_weeks']} – "
            f"TSS-Ziel {current_block['tss_target']:.0f}/Woche"
        )
    elif current_block:
        block_str = f"Vor dem Plan – Plan startet {current_block['start_date']}"
    else:
        block_str = 'Kein Makroplan verfügbar'
    makro_summary = _makroplan_summary_str()

    # ── Optionales Nutzer-Feedback zu einer vorherigen Empfehlung ─────────────
    feedback_section = ''
    if feedback:
        prev_str = json.dumps(previous, ensure_ascii=False) if previous else '—'
        feedback_section = (
            "\n\n**MEINE RÜCKMELDUNG ZUR LETZTEN EMPFEHLUNG**\n"
            f"Vorherige Empfehlung: {prev_str}\n"
            f"Mein Feedback: {feedback}\n"
            "Passe die Empfehlung gemäß meinem Feedback an und erkläre kurz die Änderung."
        )

    prompt = f"""Du bist mein persönlicher Trainer.{profile_section}

**AKTUELLER TRAININGSBLOCK**
{block_str}

**MAKROPLAN ÜBERSICHT**
{makro_summary}

**AKTUELLE LAGE – {today_str}**
- CTL (Fitness): {ctl:.0f} | ATL (Müdigkeit): {atl:.0f} | TSB (Form): {tsb:+.0f} | ATL/CTL-Ratio: {ratio:.2f}
- Recovery Index heute: {ri_str}
- RI-Subwerte (0–100): {sub_str}
- LTHR: {LTHR} bpm | Max-HF: {HR_MAX} bpm | FTP: {FTP} W

**AKTIVE VERLETZUNGEN**
{injuries_str}

**LETZTE 7 EINHEITEN**
{workouts_str}{feedback_section}

Empfiehl mir eine konkrete Trainingseinheit für heute, die zum aktuellen Trainingsblock passt und auf meiner Ermüdung, Erholung und Trainingshistorie basiert. Berücksichtige aktive Verletzungen unbedingt: meide belastende Bewegungen für den betroffenen Körperbereich, schlage bei hohem Schweregrad (4-5) eher Ruhe oder eine schonende Alternative (z.B. Rad/Schwimmen statt Laufen) vor. Antworte ausschließlich als gültiges JSON-Objekt, kein Text davor oder danach:
{{
  "workout": "Kurzer Name z.B. '45min Z2 Lauf' oder 'Ruhetag'",
  "type": "Run oder Ride oder Swim oder Rest",
  "zone": <1-5 als Zahl, oder 0 bei Rest>,
  "duration_min": <Gesamtminuten als Zahl, 0 bei Rest>,
  "formula": "Intervall-Formel im Format z.B. '5min@z1+3x10min@z4/2min@z1+5min@z1' bei strukturierten Workouts, sonst z.B. '45min@z2'. Erster Block Z1 = Warmup, letzter Block Z1 = Cooldown. Bei Wiederholungen (NxDauer@zZone) IMMER eine Pause danach angeben im Format /Dauer@zZone (z.B. '/2min@z1' oder '/90sek@z1') – Intervalle dürfen nie ohne Pause aneinandergereiht werden. Leer bei Ruhetag.",
  "pace_hint": "Richtwert z.B. '5:45 /km' oder 'Ø 180 W' oder '—'",
  "reason": "Ein präziser Satz Begründung warum diese Empfehlung jetzt physiologisch sinnvoll ist und zum aktuellen Block passt.",
  "intention": "Kurze, prägnante Stichpunkte (2-4 Worte) für die Zielsetzung dieser Einheit, z.B. 'Locker laufen, aerobe Grundlage' oder 'Harte Intervalle, anaerobe Schwelle' oder 'Regeneration, lockeres Ausschütteln'. Kein ganzer Satz, keine Begründung."
}}"""

    try:
        response = _groq.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model='llama-3.3-70b-versatile',
            max_tokens=280,
            temperature=0.45,
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON even if wrapped in ```json ... ```
        m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        rec = json.loads(m.group() if m else raw)
        return jsonify({'status': 'ok', 'recommendation': rec,
                        'context': {'ctl': ctl, 'atl': atl, 'tsb': tsb, 'ratio': ratio,
                                    'ri': ri_val, 'ri_str': ri_str}})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/training_explainer')
def api_training_explainer():
    """Streamt eine kurze Fließtext-Erklärung (via OpenRouter), warum die
    heutige Algorithmus-Empfehlung optimal ist und was eine Alternative wäre.
    Nutzt trainer_profile.md + Trainingsdaten der letzten 30 Tage als Kontext."""
    if not OPENROUTER_API_KEY:
        return jsonify({'status': 'error',
                        'message': 'OPENROUTER_API_KEY nicht gesetzt (Umgebungsvariable).'}), 503

    rec_resp = api_training_recommendation()
    rec_data = rec_resp.get_json() if hasattr(rec_resp, 'get_json') else None
    if not rec_data or rec_data.get('status') != 'ok':
        msg = (rec_data or {}).get('message', 'Empfehlung konnte nicht erzeugt werden.')
        return jsonify({'status': 'error', 'message': msg}), 503
    rec = rec_data['recommendation']

    trainer_profile = _load_trainer_profile()
    profile_section  = f"\n\n{trainer_profile}" if trainer_profile else ""

    df = load_activities()
    last30_str = '  Keine Einträge vorhanden.'
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        cutoff = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=30)
        last30 = df[df['date'] >= cutoff].sort_values('date').copy()
        if not last30.empty:
            last30['tss'] = last30.apply(calc_tss, axis=1)
            lines = []
            for _, w in last30.iterrows():
                mt    = int(w.get('moving_time', 0) or 0)
                dist  = float(w.get('distance_km', 0) or 0)
                hr    = int(w.get('avg_hr', 0) or 0)
                tss_w = int(w.get('tss', 0) or 0)
                wdate = str(w['date'])[:10]
                lines.append(
                    f"  {wdate} | {str(w.get('type', '?'))[:12]:12s} | "
                    f"{dist:5.1f}km | {mt // 60:3d}min | HF {hr or '—':>3} bpm | TSS {tss_w}"
                )
            last30_str = '\n'.join(lines)

    user_prompt = f"""Du bist mein persönlicher Ausdauer-Trainer.{profile_section}

**MEINE TRAININGSDATEN DER LETZTEN 30 TAGE**
{last30_str}

**HEUTIGE EMPFEHLUNG DES ALGORITHMUS**
Workout: {rec.get('workout', '—')} ({rec.get('type', '—')}, Zone {rec.get('zone', '—')}, {rec.get('duration_min', '—')} min)
Formel: {rec.get('formula') or '—'}
Pace/Leistung: {rec.get('pace_hint') or '—'}
Begründung des Algorithmus: {rec.get('reason') or '—'}

Erkläre mir in einem kurzen Absatz im Fließtext, warum dieses Workout genau jetzt für mich optimal ist, und nenne danach eine sinnvolle Alternative dazu. Keine Aufzählungen, keine Überschriften, nur Fließtext."""

    def generate():
        yield '__REC__' + json.dumps(
            {'rec': rec, 'context': rec_data.get('context', {})}, ensure_ascii=False
        ) + '\n'
        try:
            or_resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-oss-20b:free",
                    "stream": True,
                    "messages": [
                        {"role": "system",
                         "content": "Du bist ein wortkarger Sportwissenschaftler und sprichst nur in "
                                    "ganzen Sätzen, in einem kurzen Fließtext-Absatz auf Deutsch."},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                stream=True,
                timeout=60,
            )
            for line in or_resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode('utf-8')
                if not decoded.startswith('data: '):
                    continue
                data = decoded[6:]
                if data == '[DONE]':
                    break
                try:
                    chunk   = json.loads(data)
                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                    if content:
                        yield content
                except Exception:
                    continue
        except Exception as e:
            yield f"\n\n(Fehler beim Abrufen der Erklärung: {e})"

    return Response(stream_with_context(generate()), mimetype='text/plain; charset=utf-8')


# ═════════════════════════════════════════════════════════════════════════════
# Trainings-Zustandsmodell – 3-System-Fitness (aerob / Schwelle / VO2)
# Liefert: System-Scores (0–100), Soll-/Ist-Verteilung, Zielsystem je Phase,
# Composite-Fitness, Momentum, Race-Readiness, Injury-Risk, Confidence.
# ═════════════════════════════════════════════════════════════════════════════

def _confidence_inputs(df, today):
    """(motivation, consistency, recent_success) je 0–100 für confidence()."""
    # Motivation: jüngster Morning-Checkin (Slider 1–5 → 0–100)
    motivation = 60.0
    try:
        if os.path.exists(MORNING_PATH):
            mc = pd.read_csv(MORNING_PATH)
            if not mc.empty:
                last = mc.sort_values('date').iloc[-1]
                rd = float(last.get('readiness') or 3)
                mcl = float(last.get('mental_clarity') or 3)
                motivation = max(0.0, min(100.0, (rd + mcl) / 2 * 20))
    except Exception:
        pass
    # Konstanz: Trainingstage der letzten 14 Tage
    try:
        cutoff = today - pd.Timedelta(days=14)
        days = df[(df['date'] >= cutoff) & (df['date'] <= today)]['date'].dt.normalize().nunique()
        consistency = max(0.0, min(100.0, days / 14 * 100))
    except Exception:
        consistency = 50.0
    # Jüngste Erfolge: PB in den letzten 30 Tagen?
    recent_success = 50.0
    try:
        be = _load_best_efforts()
        if not be.empty:
            be = be.copy()
            be['d'] = pd.to_datetime(be['date'], errors='coerce')
            ispb = be['is_pb'].astype(str).str.lower().isin(('true', '1'))
            recent = be[ispb & (be['d'] >= today - pd.Timedelta(days=30))]
            recent_success = 100.0 if not recent.empty else 50.0
    except Exception:
        pass
    return motivation, consistency, recent_success


def _injury_pain_score():
    """Aktive Verletzungen → 0–100 (Severity 1–5 ×20, höchste zählt)."""
    try:
        inj = _load_injuries()
        if inj.empty:
            return 0.0
        active = inj[inj['status'].astype(str).str.lower() != 'resolved']
        if active.empty:
            return 0.0
        sev = pd.to_numeric(active['severity'], errors='coerce').max()
        return float(max(0.0, min(100.0, (sev or 0) * 20)))
    except Exception:
        return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Readiness R – Tageszustand aus Check-in + HRV/RHR
#
#   R = 0,35·Ẽ + 0,25·M̃ − 0,20·Q̃ + 0,15·z_HRV − 0,05·z_RHR
#
#   Ẽ ← readiness        (Energie)         c̃ = (c−3)/2
#   M̃ ← mental_clarity   (Motivation)
#   Q̃ ← 6 − sleep_quality (schlechter Schlaf als Belastungssignal)
#   z  = (x − μ_60)/σ_60 ; HRV vorher ln des rollenden 7-Tage-Mittels
#
# Formeln in backend/readiness.py; hier nur das Zusammensuchen der Reihen.
# ═════════════════════════════════════════════════════════════════════════════

CHECKIN_MAX_AGE_DAYS = 3    # ältere Check-ins gelten als nicht mehr aussagekräftig


def _wellness_z_series() -> pd.DataFrame:
    """Tagesreihe mit z_hrv und z_rhr aus GarminConnectData_Koerperdaten.csv.

    HRV bevorzugt die Rohwerte der Nacht (hrv_last_night) und glättet sie
    selbst über 7 Tage; liefert die Uhr nur Garmins eigenen Wochenschnitt
    (hrv_weekly_avg), ist die Glättung bereits enthalten und wird übersprungen,
    damit nicht zweimal gemittelt wird.
    """
    empty = pd.DataFrame(columns=['date', 'hrv', 'rhr', 'z_hrv', 'z_rhr'])
    if not os.path.exists(GARMIN_BODY_PATH):
        return empty
    try:
        k = pd.read_csv(GARMIN_BODY_PATH, parse_dates=['date'])
    except Exception:
        return empty
    if k.empty:
        return empty
    k = k.sort_values('date')

    raw = pd.to_numeric(k.get('hrv_last_night'), errors='coerce') \
        if 'hrv_last_night' in k.columns else pd.Series(dtype='float64')
    if raw.notna().sum() >= 10:
        hrv, smooth, hrv_src = raw, rd.HRV_SMOOTH_DAYS, 'hrv_last_night'
    else:
        hrv = pd.to_numeric(k.get('hrv_weekly_avg'), errors='coerce') \
            if 'hrv_weekly_avg' in k.columns else pd.Series(0.0, index=k.index)
        smooth, hrv_src = 1, 'hrv_weekly_avg'      # bereits von Garmin gemittelt

    rhr = pd.to_numeric(k.get('resting_hr'), errors='coerce') \
        if 'resting_hr' in k.columns else pd.Series(dtype='float64')

    out = pd.DataFrame({
        'date':  k['date'].dt.normalize(),
        'hrv':   hrv.values if len(hrv) == len(k) else float('nan'),
        'rhr':   rhr.values if len(rhr) == len(k) else float('nan'),
    })
    # Lücken vorwärts füllen: ein fehlender Nachtwert soll die z-Skala nicht
    # verschieben, sondern den letzten bekannten Zustand fortschreiben.
    out['hrv'] = out['hrv'].ffill()
    out['rhr'] = out['rhr'].ffill()
    out['z_hrv'] = rd.hrv_z(out['hrv'], smooth=smooth).values
    out['z_rhr'] = rd.rolling_z(out['rhr']).values
    out.attrs['hrv_source'] = hrv_src
    return out


def _latest_checkin(today: pd.Timestamp) -> tuple:
    """Jüngster Morning-Checkin und sein Alter in Tagen (None, None wenn keiner)."""
    try:
        mc = _load_morning()
        if mc.empty:
            return None, None
        mc = mc.copy()
        mc['d'] = pd.to_datetime(mc['date'], errors='coerce')
        mc = mc.dropna(subset=['d']).sort_values('d')
        mc = mc[mc['d'] <= today]
        if mc.empty:
            return None, None
        last = mc.iloc[-1]
        return last, int((today - last['d'].normalize()).days)
    except Exception:
        return None, None


def compute_readiness(today=None) -> dict:
    """Readiness R für heute inkl. aller Einzelterme und Datenherkunft."""
    today = pd.Timestamp(today or datetime.now().date()).normalize()

    z = _wellness_z_series()
    z_hrv = z_rhr = 0.0
    hrv_raw = rhr_raw = None
    z_date  = None
    if not z.empty:
        past = z[z['date'] <= today]
        if not past.empty:
            row   = past.iloc[-1]
            z_hrv = float(row['z_hrv'])
            z_rhr = float(row['z_rhr'])
            hrv_raw = None if pd.isna(row['hrv']) else float(row['hrv'])
            rhr_raw = None if pd.isna(row['rhr']) else float(row['rhr'])
            z_date  = str(row['date'])[:10]

    last, age = _latest_checkin(today)
    stale = last is None or age is None or age > CHECKIN_MAX_AGE_DAYS
    if stale:
        # Kein frischer Check-in → Items neutral (3/5), R trägt dann nur die
        # objektiven HRV/RHR-Anteile. Ehrlicher als einen alten Tag fortzuschreiben.
        energy = motivation = sleep_q = None
    else:
        energy     = last.get('readiness')
        motivation = last.get('mental_clarity')
        sleep_q    = last.get('sleep_quality')

    res = rd.readiness(energy=energy, motivation=motivation,
                       sleep_quality=sleep_q, z_hrv=z_hrv, z_rhr=z_rhr)
    res.update({
        'checkin': {
            'date':          None if last is None else str(last.get('date'))[:10],
            'age_days':      age,
            'stale':         bool(stale),
            'energy':        _num_or_none(energy),
            'motivation':    _num_or_none(motivation),
            'sleep_quality': _num_or_none(sleep_q),
        },
        'wellness': {
            'date':       z_date,
            'hrv':        hrv_raw,
            'rhr':        rhr_raw,
            'hrv_source': z.attrs.get('hrv_source') if not z.empty else None,
            'z_hrv':      round(z_hrv, 3),
            'z_rhr':      round(z_rhr, 3),
        },
        'duration_factor': rd.duration_factor(res['R']),
    })
    return res


def _num_or_none(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _week_flags(df, today: pd.Timestamp) -> dict:
    """Schlafwoche / Kraftwoche der letzten 7 Tage – steuert r_aer und r_mech.

    Schlafwoche: Ø sleep_quality der Check-ins ≥ 4/5, ersatzweise Ø Garmin
    sleep_score ≥ 80 (dieselbe Schwelle, nur auf der 0–100-Skala).
    Kraftwoche: mindestens eine Kraft-/Boulder-Einheit in den letzten 7 Tagen.
    """
    cutoff = today - pd.Timedelta(days=7)

    sleep_week = False
    try:
        mc = _load_morning()
        if not mc.empty:
            mc = mc.copy()
            mc['d'] = pd.to_datetime(mc['date'], errors='coerce')
            recent = mc[(mc['d'] > cutoff) & (mc['d'] <= today)]
            vals = pd.to_numeric(recent.get('sleep_quality'), errors='coerce').dropna()
            if not vals.empty:
                sleep_week = bool(vals.mean() >= rd.SLEEP_WEEK_MIN)
    except Exception:
        pass
    if not sleep_week and os.path.exists(GARMIN_BODY_PATH):
        try:
            k = pd.read_csv(GARMIN_BODY_PATH, parse_dates=['date'])
            recent = k[(k['date'] > cutoff) & (k['date'] <= today)]
            sc = pd.to_numeric(recent.get('sleep_score'), errors='coerce').dropna()
            if not sc.empty:
                sleep_week = bool(sc.mean() >= rd.SLEEP_WEEK_MIN * 20)
        except Exception:
            pass

    strength_week = False
    try:
        if not df.empty:
            recent = df[(df['date'] > cutoff) & (df['date'] <= today)]
            strength_week = bool(recent['type'].apply(is_strength_type).any())
    except Exception:
        pass

    return rd.growth_rates(sleep_week, strength_week)


def _active_injuries() -> list:
    """Aktive Verletzungen als schlanke Dicts für die Kaskade."""
    try:
        inj = _load_injuries()
        if inj.empty:
            return []
        active = inj[inj['status'].astype(str).str.lower() != 'resolved']
        return [{'body_part': str(i.get('body_part', '')),
                 'title':     str(i.get('title', '')),
                 'severity':  _num_or_none(i.get('severity')) or 0}
                for _, i in active.iterrows()]
    except Exception:
        return []


@app.route('/api/readiness')
def api_readiness():
    """Readiness R für heute – Einzelterme, Datenherkunft, Dauerfaktor f."""
    try:
        return jsonify({'available': True, **compute_readiness()})
    except Exception as e:
        return jsonify({'available': False, 'reason': str(e)}), 500


@app.route('/api/unit_load/<int:activity_id>')
def api_unit_load(activity_id):
    """Rechner: L_aer und L_mech einer Einheit, aufgeschlüsselt nach Runden.

    Zeigt je Runde IF, φ (Tempo- und Steigungsanteil), Schritte und Beitrag –
    plus `jensen_gain`, also wie viel Last die Rundenauflösung gegenüber einer
    Rechnung auf der Ø-Intensität sichtbar macht.
    """
    try:
        df = load_activities()
        row = df[df['id'] == activity_id]
        if row.empty:
            return jsonify({'available': False, 'reason': 'Aktivität nicht gefunden'}), 404
        r   = row.iloc[0]
        res = calc_unit_load(r)
        return jsonify({
            'available': True,
            'activity': {
                'id':   int(activity_id),
                'name': '' if pd.isna(r.get('name')) else str(r['name']),
                'type': str(r.get('type', '')),
                'date': str(r['date'])[:10],
                'duration_min': round(float(r.get('moving_time') or 0) / 60, 1),
                'distance_km':  round(float(r.get('distance_km') or 0), 2),
            },
            'l_aer':  res['l_aer'],
            'l_mech': res['l_mech'],
            'n_laps': res['n_laps'],
            'laps_source': 'garmin' if res['n_laps'] > 1 else 'aktivität',
            'aer':    res['aer'],
            'mech':   res['mech'],
            'params': {
                'lthr': LTHR, 'ftp': FTP, 'run_ftp': RUN_FTP,
                'v_ref': ul.V_REF, 'phi_min': ul.PHI_MIN, 'beta': ul.BETA,
                'kappa_desc': ul.KAPPA_DESC, 'kappa_asc': ul.KAPPA_ASC,
                'lambda_road': ul.LAMBDA_ROAD, 'lambda_mtb': ul.LAMBDA_MTB,
            },
        })
    except Exception as e:
        return jsonify({'available': False, 'reason': str(e)}), 500


@app.route('/api/training_cascade', methods=['GET', 'POST'])
def api_training_cascade():
    """Kaskade: passt eine geplante Einheit an den Tageszustand an.

    Body (optional): {"type": "Run", "zone": 4, "duration_min": 60,
                      "speed_kmh": 11, "ascent_m": 0, "descent_m": 0}
    Ohne Body wird eine neutrale Einheit (60 min Z2 Lauf) durchgerechnet –
    dann zeigt die Antwort, WAS der Zustand heute überhaupt zulässt.
    """
    try:
        payload = request.get_json(silent=True) or {}
        planned = {
            'type':         payload.get('type', 'Run'),
            'zone':         payload.get('zone', 2),
            'duration_min': payload.get('duration_min', 60),
            'workout':      payload.get('workout', ''),
        }

        df    = load_activities()
        today = pd.Timestamp(datetime.now().date())
        load  = compute_load(df)
        trow  = load[load['date'] == today]
        ctl = float(trow['ctl'].iloc[-1]) if not trow.empty else 0.0
        atl = float(trow['atl'].iloc[-1]) if not trow.empty else 0.0
        tsb = float(trow['tsb'].iloc[-1]) if not trow.empty else 0.0
        ratio = round(atl / ctl, 2) if ctl > 0 else 0.0

        mech_week      = float(load['mech_week'].iloc[-1])
        mech_week_prev = float(load['mech_week'].iloc[-8]) if len(load) >= 8 else 0.0

        readiness_data = compute_readiness(today)
        rates          = _week_flags(df, today)
        forecast = rd.forecast_mech(
            planned['duration_min'], planned['type'],
            speed_kmh=payload.get('speed_kmh'),
            ascent_m=float(payload.get('ascent_m') or 0),
            descent_m=float(payload.get('descent_m') or 0),
        )

        ci = readiness_data['checkin']
        result = rd.cascade(
            planned=planned,
            R=readiness_data['R'],
            ctl=ctl, tsb=tsb,
            mech_week=mech_week, mech_week_prev=mech_week_prev,
            mech_forecast=forecast,
            load_ratio=ratio,
            z_hrv=readiness_data['wellness']['z_hrv'],
            motivation=ci['motivation'], sleep_quality=ci['sleep_quality'],
            injuries=_active_injuries(),
            rates=rates,
        )
        result.update({
            'available': True,
            'readiness': readiness_data,
            'load': {'ctl': round(ctl, 1), 'atl': round(atl, 1),
                     'tsb': round(tsb, 1), 'ratio': ratio},
        })
        return jsonify(result)
    except Exception as e:
        return jsonify({'available': False, 'reason': str(e)}), 500


@app.route('/api/training_state')
def api_training_state():
    """3-System-Fitnessmodell für heute (deterministisch)."""
    try:
        df = load_activities()
        if df.empty:
            return jsonify({'available': False, 'reason': 'Keine Aktivitäten'})

        df = df.copy()
        df['tss']  = df.apply(calc_tss, axis=1)
        df['date'] = pd.to_datetime(df['date'])
        today      = pd.Timestamp(datetime.now().date())

        hr_lt1 = ZONE_BPM[2][1]   # Oberkante Z2 ≈ aerobe Schwelle (LT1)
        hr_lt2 = LTHR             # Laktatschwelle (LT2)

        # ── System-Scores + Verteilung + Zielsystem ──────────────────────────
        df['is_strength'] = df['type'].apply(is_strength_type)
        acts = df[['date', 'avg_hr', 'tss', 'is_strength']]
        ts        = tm.build_system_timeseries(acts, hr_lt1, hr_lt2, today=today)
        scores    = tm.normalize_scores(ts)
        scores_prev = tm.normalize_scores(ts, offset=42)   # vor 6 Wochen (Geist-Polygon)
        dist      = tm.recent_distribution(df, hr_lt1, hr_lt2, days=14, today=today)
        composite = tm.composite_fitness(scores)
        mom       = tm.momentum(ts)

        # ── Absolute TSS pro System (14d) für Garmin-Style-Balken ────────────
        cutoff14 = today - pd.Timedelta(days=14)
        dist_tss  = {s: 0.0 for s in tm.SYSTEMS}
        for _, r in acts[acts['date'] >= cutoff14].iterrows():
            parts = tm.split_tss(r.get('avg_hr'), r.get('tss'), hr_lt1, hr_lt2,
                                 bool(r.get('is_strength', False)))
            for s in tm.SYSTEMS:
                dist_tss[s] += parts[s]
        total_tss_14d = sum(dist_tss.values())

        block = _get_current_makro_block() or {}
        phase = block.get('phase', 'Base')
        if phase in ('Pre', ''):
            phase = 'Base'
        emphasis = tm.phase_emphasis(phase)
        target, ranking = tm.choose_target_system(scores, dist, phase)

        # ── CTL/ATL/TSB + abgeleitete Belastungsgrößen ───────────────────────
        load_df = compute_load(df)
        trow    = load_df[load_df['date'].astype(str).str[:10] == today.strftime('%Y-%m-%d')]
        ctl = float(trow['ctl'].iloc[-1]) if not trow.empty else 0.0
        atl = float(trow['atl'].iloc[-1]) if not trow.empty else 0.0
        tsb = float(trow['tsb'].iloc[-1]) if not trow.empty else 0.0

        atl_peak90 = float(load_df['atl'].tail(90).max()) or 1.0
        fatigue    = max(0.0, min(100.0, 100.0 * atl / atl_peak90))
        last7_tss  = load_df['tss'].tail(7)
        monotony   = max(0.0, min(100.0, (last7_tss.mean() / (last7_tss.std() + 1e-6)) * 20))
        volume_jump = max(0.0, min(100.0, (atl / ctl - 1) * 100)) if ctl > 0 else 0.0

        # ── Recovery Index heute (oder jüngster Tag) ─────────────────────────
        recovery = 50.0
        ri_label = '—'
        grow, past_df, _ = _get_recent_garmin_row()
        if grow:
            atl_today, atl_7d = _atl_for_date(today.strftime('%Y-%m-%d'))
            ri = compute_recovery_index(grow, past_df, atl_today, atl_7d)
            recovery = ri['recovery_index']
            ri_label = _ri_label(recovery)

        # ── Confidence / Injury-Risk / Race-Readiness ────────────────────────
        pain = _injury_pain_score()
        motivation, consistency, recent_success = _confidence_inputs(df, today)
        inj_risk   = tm.injury_risk(fatigue, monotony, volume_jump, pain)
        conf       = tm.confidence(motivation, consistency, recent_success)
        readiness  = tm.race_readiness(composite, recovery, conf, fatigue, inj_risk)

        # ── Quality-Day-Gate: ob heute eine harte Einheit sinnvoll ist ───────
        week_start = today - pd.Timedelta(days=today.weekday())
        week_tss   = float(df[df['date'] >= week_start]['tss'].sum())
        tss_target = float(block.get('tss_target', 0) or 0)
        quota_left = (tss_target - week_tss) if tss_target > 0 else None
        quality_day = (recovery >= 50) and (tsb > -25) and \
                      (quota_left is None or quota_left > tss_target * 0.10)

        SYS_LABEL = {'aerobic': 'Aerobe Basis', 'threshold': 'Schwelle',
                     'vo2': 'VO2max', 'strength': 'Kraft'}

        # ── Mechanisches Wochenbudget + Readiness R ──────────────────────────
        mech_week      = float(load_df['mech_week'].iloc[-1])
        mech_week_prev = float(load_df['mech_week'].iloc[-8]) if len(load_df) >= 8 else 0.0
        rates          = _week_flags(df, today)
        mech_budget    = mech_week_prev * (1 + rates['r_mech'])
        readiness_data = compute_readiness(today)

        return jsonify({
            'available': True,
            'phase':       phase,
            'block_name':  block.get('block_name', ''),
            'systems':     scores,
            'systems_prev': scores_prev,
            'system_labels': SYS_LABEL,
            'emphasis':    emphasis,       # Soll-Lastanteil
            'distribution': dist,          # Ist-Lastanteil (14 Tage)
            'distribution_tss': {s: round(v, 0) for s, v in dist_tss.items()},
            'total_tss_14d': round(total_tss_14d, 0),
            'target_system': target,
            'target_label':  SYS_LABEL.get(target, '—'),
            'ranking':     ranking,
            'composite_fitness': composite,
            'momentum':    mom,
            'race_readiness': readiness,
            'injury_risk': inj_risk,
            'confidence':  conf,
            'quality_day': bool(quality_day),
            'recovery_index': recovery,
            'recovery_label': ri_label,
            'hr_lt1': int(hr_lt1),
            'hr_lt2': int(hr_lt2),
            'load': {'ctl': round(ctl, 1), 'atl': round(atl, 1), 'tsb': round(tsb, 1),
                     'week_tss': round(week_tss, 0), 'tss_target': round(tss_target, 0),
                     'tsb_limit': round(rd.TSB_CTL_FACTOR * ctl, 1)},
            'mech': {'week': round(mech_week, 2),
                     'week_prev': round(mech_week_prev, 2),
                     'budget': round(mech_budget, 2),
                     'headroom': round(mech_budget - mech_week, 2)},
            'rates': rates,
            'readiness': readiness_data,
        })
    except Exception as e:
        return jsonify({'available': False, 'reason': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# Lastmodell-Gates als Wochenreihe – Zielkorridor, Mech-Decke, TSB-Grenze
#
# Liefert dem Dashboard die drei Sichten auf dieselben Kaskaden-Regeln:
#   • Zielkorridor (Regel 2 + Steigerungsraten): kommende Wochen mit dem Band
#     [Vorwoche, Vorwoche·(1+r)] je Kanal und der geplanten Last L̂ darin.
#   • Mech-Decke: Wochen-L_mech gegen Mech-Last_{k−1}·(1+r_mech) – wo die Woche
#     die Decke durchstößt, hätte Regel 2 Lauf → Rad ausgelöst.
#   • Form-Grenze: TSB gegen −0,35·CTL (Regel 3) plus ATL/CTL (Regel 4).
# ═════════════════════════════════════════════════════════════════════════════

GATES_PAST_WEEKS   = 10
GATES_FUTURE_MAX   = 4


def _gates_week_row(load, wk_start, wk_end):
    """Zustandsgrößen am Ende einer Woche (spätestens heute)."""
    w = load[(load['date'] >= wk_start) & (load['date'] <= wk_end)]
    if w.empty:
        return None
    return w.iloc[-1]


def _planned_week_estimate(planned_df, wk_start, wk_end, speed_kmh):
    """L̂_aer / L̂_mech aller geplanten Einheiten einer Woche."""
    aer = mech = 0.0
    n = 0
    for _, row in planned_df.iterrows():
        d = str(row.get('date', ''))[:10]
        if len(d) != 10:
            continue
        try:
            ts = pd.Timestamp(d)
        except Exception:
            continue
        if ts < wk_start or ts > wk_end:
            continue
        tss, dur_s, _blocks = _estimate_planned_session(row)
        sport = str(row.get('type', '') or 'run')
        aer  += float(tss or 0)
        mech += rd.forecast_mech(dur_s / 60.0, sport, speed_kmh=speed_kmh)
        n += 1
    return round(aer, 1), round(mech, 1), n


@app.route('/api/load_gates')
def api_load_gates():
    """Wochenreihe der Gates: 10 Wochen rückwärts, bis zu 4 Wochen Ausblick."""
    try:
        try:
            n_future = int(request.args.get('future', GATES_FUTURE_MAX))
        except (TypeError, ValueError):
            n_future = GATES_FUTURE_MAX
        n_future = max(0, min(GATES_FUTURE_MAX, n_future))

        df = load_activities()
        if df.empty:
            return jsonify({'available': False, 'reason': 'Keine Aktivitäten'})
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        load = compute_load(df)

        today       = pd.Timestamp(datetime.now().date())
        this_monday = today - pd.Timedelta(days=today.weekday())
        rates_now   = _week_flags(df, today)

        # ── Vergangenheit inkl. laufender Woche ──────────────────────────────
        weeks = []
        for i in range(GATES_PAST_WEEKS - 1, -1, -1):
            wk_start = this_monday - pd.Timedelta(weeks=i)
            wk_end   = wk_start + pd.Timedelta(days=6)
            w        = load[(load['date'] >= wk_start) & (load['date'] <= wk_end)]
            row      = _gates_week_row(load, wk_start, wk_end)
            prev_row = _gates_week_row(load, wk_start - pd.Timedelta(weeks=1),
                                       wk_start - pd.Timedelta(days=1))
            l_aer  = float(w['tss'].sum())    if len(w) else 0.0
            l_mech = float(w['l_mech'].sum()) if len(w) else 0.0
            mech_avg  = float(row['mech_week'])      if row is not None      else 0.0
            mech_prev = float(prev_row['mech_week']) if prev_row is not None else 0.0
            # Steigerungsrate, wie sie in dieser Woche galt (Kraft-/Schlafwoche davor)
            r_week  = _week_flags(df, min(wk_end, today))
            ceiling = mech_prev * (1.0 + r_week['r_mech'])
            over    = ceiling > 0 and l_mech > ceiling
            ctl = float(row['ctl']) if row is not None else 0.0
            atl = float(row['atl']) if row is not None else 0.0
            tsb = float(row['tsb']) if row is not None else 0.0
            tsb_limit = rd.TSB_CTL_FACTOR * ctl
            weeks.append({
                'label':      f"KW {wk_end.isocalendar()[1]}",
                'start':      wk_start.strftime('%Y-%m-%d'),
                'end':        wk_end.strftime('%Y-%m-%d'),
                'is_current': bool(i == 0),
                'l_aer':      round(l_aer, 1),
                'l_mech':     round(l_mech, 1),
                'mech_avg':   round(mech_avg, 1),
                'mech_prev':  round(mech_prev, 1),
                'r_mech':     r_week['r_mech'],
                'ceiling':    round(ceiling, 1),
                'over':       bool(over),
                'over_pct':   round((l_mech / ceiling - 1.0) * 100.0) if over else 0,
                'rest_budget': round(ceiling - mech_avg, 1),
                'ctl':        round(ctl, 1),
                'atl':        round(atl, 1),
                'tsb':        round(tsb, 1),
                'tsb_limit':  round(tsb_limit, 1),
                'headroom':   round(tsb - tsb_limit, 1),
                'ratio':      round(atl / ctl, 2) if ctl > 0 else None,
            })

        # ── Ausblick: Korridor je Kanal, verkettet über die geplante Last ────
        planned_df = _load_planned()
        makro      = _load_makroplan()
        makro_rows = []
        for _, m in makro.iterrows():
            try:
                makro_rows.append((str(m.get('start_date', '')), str(m.get('end_date', '')),
                                   float(m.get('tss_target', 0) or 0)))
            except (TypeError, ValueError):
                continue

        # Mech-Anteil der letzten 4 Wochen: übersetzt ein reines L_aer-Ziel
        # (Makroplan) in eine grobe L̂_mech-Erwartung.
        recent4 = weeks[-4:]
        sum_aer  = sum(w['l_aer']  for w in recent4)
        sum_mech = sum(w['l_mech'] for w in recent4)
        mech_ratio = (sum_mech / sum_aer) if sum_aer > 0 else 0.0

        # Typisches Lauftempo der letzten 28 Tage für die Mech-Prognose
        run_mask = df['type'].apply(ul.is_run) & (df['date'] >= today - pd.Timedelta(days=28))
        speeds   = pd.to_numeric(df.loc[run_mask, 'avg_speed_kmh'], errors='coerce').dropna()
        speeds   = speeds[speeds > 0]
        speed_kmh = float(speeds.mean()) if not speeds.empty else 10.0

        r_aer, r_mech = rates_now['r_aer'], rates_now['r_mech']
        base_aer  = weeks[-1]['l_aer']
        base_mech = weeks[-1]['l_mech']
        future = []
        for k in range(1, n_future + 1):
            wk_start = this_monday + pd.Timedelta(weeks=k)
            wk_end   = wk_start + pd.Timedelta(days=6)
            aer_lo, aer_hi   = base_aer,  base_aer  * (1.0 + r_aer)
            mech_lo, mech_hi = base_mech, base_mech * (1.0 + r_mech)

            p_aer, p_mech, n_sess = _planned_week_estimate(planned_df, wk_start, wk_end, speed_kmh)
            source = 'sessions' if n_sess > 0 else None
            if n_sess == 0:
                ws, we = wk_start.strftime('%Y-%m-%d'), wk_end.strftime('%Y-%m-%d')
                for ms, me, tss in makro_rows:
                    if ms <= we and me >= ws and tss > 0:
                        p_aer  = round(tss, 1)
                        p_mech = round(tss * mech_ratio, 1)
                        source = 'makroplan'
                        break

            def _status(val, lo, hi):
                if source is None or val <= 0:
                    return None
                if val > hi + 1e-9:
                    return 'gate2'
                if val < lo - 1e-9:
                    return 'erhaltung'
                return 'reiz'

            future.append({
                'label':      f"KW {wk_end.isocalendar()[1]}",
                'start':      wk_start.strftime('%Y-%m-%d'),
                'end':        wk_end.strftime('%Y-%m-%d'),
                'aer_lo':     round(aer_lo, 1),  'aer_hi':  round(aer_hi, 1),
                'mech_lo':    round(mech_lo, 1), 'mech_hi': round(mech_hi, 1),
                'planned_aer':  p_aer  if source else None,
                'planned_mech': p_mech if source else None,
                'planned_source': source,
                'planned_n':  n_sess,
                'status_aer':  _status(p_aer,  aer_lo,  aer_hi),
                'status_mech': _status(p_mech, mech_lo, mech_hi),
            })
            # Verkettung: die nächste Woche misst sich an dem, was für diese
            # Woche geplant ist – ohne Plan am oberen Rand des Korridors.
            base_aer  = p_aer  if (source and p_aer  > 0) else aer_hi
            base_mech = p_mech if (source and p_mech > 0) else mech_hi

        return jsonify({
            'available':  True,
            'today':      today.strftime('%Y-%m-%d'),
            'rates':      rates_now,
            'tsb_factor': rd.TSB_CTL_FACTOR,
            'acwr_band':  [0.8, rd.GATE_LOAD_RATIO],
            'mech_ratio': round(mech_ratio, 3),
            'weeks':      weeks,
            'future':     future,
        })
    except Exception as e:
        return jsonify({'available': False, 'reason': str(e)}), 500


QUALITY_ZONE4PLUS_MIN_S = 480   # mind. 8 Minuten in Zone 4/5 -> zählt als Quality Session


def _is_quality_activity(act_id, avg_hr):
    """True, wenn die Aktivität einen echten Schwelle/VO2-Reiz enthielt.

    Reine Ø-Herzfrequenz reicht bei strukturierten Intervall-Workouts nicht
    (Erholungsphasen drücken den Schnitt unter die Schwelle) – deshalb zuerst
    der schnelle Ø-HF-Check, sonst ein Blick in den (bereits lokal gecachten)
    Sekunden-Stream auf kumulierte Zone-4/5-Zeit. Kein Live-Garmin-Abruf hier,
    um die Karte schnell und ratenlimit-sicher zu halten – ungecachte, weiter
    zurückliegende Aktivitäten werden dann konservativ als "nicht quality"
    behandelt.
    """
    try:
        if pd.notna(avg_hr) and float(avg_hr) >= LTHR:
            return True
    except (TypeError, ValueError):
        pass
    cache_path = os.path.join(STREAMS_DIR, f'{int(act_id)}.json')
    if not os.path.exists(cache_path):
        return False
    try:
        with open(cache_path, encoding='utf-8') as f:
            stream = json.load(f)
    except Exception:
        return False
    secs = _zone_seconds_from_stream(stream)
    return (secs.get(4, 0) + secs.get(5, 0)) >= QUALITY_ZONE4PLUS_MIN_S


@app.route('/api/quality_session_nudge')
def api_quality_session_nudge():
    """Tage seit der letzten 'Quality Session' (Schwelle/VO2-Reiz statt
    lockerem Aeroblauf) – neueste Aktivität zuerst, erster Treffer zählt."""
    try:
        df = load_activities()
        if df.empty:
            return jsonify({'available': False})

        df = df.copy()
        df['date']   = pd.to_datetime(df['date'])
        df['avg_hr'] = pd.to_numeric(df['avg_hr'], errors='coerce')
        df = df[~df['type'].apply(is_strength_type)].sort_values('date', ascending=False)

        last = None
        for _, row in df.iterrows():
            act_id = row.get('id')
            if pd.isna(act_id):
                continue
            if _is_quality_activity(act_id, row.get('avg_hr')):
                last = row
                break

        if last is None:
            return jsonify({'available': False})

        last_date = last['date'].normalize()
        days      = (pd.Timestamp(datetime.now().date()) - last_date).days

        quality_day = False
        try:
            quality_day = bool(api_training_state().get_json().get('quality_day'))
        except Exception:
            pass

        return jsonify({
            'available':   True,
            'days':        int(days),
            'last_name':   '' if pd.isna(last.get('name')) else str(last['name']),
            'last_date':   last_date.strftime('%d.%m.'),
            'quality_day': quality_day,
        })
    except Exception as e:
        return jsonify({'available': False, 'reason': str(e)}), 500


@app.route('/api/garmin_workouts', methods=['GET'])
def api_garmin_workouts():
    """Return list of workouts from the Garmin Connect library."""
    try:
        api = _login_garmin()
        raw = api.get_workouts(0, 100)
        result = []
        for w in raw:
            wid   = w.get('workoutId') or (w.get('workoutDTO') or {}).get('workoutId')
            wname = w.get('workoutName') or (w.get('workoutDTO') or {}).get('workoutName', '')
            if wid and wname:
                result.append({'id': wid, 'name': wname})
        result.sort(key=lambda x: x['name'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/schedule_workout', methods=['POST'])
def api_schedule_workout():
    """Schedule an existing Garmin workout (by ID) on a given date."""
    try:
        data        = request.get_json() or {}
        workout_id  = data.get('workout_id')
        date_str    = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        intention   = (data.get('intention', '') or '').strip()
        api = _login_garmin()
        api.schedule_workout(workout_id, date_str)
        name = data.get('name', f'Workout {workout_id}')
        gcal_id = _gcal_add_safe(date_str, name, 60,
                                 f'🎯 {intention}' if intention else '')
        try:
            _add_planned_internal(date_str, name, 'run', '', str(workout_id),
                                  gcal_id=gcal_id, intention=intention)
        except Exception:
            pass
        return jsonify({'status': 'ok', 'workout_id': workout_id, 'date': date_str})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Garmin / Calendar workout helpers ─────────────────────────────────────────

def _login_garmin():
    from garminconnect import Garmin
    cfg = CONFIG.get('garmin', {})
    api = Garmin(cfg['username'], cfg['password'])
    api.login(GARMIN_TOKENSTORE)
    return api


def _wstep(order, type_id, type_key, dur_s, target_id, target_key, bpm_lo=None, bpm_hi=None):
    s = {
        'type':              'ExecutableStepDTO',
        'stepOrder':         order,
        'stepType':          {'stepTypeId': type_id, 'stepTypeKey': type_key},
        'endCondition':      {'conditionTypeId': 2, 'conditionTypeKey': 'time'},
        'endConditionValue': dur_s,
        'targetType':        {'workoutTargetTypeId': target_id, 'workoutTargetTypeKey': target_key},
    }
    if bpm_lo is not None:
        s['targetValueOne'] = bpm_lo
        s['targetValueTwo'] = bpm_hi
    return s


def _build_simple_workout(name, duration_s, zone):
    """Return (payload, description, est_seconds) for a warmup/interval/cooldown run."""
    hr_lo, hr_hi = ZONE_BPM.get(zone, ZONE_BPM[2])
    est_s = 300 + duration_s + 300
    description = f'Zone {zone} Lauf – {duration_s // 60} min ({hr_lo}–{hr_hi} bpm)'
    payload = {
        'workoutName':             name,
        'description':             description,
        'sportType':               {'sportTypeId': 1, 'sportTypeKey': 'running'},
        'estimatedDurationInSecs': est_s,
        'workoutSegments': [{
            'segmentOrder': 1,
            'sportType':    {'sportTypeId': 1, 'sportTypeKey': 'running'},
            'workoutSteps': [
                _wstep(1, 1, 'warmup',   300,        1, 'no.target'),
                _wstep(2, 3, 'interval', duration_s, 4, 'heart.rate.zone', bpm_lo=hr_lo, bpm_hi=hr_hi),
                _wstep(3, 2, 'cooldown', 300,        1, 'no.target'),
            ],
        }],
    }
    return payload, description, est_s


def _build_formula_workout(name, formula):
    """Return (payload, description, est_seconds) or None if the formula is invalid.

    Garmin-Stufentypen (aus der offiziellen garminconnect-Bibliothek, workout.py
    StepType): WARMUP=1, COOLDOWN=2, INTERVAL=3, RECOVERY=4, REST=5, REPEAT=6.
    Ein Z1-Block am Anfang der Formel ist Warmup, einer am Ende ist Cooldown
    (nicht nochmal Warmup!) – alles dazwischen bleibt ein normaler Interval-Schritt.
    Wiederholungs-Blöcke (reps > 1) bekommen zusätzlich einen Recovery-Schritt
    zwischen den Reps, statt sie ohne Pause aneinanderzureihen."""
    blocks = _parse_workout_formula(formula)
    if not blocks:
        return None
    last_i = len(blocks) - 1
    workout_steps = []
    desc_parts    = []
    for i, b in enumerate(blocks):
        lo, hi = ZONE_BPM.get(b['zone'], ZONE_BPM[2])
        dur_s  = b['duration_s']
        order  = i + 1
        if b['zone'] == 1 and i == 0:
            tid, tkey = 1, 'warmup'
        elif b['zone'] == 1 and i == last_i and last_i > 0:
            tid, tkey = 2, 'cooldown'
        else:
            tid, tkey = 3, 'interval'

        m_s, s_s  = divmod(dur_s, 60)
        dur_label = (f'{m_s}min' if s_s == 0 else
                     f'{dur_s}sek' if m_s == 0 else f'{m_s}min {s_s}sek')

        if b['reps'] == 1:
            workout_steps.append(_wstep(order, tid, tkey, dur_s, 4, 'heart.rate.zone', bpm_lo=lo, bpm_hi=hi))
            desc_parts.append(f'{dur_label} Z{b["zone"]} ({lo}–{hi} bpm)')
        else:
            rest_s    = b['rest_duration_s']
            rest_zone = b['rest_zone']
            r_lo, r_hi = ZONE_BPM.get(rest_zone, ZONE_BPM[1])
            r_m, r_s   = divmod(rest_s, 60)
            rest_label = f'{r_m}min' if r_s == 0 else (f'{rest_s}sek' if r_m == 0 else f'{r_m}min {r_s}sek')
            workout_steps.append({
                'type':               'RepeatGroupDTO',
                'stepOrder':          order,
                'stepType':           {'stepTypeId': 6, 'stepTypeKey': 'repeat'},
                'numberOfIterations': b['reps'],
                'workoutSteps': [
                    _wstep(1, tid, tkey, dur_s,  4, 'heart.rate.zone', bpm_lo=lo, bpm_hi=hi),
                    _wstep(2, 4, 'recovery', rest_s, 1, 'no.target'),
                ],
            })
            desc_parts.append(f'{b["reps"]}× {dur_label} Z{b["zone"]} ({lo}–{hi} bpm) + {rest_label} Pause (Z{rest_zone})')

    total_s = sum(
        b['reps'] * (b['duration_s'] + b.get('rest_duration_s', 0)) if b['reps'] > 1 else b['duration_s']
        for b in blocks
    )
    description = ' → '.join(desc_parts)
    payload = {
        'workoutName':             name,
        'description':             description,
        'sportType':               {'sportTypeId': 1, 'sportTypeKey': 'running'},
        'estimatedDurationInSecs': total_s,
        'workoutSegments': [{
            'segmentOrder': 1,
            'sportType':    {'sportTypeId': 1, 'sportTypeKey': 'running'},
            'workoutSteps': workout_steps,
        }],
    }
    return payload, description, total_s


def _apply_intention(payload, description, intention):
    """Prepend the workout intention to the Garmin description + calendar text.

    Returns the (possibly extended) description string and mutates payload so the
    intention shows up at the top of the workout in Garmin Connect."""
    intention = (intention or '').strip()
    if not intention:
        return description
    full = f'🎯 {intention}\n\n{description}' if description else f'🎯 {intention}'
    if isinstance(payload, dict):
        payload['description'] = full
    return full


def _upload_and_schedule(api, payload, name, date_str):
    """Remove any same-named workout, upload the new one and schedule it. Returns workout_id."""
    for w in api.get_workouts(0, 100):
        wname = w.get('workoutName') or (w.get('workoutDTO') or {}).get('workoutName', '')
        if wname == name:
            wid = w.get('workoutId') or (w.get('workoutDTO') or {}).get('workoutId')
            try:
                api.delete_workout(wid)
            except Exception:
                pass
    result     = api.upload_workout(payload)
    workout_id = result.get('workoutId') or (result.get('workoutDTO') or {}).get('workoutId')
    if workout_id:
        api.schedule_workout(workout_id, date_str)
    return workout_id


def _garmin_delete_safe(api, workout_id):
    if not workout_id or str(workout_id) in ('', 'nan', 'None'):
        return
    for wid in (workout_id, str(workout_id)):
        try:
            api.delete_workout(int(float(workout_id)))
            return
        except Exception:
            try:
                api.delete_workout(wid)
                return
            except Exception:
                continue


def _gcal_add_safe(date_str, name, duration_min, description):
    """Create a calendar event, returning its id ('' on any failure)."""
    try:
        sys.path.insert(0, BACKEND_DIR)
        from google_calendar import add_run_event
        ev = add_run_event(date_str, name, duration_min=duration_min, description=description)
        return ev.get('id', '') if isinstance(ev, dict) else ''
    except Exception:
        return ''


def _gcal_delete_safe(event_id):
    if not event_id or str(event_id) in ('', 'nan', 'None'):
        return
    try:
        sys.path.insert(0, BACKEND_DIR)
        from google_calendar import delete_event
        delete_event(str(event_id))
    except Exception:
        pass


@app.route('/api/push_workout', methods=['POST'])
def api_push_workout():
    """Create a running workout on Garmin Connect and schedule it for today."""
    try:
        data        = request.get_json() or {}
        duration_s  = int(data.get('duration_s', 1800))
        zone        = int(data.get('zone', 2))
        date_str    = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        name        = data.get('name', f'Dashboard {datetime.now().strftime("%d.%m.%Y")}')
        intention   = (data.get('intention', '') or '').strip()

        api = _login_garmin()
        payload, description, est_s = _build_simple_workout(name, duration_s, zone)
        description = _apply_intention(payload, description, intention)
        workout_id = _upload_and_schedule(api, payload, name, date_str)
        if not workout_id:
            return jsonify({'status': 'error', 'message': 'Keine workout_id erhalten'}), 500

        gcal_id = _gcal_add_safe(date_str, name, est_s // 60, description)
        try:
            _add_planned_internal(date_str, name, 'run', '', str(workout_id),
                                  gcal_id=gcal_id, duration_s=duration_s, zone=zone,
                                  intention=intention)
        except Exception:
            pass

        return jsonify({'status': 'ok', 'workout_id': workout_id, 'name': name, 'date': date_str})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# Verletzungstagebuch
# ═════════════════════════════════════════════════════════════════════════════

INJURY_COLS = ['id', 'date', 'body_part', 'title', 'description', 'severity', 'status', 'resolved_date']


def _load_injuries():
    if not os.path.exists(INJURIES_PATH):
        return pd.DataFrame(columns=INJURY_COLS)
    try:
        df = pd.read_csv(INJURIES_PATH, dtype={'id': int, 'severity': int})
        for col in INJURY_COLS:
            if col not in df.columns:
                df[col] = '' if col not in ('id', 'severity') else 0
        df['resolved_date'] = df['resolved_date'].astype(object).where(df['resolved_date'].notna(), '')
        return df
    except Exception:
        return pd.DataFrame(columns=INJURY_COLS)


def _save_injuries(df):
    df.to_csv(INJURIES_PATH, index=False)


NON_CARDIO_COLS = ['id', 'date', 'type', 'name', 'duration', 'rpe', 'comment', 'intention', 'intention_met']


def _load_non_cardio():
    if not os.path.exists(NON_CARDIO_PATH):
        return pd.DataFrame(columns=NON_CARDIO_COLS)
    try:
        df = pd.read_csv(NON_CARDIO_PATH, dtype={'id': int})
        for col in NON_CARDIO_COLS:
            if col not in df.columns:
                df[col] = ''
        return df
    except Exception:
        return pd.DataFrame(columns=NON_CARDIO_COLS)


@app.route('/api/non_cardio', methods=['GET'])
def api_get_non_cardio():
    df = _load_non_cardio()
    df = df.where(df.notna(), other='')
    records = df.sort_values('date', ascending=False).to_dict(orient='records') if not df.empty else []
    return jsonify(records)


@app.route('/api/non_cardio', methods=['POST'])
def api_add_non_cardio():
    data = request.get_json() or {}
    df   = _load_non_cardio()
    new_id = int(df['id'].max()) + 1 if not df.empty else 1
    new_row = {
        'id':       new_id,
        'date':     data.get('date', datetime.now().strftime('%Y-%m-%d')),
        'type':     data.get('type', ''),
        'name':     data.get('name', data.get('type', '')),
        'duration': data.get('duration', ''),
        'rpe':      data.get('rpe', ''),
        'comment':  '',
        'intention': '',
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(NON_CARDIO_PATH, index=False)
    return jsonify({'status': 'ok', 'entry': new_row})


def _patch_non_cardio_activity(entry_id: int, data: dict):
    """Gemeinsame Update-Logik für manuelle Kraft-/Boulder-Einträge – aufgerufen
    sowohl von PATCH /api/non_cardio/<id> als auch von PATCH /api/activities/<-id>
    (Detailansicht behandelt manuelle Einträge wie normale Aktivitäten)."""
    df   = _load_non_cardio()
    mask = df['id'] == entry_id
    if not mask.any():
        return jsonify({'status': 'error', 'message': 'Nicht gefunden'}), 404
    for field in ('date', 'type', 'name', 'duration', 'comment', 'intention', 'intention_met'):
        if field in data:
            df.loc[mask, field] = data[field]
    if 'rpe' in data:
        raw = str(data['rpe']).strip().replace(',', '.')
        try:
            val = '' if raw == '' else str(round(max(0.0, min(10.0, float(raw))), 1))
        except ValueError:
            val = ''
        df['rpe'] = df['rpe'].astype('object')
        df.loc[mask, 'rpe'] = val
    df.to_csv(NON_CARDIO_PATH, index=False)
    return jsonify({'status': 'ok'})


@app.route('/api/non_cardio/<int:entry_id>', methods=['PATCH'])
def api_update_non_cardio(entry_id):
    return _patch_non_cardio_activity(entry_id, request.get_json() or {})


@app.route('/api/injuries', methods=['GET'])
def api_get_injuries():
    df = _load_injuries()
    df = df.where(df.notna(), other='')
    records = df.sort_values('date', ascending=False).to_dict(orient='records') if not df.empty else []
    return jsonify(records)


@app.route('/api/injuries', methods=['POST'])
def api_add_injury():
    data = request.get_json() or {}
    df   = _load_injuries()
    new_id = int(df['id'].max()) + 1 if not df.empty else 1
    new_row = {
        'id':            new_id,
        'date':          data.get('date',        datetime.now().strftime('%Y-%m-%d')),
        'body_part':     data.get('body_part',   'Unbekannt'),
        'title':         data.get('title',       ''),
        'description':   data.get('description', ''),
        'severity':      int(data.get('severity', 3)),
        'status':        'active',
        'resolved_date': '',
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_injuries(df)
    return jsonify({'status': 'ok', 'entry': new_row})


@app.route('/api/injuries/<int:injury_id>', methods=['PATCH'])
def api_update_injury(injury_id):
    df   = _load_injuries()
    mask = df['id'] == injury_id
    if not mask.any():
        return jsonify({'status': 'error', 'message': 'Nicht gefunden'}), 404
    data = request.get_json() or {}
    for field in ('description', 'severity', 'date', 'title'):
        if field in data:
            df.loc[mask, field] = data[field]
    if 'status' in data:
        new_status = data['status']
        df.loc[mask, 'status'] = new_status
        if 'resolved_date' in data:
            df.loc[mask, 'resolved_date'] = data['resolved_date']
        elif new_status == 'resolved':
            df.loc[mask, 'resolved_date'] = datetime.now().strftime('%Y-%m-%d')
        else:
            df.loc[mask, 'resolved_date'] = ''
    elif 'resolved_date' in data:
        df.loc[mask, 'resolved_date'] = data['resolved_date']
    _save_injuries(df)
    return jsonify({'status': 'ok'})


@app.route('/api/injuries/<int:injury_id>', methods=['DELETE'])
def api_delete_injury(injury_id):
    df = _load_injuries()
    df = df[df['id'] != injury_id]
    _save_injuries(df)
    return jsonify({'status': 'ok'})


# ── Injury update log ─────────────────────────────────────────────────────────

INJURY_UPDATES_COLS = ['id', 'injury_id', 'ts', 'text']


def _load_injury_updates():
    if not os.path.exists(INJURY_UPDATES_PATH):
        return pd.DataFrame(columns=INJURY_UPDATES_COLS)
    try:
        df = pd.read_csv(INJURY_UPDATES_PATH, dtype=str)
        for col in INJURY_UPDATES_COLS:
            if col not in df.columns:
                df[col] = ''
        return df
    except Exception:
        return pd.DataFrame(columns=INJURY_UPDATES_COLS)


def _save_injury_updates(df):
    df.to_csv(INJURY_UPDATES_PATH, index=False)


@app.route('/api/injuries/<int:injury_id>/updates', methods=['GET'])
def api_get_injury_updates(injury_id):
    df = _load_injury_updates()
    df['injury_id'] = df['injury_id'].astype(str)
    rows = df[df['injury_id'] == str(injury_id)].sort_values('ts')
    return jsonify(rows[INJURY_UPDATES_COLS].to_dict(orient='records'))


@app.route('/api/injuries/<int:injury_id>/updates', methods=['POST'])
def api_add_injury_update(injury_id):
    data = request.get_json() or {}
    text = str(data.get('text', '')).strip()
    if not text:
        return jsonify({'status': 'error', 'message': 'Kein Text'}), 400
    df     = _load_injury_updates()
    new_id = int(df['id'].astype(float).max()) + 1 if not df.empty and df['id'].notna().any() else 1
    ts     = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    new_row = {'id': new_id, 'injury_id': injury_id, 'ts': ts, 'text': text}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_injury_updates(df)
    return jsonify({'status': 'ok', 'update': new_row})


# ═════════════════════════════════════════════════════════════════════════════
# PR Tracker
# ═════════════════════════════════════════════════════════════════════════════

PR_PATH = os.path.join(DB_DIR, 'prs.csv')
PR_COLS = ['id', 'date', 'category', 'value_str', 'value_s', 'is_pb', 'notes', 'activity_id']

# lower = smaller is better (time), higher = bigger is better (power)
PR_CATS = {
    '100m':     {'label': '100 m',    'unit': 'time',  'better': 'lower'},
    '400m':     {'label': '400 m',    'unit': 'time',  'better': 'lower'},
    '1km':      {'label': '1 km',     'unit': 'time',  'better': 'lower'},
    'mile':     {'label': '1 Meile',  'unit': 'time',  'better': 'lower'},
    '5km':      {'label': '5 km',     'unit': 'time',  'better': 'lower'},
    '10km':     {'label': '10 km',    'unit': 'time',  'better': 'lower'},
    'hm':       {'label': 'HM',       'unit': 'time',  'better': 'lower'},
    'marathon': {'label': 'Marathon', 'unit': 'time',  'better': 'lower'},
    'ftp':      {'label': 'FTP',      'unit': 'watts', 'better': 'higher'},
}

# ── Best Efforts (per-activity Zwischenzeiten, berechnet aus GPS-Streams) ─────
BEST_EFFORTS_PATH = os.path.join(DB_DIR, 'best_efforts.csv')
BEST_EFFORTS_COLS = ['activity_id', 'date', 'category', 'time_s', 'value_str', 'is_pb',
                      'start_km', 'end_km', 'start_time_s']

# Zieldistanzen in Metern für die Best-Effort-Berechnung (nur Run/TrailRun)
BEST_EFFORT_DISTANCES = {
    '100m':     100,
    '400m':     400,
    '1km':      1000,
    'mile':     1609.34,
    '5km':      5000,
    '10km':     10000,
    'hm':       21097.5,
    'marathon': 42195,
}

# Schnellste physiologisch plausible Geschwindigkeit (etwas über Usain Bolts WR-Schnitt),
# um GPS-Ausreißer (Sprünge im Stream) aus den Best Efforts zu filtern.
MAX_PLAUSIBLE_SPEED_MS = 10.5
BEST_EFFORT_MIN_TIME_S = {
    cat: dist / MAX_PLAUSIBLE_SPEED_MS for cat, dist in BEST_EFFORT_DISTANCES.items()
}


def _load_prs():
    if not os.path.exists(PR_PATH):
        return pd.DataFrame(columns=PR_COLS)
    try:
        df = pd.read_csv(PR_PATH)
        for col in PR_COLS:
            if col not in df.columns:
                df[col] = ''
        return df
    except Exception:
        return pd.DataFrame(columns=PR_COLS)


def _save_prs(df):
    df.to_csv(PR_PATH, index=False)


@app.route('/api/prs', methods=['GET'])
def api_get_prs():
    df = _load_prs()
    entries = df.sort_values('date', ascending=False).to_dict(orient='records') if not df.empty else []

    bests = {}
    for cat, cfg in PR_CATS.items():
        cat_df = df[df['category'] == cat].copy() if not df.empty else pd.DataFrame()
        if cat_df.empty:
            continue
        cat_df['value_s'] = pd.to_numeric(cat_df['value_s'], errors='coerce')
        cat_df = cat_df.dropna(subset=['value_s'])
        if cat_df.empty:
            continue
        best_row = cat_df.loc[
            cat_df['value_s'].idxmin() if cfg['better'] == 'lower' else cat_df['value_s'].idxmax()
        ]
        cat_df['date'] = pd.to_datetime(cat_df['date'], errors='coerce')
        latest_row = cat_df.sort_values('date').iloc[-1]
        bests[cat] = {
            'best_str':    best_row['value_str'],
            'best_s':      float(best_row['value_s']),
            'best_date':   str(best_row['date'])[:10],
            'latest_str':  latest_row['value_str'],
            'latest_s':    float(latest_row['value_s']),
            'latest_date': str(latest_row['date'])[:10],
            'count':       len(cat_df),
        }

    # Normalise is_pb to proper JSON boolean (CSV stores as string 'True'/'False')
    for entry in entries:
        raw = entry.get('is_pb', False)
        entry['is_pb'] = raw if isinstance(raw, bool) else str(raw).strip().lower() in ('true', '1', 'yes')
        # NaN (z.B. fehlende activity_id bei älteren Einträgen) ist kein gültiges JSON
        for key, val in entry.items():
            if isinstance(val, float) and pd.isna(val):
                entry[key] = None

    return jsonify({'entries': entries, 'bests': bests, 'cats': PR_CATS})


@app.route('/api/prs', methods=['POST'])
def api_add_pr():
    data    = request.get_json() or {}
    df      = _load_prs()
    cat     = data.get('category', '')
    cfg     = PR_CATS.get(cat, {})
    new_val = float(data.get('value_s', 0))

    # Determine if this is a new personal best
    is_pb = True
    if not df.empty and cat in df['category'].values:
        existing = pd.to_numeric(df[df['category'] == cat]['value_s'], errors='coerce').dropna()
        if not existing.empty:
            is_pb = (
                new_val < existing.min() if cfg.get('better') == 'lower'
                else new_val > existing.max()
            )

    new_id  = int(df['id'].max()) + 1 if not df.empty else 1
    new_row = {
        'id':        new_id,
        'date':      data.get('date', datetime.now().strftime('%Y-%m-%d')),
        'category':  cat,
        'value_str': data.get('value_str', ''),
        'value_s':   new_val,
        'is_pb':     is_pb,
        'notes':     data.get('notes', ''),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_prs(df)
    return jsonify({'status': 'ok', 'entry': new_row, 'is_pb': is_pb})


@app.route('/api/prs/<int:pr_id>', methods=['DELETE'])
def api_delete_pr(pr_id):
    df = _load_prs()
    df = df[df['id'] != pr_id]
    _save_prs(df)
    return jsonify({'status': 'ok'})


# ═════════════════════════════════════════════════════════════════════════════
# Garmin als Datenquelle für Aktivitäten (Strava-API ist deaktiviert)
#
# Die Strava-App wurde serverseitig auf "Inactive" gesetzt und liefert keine
# Aktivitäten mehr. Alle bereits vorhandenen (früher via Strava geladenen)
# Aktivitäten in activities.csv – inkl. Kommentaren, KI-Analysen und Polylines –
# bleiben unangetastet erhalten. Neue Aktivitäten, Sekunden-Streams und Routen
# werden ab jetzt aus der Garmin-Connect-API im selben Schema nachgeführt.
# ═════════════════════════════════════════════════════════════════════════════

_garmin_api_client = {'client': None}


def _garmin_client():
    """Eingeloggter Garmin-Connect-Client (einmal pro Prozess wiederverwendet)."""
    if _garmin_api_client['client'] is not None:
        return _garmin_api_client['client']
    from garminconnect import Garmin
    cfg = CONFIG.get('garmin', {})
    api = Garmin(cfg.get('username', ''), cfg.get('password', ''))
    api.login(GARMIN_TOKENSTORE)
    _garmin_api_client['client'] = api
    return api


def _encode_polyline(coords):
    """Kodiert [(lat, lon), ...] in einen Google-Encoded-Polyline-String (Präz. 5),
    identisch zum Strava-Format, das das Frontend (_decodePolyline) bereits liest."""
    def _enc(value):
        v = int(round(value * 1e5))
        v = ~(v << 1) if v < 0 else (v << 1)
        out = ''
        while v >= 0x20:
            out += chr((0x20 | (v & 0x1f)) + 63)
            v >>= 5
        return out + chr(v + 63)

    result, prev_lat, prev_lon = '', 0, 0
    for lat, lon in coords:
        ilat, ilon = int(round(lat * 1e5)), int(round(lon * 1e5))
        result += _enc((ilat - prev_lat) / 1e5) + _enc((ilon - prev_lon) / 1e5)
        prev_lat, prev_lon = ilat, ilon
    return result


# Garmin-sport-typeKey → Strava-Style-Typ, damit activities.csv schema-kompatibel
# bleibt und die bestehende Typ-Filterung (Run/Ride/…) im gesamten App unverändert
# weiterläuft.
GARMIN_SPORT_MAP = {
    'running': 'Run', 'treadmill_running': 'Run', 'track_running': 'Run',
    'indoor_running': 'Run', 'trail_running': 'TrailRun', 'virtual_run': 'VirtualRun',
    'cycling': 'Ride', 'road_biking': 'Ride', 'cyclocross': 'Ride',
    'indoor_cycling': 'VirtualRide', 'virtual_ride': 'VirtualRide',
    'mountain_biking': 'MountainBikeRide', 'gravel_cycling': 'GravelRide',
    'e_bike_fitness': 'EBikeRide', 'e_bike_mountain': 'EBikeRide',
    'walking': 'Walk', 'casual_walking': 'Walk', 'speed_walking': 'Walk',
    'hiking': 'Hike', 'lap_swimming': 'Swim', 'open_water_swimming': 'Swim',
    'strength_training': 'WeightTraining', 'indoor_cardio': 'Workout',
    'hiit': 'HighIntensityIntervalTraining', 'yoga': 'Yoga', 'pilates': 'Pilates',
    'breathwork': 'Workout', 'multi_sport': 'Workout', 'other': 'Workout',
    'stop_watch': 'Workout', 'indoor_climbing': 'RockClimbing', 'bouldering': 'Bouldering',
}


def _garmin_sport_to_type(sport):
    if not sport:
        return 'Workout'
    key = str(sport).strip().lower()
    if key in GARMIN_SPORT_MAP:
        return GARMIN_SPORT_MAP[key]
    # Fallback: snake_case → CamelCase (z. B. neue, unbekannte Garmin-Typen)
    return ''.join(w.capitalize() for w in key.split('_')) or 'Workout'


def _garmin_activity_ids() -> set:
    """Alle Garmin-activity_ids – dient der Quellen-Unterscheidung: Aktivitäten in
    dieser Menge stammen aus Garmin (neue Streams/Runden via Garmin-API), alle
    übrigen IDs in activities.csv sind alte Strava-Aktivitäten (gecachte Streams)."""
    if not os.path.exists(GARMIN_AKTIVITIES_PATH):
        return set()
    try:
        df = pd.read_csv(GARMIN_AKTIVITIES_PATH, usecols=['activity_id'])
        return set(pd.to_numeric(df['activity_id'], errors='coerce').dropna().astype('int64'))
    except Exception:
        return set()


def merge_garmin_activities_into_csv():
    """Neue Garmin-Aktivitäten in activities.csv einspeisen (gleiches Schema wie
    zuvor Strava). Bestehende Zeilen bleiben unangetastet – es werden ausschließlich
    fehlende Aktivitäten angefügt.

    Doppelungen mit bereits vorhandenen (früher via Strava geladenen) Aktivitäten
    werden über Datum + Distanz erkannt und übersprungen, damit die Trainingslast
    nicht doppelt zählt. Lokal gelöschte Aktivitäten (Tombstones) bleiben gelöscht."""
    if not os.path.exists(GARMIN_AKTIVITIES_PATH):
        return load_csv()

    garmin = pd.read_csv(GARMIN_AKTIVITIES_PATH)
    if garmin.empty:
        return load_csv()
    garmin['start_dt'] = pd.to_datetime(garmin['start_time'], errors='coerce')
    garmin = garmin.dropna(subset=['start_dt'])

    def _num(val, default=0.0):
        """Garmin-Zellen können NaN/None sein – robust in eine Zahl wandeln
        (NaN ist truthy, daher reicht `x or 0` nicht)."""
        try:
            f = float(val)
            return default if math.isnan(f) else f
        except (TypeError, ValueError):
            return default

    existing     = load_csv()
    existing_ids = (set(pd.to_numeric(existing['id'], errors='coerce').dropna().astype('int64'))
                    if not existing.empty else set())
    deleted      = _load_deleted_ids()

    # Strava war bis zum Ausfall vollständig. Alles vor dem letzten vorhandenen
    # Eintrag ist also bereits abgedeckt – genuin neu ist nur, was am/nach diesem
    # Datum liegt. Der Cutoff verhindert, dass historische Aktivitäten, deren
    # Garmin-Distanz stärker von der Strava-Distanz abweicht, fälschlich als "neu"
    # doppelt angefügt werden. Bei leerer CSV (Erstinstallation) → alles übernehmen.
    cutoff = None
    if not existing.empty:
        cutoff = pd.to_datetime(existing['date']).max().date()

    # Bestehende Aktivitäten nach Datum gruppieren, um Strava-Zwillinge zu erkennen.
    # Strava-"Moving"-Distanz weicht leicht von Garmins Gesamtdistanz ab, daher
    # Toleranz-Matching (Datum + Distanz ±max(0,25 km / 3 %) bzw. Typ bei Indoor).
    # Einmal zugeordnete Zeilen werden "verbraucht", damit zwei ähnliche
    # Aktivitäten am selben Tag nicht beide auf dieselbe Zeile matchen.
    by_date = {}
    if not existing.empty:
        for i, r in existing.iterrows():
            d  = r['date']
            ds = d.date() if hasattr(d, 'date') else pd.to_datetime(d).date()
            by_date.setdefault(ds, []).append(
                (i, _num(r.get('distance_km')), str(r.get('type', ''))))
    consumed = set()

    def _is_twin(day, dist_km, act_type):
        """True, wenn day/dist bereits als (Strava-)Zeile existiert."""
        for i, ex_km, ex_type in by_date.get(day, []):
            if i in consumed:
                continue
            if dist_km > 0.3:
                if abs(ex_km - dist_km) <= max(0.25, 0.03 * dist_km):
                    consumed.add(i)
                    return True
            else:  # Indoor/ohne Distanz → über Typ + fehlende Distanz matchen
                if ex_km <= 0.3 and ex_type == act_type:
                    consumed.add(i)
                    return True
        return False

    new_rows = []
    for _, a in garmin.iterrows():
        aid = int(a['activity_id'])
        if aid in existing_ids or aid in deleted:
            continue
        dt      = a['start_dt']
        if cutoff is not None and dt.date() < cutoff:
            continue  # historischer Zeitraum → bereits durch Strava abgedeckt
        dist_km = _num(a.get('distance_km'))
        act_type = _garmin_sport_to_type(a.get('sport'))
        if _is_twin(dt.date(), dist_km, act_type):
            continue  # existiert bereits (Strava-Zwilling) → nicht doppelt zählen
        dur = _num(a.get('moving_duration_s')) or _num(a.get('duration_s'))
        new_rows.append({
            'id':            aid,
            'name':          _clean_nc_str(a.get('activity_name')) or 'Garmin-Aktivität',
            'type':          act_type,
            'date':          dt.strftime('%Y-%m-%d'),
            'moving_time':   int(dur),
            'distance_km':   round(dist_km, 2),
            'elevation_m':   round(_num(a.get('ascent_m'))),
            'avg_hr':        _num(a.get('avg_hr')),
            'max_hr':        _num(a.get('max_hr')),
            'avg_watts':     _num(a.get('avg_power_w')),
            'np_watts':      _num(a.get('norm_power_w')),
            'avg_speed_kmh': round(_num(a.get('avg_speed_ms')) * 3.6, 2),
            'polyline':      _fetch_garmin_polyline(aid),
            'comment':       '', 'ai_analysis': '', 'intention': '',
            'rpe':           '', 'intention_met': '',
        })

    if not new_rows:
        return existing

    new_df = pd.DataFrame(new_rows)
    new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
    merged = (pd.concat([existing, new_df], ignore_index=True)
              if not existing.empty else new_df)
    merged = merged.sort_values('date').reset_index(drop=True)
    merged.to_csv(CSV_PATH, index=False)
    print(f"Garmin→activities.csv: {len(new_rows)} neue Aktivität(en) angefügt, {len(merged)} gesamt")
    return merged


def _fetch_garmin_polyline(activity_id):
    """Encodierte GPS-Route einer Garmin-Aktivität (für Karte & Heatmap). Bei
    Aktivitäten ohne GPS (Indoor) oder Fehler: leerer String."""
    try:
        details = _garmin_client().get_activity_details(int(activity_id), maxchart=1, maxpoly=4000)
        pts = ((details.get('geoPolylineDTO') or {}).get('polyline')) or []
        coords = [(p['lat'], p['lon']) for p in pts
                  if p.get('lat') is not None and p.get('lon') is not None]
        return _encode_polyline(coords) if len(coords) >= 2 else ''
    except Exception:
        return ''


def _get_garmin_stream(activity_id):
    """Sekunden-Streams (HF/Cadence/Höhe/Pace/Power) + km-Splits + Runden einer
    Garmin-Aktivität über die Garmin-Connect-Detail-API – gleiche Payload-Struktur
    wie zuvor die Strava-Streams, sodass das Frontend unverändert bleibt."""
    try:
        details = _garmin_client().get_activity_details(int(activity_id),
                                                        maxchart=2000, maxpoly=1)
    except Exception as e:
        return {'error': str(e)}

    descs = {d['key']: d['metricsIndex'] for d in (details.get('metricDescriptors') or [])}
    rows  = details.get('activityDetailMetrics') or []
    if not descs or not rows:
        return {'error': 'no streams available'}

    def col(key):
        idx = descs.get(key)
        if idx is None:
            return []
        return [(r['metrics'][idx] if idx < len(r.get('metrics') or []) else None)
                for r in rows]

    ts_ms = col('directTimestamp')
    t0    = next((t for t in ts_ms if t is not None), None)
    time_full = [round((t - t0) / 1000) if (t is not None and t0 is not None) else None
                 for t in ts_ms]

    hr_full   = col('directHeartRate')
    # Lauf-Kadenz: Garmin liefert 'directRunCadence' pro EINZELFUSS (~80 spm),
    # 'directDoubleCadence' ist der beidbeinige Wert, den die Uhr anzeigt
    # (~160 spm) und den auch die Runden (avg_run_cadence) verwenden. Fehlt der
    # Doppelwert, wird der Einzelfußwert verdoppelt. Rad-Kadenz (RPM) bleibt.
    cad_full  = col('directDoubleCadence') or col('directBikeCadence')
    if not cad_full:
        cad_full = [(v * 2 if v is not None else None) for v in col('directRunCadence')]
    alt_full  = col('directElevation')
    spd_full  = col('directSpeed')     # m/s
    # Power: nativ (Rad-Powermeter, Garmin-Laufpower) – sonst Stryd-Laufpower
    # aus dem Connect-IQ-Entwicklerfeld (ist Garmins Laufleistung an der Uhr
    # abgeschaltet, bleibt directPower leer, der Pod schreibt trotzdem).
    pwr_full  = col('directPower')
    if not any(v is not None and v > 0 for v in pwr_full):
        _t, stryd_w = sp.stryd_power_series(details)
        if stryd_w:
            pwr_full = stryd_w
    dist_full = col('sumDistance')      # m

    # Fehlende Zeit-/Distanzwerte für die Split-Berechnung mit letztem gültigen Wert füllen
    def _ffill(arr):
        out, last = [], 0
        for v in arr:
            last = v if v is not None else last
            out.append(last)
        return out

    tf, dfull = _ffill(time_full), _ffill(dist_full)

    n       = len(rows)
    indices = ([int(i * n / 500) for i in range(500)] if n > 500 else list(range(n)))

    def sub(arr):
        if not arr:
            return []
        return [arr[i] if i < len(arr) else None for i in indices]

    pace = []
    for v in sub(spd_full):
        pace.append(round(1000 / v, 1) if (v and v > 0.5) else None)

    splits = _compute_km_splits(tf, dfull, hr_full, alt_full)
    laps   = _get_garmin_laps(int(activity_id))

    return {
        'time':      sub(time_full),
        'heartrate': sub(hr_full),
        'cadence':   sub(cad_full),
        'altitude':  sub(alt_full),
        'pace':      pace,
        'watts':     sub(pwr_full),
        'splits':    splits,
        'laps':      laps,
        'stream_v':  STREAM_FORMAT_VERSION,    # s. api_activity_streams: Cache-Invalidierung
    }


# ═════════════════════════════════════════════════════════════════════════════
# Activity Streams (per-second HR / cadence / altitude)
# ═════════════════════════════════════════════════════════════════════════════

STREAM_FORMAT_VERSION = 3   # 2 = beidbeinige Lauf-Kadenz · 3 = Stryd-Power im Watt-Stream


def _subsample(arr, target=500):
    """Downsample a list to at most `target` evenly-spaced points."""
    if not arr or len(arr) <= target:
        return arr
    step = len(arr) / target
    return [arr[int(i * step)] for i in range(target)]


def _get_garmin_laps(garmin_activity_id):
    """Liefert die Runden (manuell oder automatisch) einer Garmin-Aktivität.

    Jede Runde enthält die seit Aktivitätsbeginn verstrichene Zeit (Start/Ende
    in Sekunden) und die kumulierte Distanz, damit sie sich auf den
    Strava-Streams (gleicher Zeitachse) als Marker einzeichnen lassen.
    """
    if not os.path.exists(GARMIN_LAPS_PATH):
        return []
    try:
        df = pd.read_csv(GARMIN_LAPS_PATH)
        laps = df[df['activity_id'] == garmin_activity_id].sort_values('lap_number')
        if laps.empty:
            return []

        laps['start_time'] = pd.to_datetime(laps['start_time'], errors='coerce')
        activity_start = laps['start_time'].iloc[0]

        result   = []
        cum_dist = 0.0
        for _, lap in laps.iterrows():
            elapsed_start = (lap['start_time'] - activity_start).total_seconds()
            duration      = float(lap['duration_s']) if pd.notna(lap['duration_s']) else 0.0
            distance_m    = float(lap['distance_m']) if pd.notna(lap['distance_m']) else 0.0
            ascent_m      = float(lap['ascent_m']) if pd.notna(lap.get('ascent_m')) else None
            cum_dist     += distance_m
            result.append({
                'lap_number':     int(lap['lap_number']),
                'elapsed_start_s': round(elapsed_start, 1),
                'elapsed_end_s':   round(elapsed_start + duration, 1),
                'duration_s':      round(duration, 1),
                'distance_m':      round(distance_m),
                'cum_distance_m':  round(cum_dist),
                'avg_pace_s':      (round(duration / (distance_m / 1000), 1)
                                    if distance_m > 0 else None),
                'ascent_m':        (round(ascent_m) if ascent_m is not None else None),
                'avg_hr':          (round(lap['avg_hr']) if pd.notna(lap.get('avg_hr')) else None),
                'avg_cadence':     (round(lap['avg_run_cadence']) if pd.notna(lap.get('avg_run_cadence'))
                                    else round(lap['avg_bike_cadence']) if pd.notna(lap.get('avg_bike_cadence'))
                                    else None),
                'avg_power_w':     (round(lap['avg_power_w']) if pd.notna(lap.get('avg_power_w')) else None),
                'end_lat':         (float(lap['end_lat']) if pd.notna(lap.get('end_lat')) else None),
                'end_lon':         (float(lap['end_lon']) if pd.notna(lap.get('end_lon')) else None),
            })
        return result
    except Exception:
        return []


def _find_garmin_activity_id(strava_id):
    """Matcht eine Strava-Aktivität (id aus activities.csv) auf die zugehörige
    Garmin-Aktivität (activity_id aus GarminConnectData_Aktivities.csv).

    Beide Systeme vergeben unterschiedliche IDs für dieselbe Aktivität (betrifft
    v.a. den Übergangszeitraum, in dem parallel Strava und Garmin synct wurden),
    daher Matching über Datum + Distanz.
    """
    if not os.path.exists(GARMIN_AKTIVITIES_PATH):
        return None
    try:
        strava_df = load_csv()
        row = strava_df[strava_df['id'] == int(strava_id)]
        if row.empty:
            return None
        target_date = row.iloc[0]['date'].date()
        target_km   = float(row.iloc[0]['distance_km'])

        garmin_df = pd.read_csv(GARMIN_AKTIVITIES_PATH)
        garmin_df['start_time'] = pd.to_datetime(garmin_df['start_time'], errors='coerce')
        same_day = garmin_df[garmin_df['start_time'].dt.date == target_date]
        if same_day.empty:
            return None

        diffs = (same_day['distance_km'] - target_km).abs()
        best_idx = diffs.idxmin()
        if diffs.loc[best_idx] > max(0.15, target_km * 0.03):
            return None
        return int(same_day.loc[best_idx, 'activity_id'])
    except Exception:
        return None


def _get_activity_stream(activity_id):
    """Liefert das (subsamplete) Stream-Payload einer Aktivität.

    Nutzt den lokalen Cache (datenbanken/streams/<id>.json). Fehlt er, wird er
    für Garmin-Aktivitäten aus der Garmin-Connect-Detail-API geladen und gecached.
    Alte Strava-Aktivitäten ohne eigene Garmin-ID werden über Datum+Distanz auf
    ihre Garmin-Zwillings-Aktivität gematcht (Übergangszeitraum, in dem beide
    Quellen parallel liefen); ist auch das nicht möglich, bleibt nur der
    bestehende Cache (die Strava-API ist deaktiviert). Rückgabe: dict mit Keys
    time/heartrate/cadence/altitude/pace/watts – oder {'error': ...}.
    """
    os.makedirs(STREAMS_DIR, exist_ok=True)
    cache_path = os.path.join(STREAMS_DIR, f'{activity_id}.json')
    is_garmin  = int(activity_id) in _garmin_activity_ids()
    garmin_id  = int(activity_id) if is_garmin else _find_garmin_activity_id(activity_id)

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        # Alte Caches (vor Einführung von Power-Stream / Pro-km-Splits / Garmin-
        # Runden / Lap-Kadenz+Power) enthalten nicht alle Schlüssel. Wenn eine
        # Garmin-(Zwillings-)ID bekannt ist, einmalig neu laden, damit Power,
        # km-Splits, Runden und Lap-Zusatzfelder erscheinen. Ohne abrufbare
        # Garmin-ID (API inaktiv) bleibt der bestehende Cache erhalten statt
        # weggeworfen zu werden.
        laps_cached  = cached.get('laps') if isinstance(cached, dict) else None
        laps_current = (not laps_cached) or ('avg_cadence' in laps_cached[0])
        # Ältere Caches enthalten die halbierte Einzelfuß-Kadenz (< v2) bzw.
        # keine Stryd-Power (< v3) → einmalig neu laden, sobald eine Garmin-ID
        # bekannt ist.
        fmt_current  = isinstance(cached, dict) and cached.get('stream_v', 1) >= STREAM_FORMAT_VERSION
        if not garmin_id or (isinstance(cached, dict) and 'splits' in cached and 'laps' in cached
                              and laps_current and fmt_current):
            return cached

    # Garmin-ID (eigene oder Zwilling) bekannt → Streams live aus der Garmin-API
    # holen. Ohne Garmin-ID und ohne Cache lässt sich nichts mehr laden (Strava-
    # API inaktiv).
    if not garmin_id:
        return {'error': 'no streams available'}

    payload = _get_garmin_stream(garmin_id)
    if isinstance(payload, dict) and not payload.get('error'):
        with open(cache_path, 'w') as f:
            json.dump(payload, f)
    return payload


def _compute_km_splits(time_full, dist_full, hr_full, alt_full):
    """Pro-Kilometer-Splits aus den Vollauflösungs-Streams.

    Liefert je km: Tempo (s/km, auf vollen km normiert), Ø-Herzfrequenz,
    Höhendifferenz (netto, wie bei Strava – kann negativ sein) und die Distanz
    des Abschnitts. Der letzte (Teil-)Kilometer wird als Bruchteil mit normiertem
    Tempo geführt.
    """
    n = min(len(time_full), len(dist_full))
    if n < 2:
        return []

    has_alt   = bool(alt_full)
    splits    = []
    km_idx    = 1                      # nächste km-Grenze (×1000 m)
    prev_t    = time_full[0]
    seg_hr    = []
    seg_alt0  = alt_full[0] if has_alt else None   # Höhe am Segmentstart

    for i in range(n):
        d = dist_full[i] or 0
        if hr_full and i < len(hr_full) and hr_full[i]:
            seg_hr.append(hr_full[i])

        while d >= km_idx * 1000:
            t_cross = time_full[i]
            alt_now = alt_full[i] if (has_alt and i < len(alt_full)) else None
            elev    = (round(alt_now - seg_alt0)
                       if (alt_now is not None and seg_alt0 is not None) else None)
            splits.append({
                'km':         str(km_idx),
                'pace_s':     round(t_cross - prev_t, 1),
                'avg_hr':     round(sum(seg_hr) / len(seg_hr)) if seg_hr else None,
                'elev':       elev,
                'distance_m': 1000,
            })
            prev_t   = t_cross
            seg_hr   = []
            seg_alt0 = alt_now
            km_idx  += 1

    # Letzter Teil-Kilometer (Tempo auf vollen km normiert)
    total_d = dist_full[n - 1] or 0
    frac    = (total_d - (km_idx - 1) * 1000) / 1000.0
    if frac >= 0.05:
        dt      = time_full[n - 1] - prev_t
        alt_now = alt_full[n - 1] if has_alt else None
        elev    = (round(alt_now - seg_alt0)
                   if (alt_now is not None and seg_alt0 is not None) else None)
        splits.append({
            'km':         f'{frac:.1f}'.replace('.', ','),
            'pace_s':     round(dt / frac, 1) if frac > 0 else 0,
            'avg_hr':     round(sum(seg_hr) / len(seg_hr)) if seg_hr else None,
            'elev':       elev,
            'distance_m': round(total_d - (km_idx - 1) * 1000),
        })

    return splits


@app.route('/api/activity_streams/<int:activity_id>')
def api_activity_streams(activity_id):
    payload = _get_activity_stream(activity_id)
    if isinstance(payload, dict) and payload.get('error'):
        # Bei echtem Fehler 500, bei "keine Streams" 200 mit error-Feld (wie zuvor)
        if payload['error'] == 'no streams available':
            return jsonify(payload)
        return jsonify(payload), 500
    return jsonify(payload)


@app.route('/api/activity_segment/<int:activity_id>')
def api_activity_segment(activity_id):
    """GPS-Koordinaten für ein Zeitfenster einer Aktivität (z.B. der schnellste
    5km-Abschnitt eines Best Efforts), damit das Frontend es auf der Karte
    hervorheben kann. Nutzt die Zeitstempel der GPS-Route (geoPolylineDTO),
    nicht die Distanz -> robust auch bei Gehpausen/GPS-Aussetzern."""
    try:
        start_s = float(request.args.get('start_time_s'))
        end_s   = float(request.args.get('end_time_s'))
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid start_time_s/end_time_s'}), 400

    # Alte Strava-Aktivitäten aus dem Übergangszeitraum haben eine andere ID als
    # ihre Garmin-Zwillings-Aktivität (siehe _get_activity_stream) -> auflösen.
    is_garmin = activity_id in _garmin_activity_ids()
    garmin_id = activity_id if is_garmin else _find_garmin_activity_id(activity_id)
    if not garmin_id:
        return jsonify({'error': 'no route data'}), 404

    try:
        details = _garmin_client().get_activity_details(garmin_id, maxchart=1, maxpoly=4000)
    except Exception as e:
        return jsonify({'error': str(e)}), 502

    pts = [p for p in ((details.get('geoPolylineDTO') or {}).get('polyline') or [])
           if p.get('time') is not None and p.get('lat') is not None and p.get('lon') is not None]
    if len(pts) < 2:
        return jsonify({'error': 'no route data'}), 404

    t0      = pts[0]['time']
    elapsed = [(p['time'] - t0) / 1000 for p in pts]

    idxs = [i for i, t in enumerate(elapsed) if start_s <= t <= end_s]
    if not idxs:
        # Grobe Polyline-Auflösung: kein Punkt exakt im Fenster -> nächstgelegenen nehmen.
        mid = (start_s + end_s) / 2
        idxs = [min(range(len(elapsed)), key=lambda i: abs(elapsed[i] - mid))]
    lo = max(0, idxs[0] - 1)
    hi = min(len(pts) - 1, idxs[-1] + 1)

    coords = [[pts[i]['lat'], pts[i]['lon']] for i in range(lo, hi + 1)]
    return jsonify({'coords': coords})


# ═════════════════════════════════════════════════════════════════════════════
# Wochen-Zonenverteilung (Zeit pro HF-Zone im aktuellen Wochenlaufvolumen)
# ═════════════════════════════════════════════════════════════════════════════

def _zone_for_hr(hr):
    """HF-Wert → Zone 1–5 (None, wenn unter Zone 1 oder kein Wert)."""
    if hr is None:
        return None
    try:
        hr = float(hr)
    except (TypeError, ValueError):
        return None
    for z in (5, 4, 3, 2, 1):
        lo, hi = ZONE_BPM[z]
        if hr >= lo:
            return z
    return None


def _zone_seconds_from_stream(stream):
    """Sekunden je Zone aus einem (subsampleten) Stream.

    Nutzt die erhaltenen Zeit-Offsets: jedem Intervall zwischen zwei Messpunkten
    wird die Zone des Startpunkts zugewiesen. So bleibt die Zeit trotz
    Subsampling korrekt.
    """
    secs = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    if not isinstance(stream, dict):
        return secs
    times = stream.get('time') or []
    hrs   = stream.get('heartrate') or []
    n = min(len(times), len(hrs))
    for i in range(n - 1):
        dt = times[i + 1] - times[i]
        if dt <= 0 or dt > 600:          # Pausen/Lücken (>10 min) ignorieren
            continue
        z = _zone_for_hr(hrs[i])
        if z:
            secs[z] += dt
    return secs


@app.route('/api/zone_distribution')
def api_zone_distribution():
    """Zeit pro HF-Zone für die Läufe der aktuellen Woche (Mo–So).

    Query: ?week=<offset> (0 = aktuelle Woche, -1 = letzte Woche, …).
    """
    try:
        week_offset = int(request.args.get('week', 0))
    except (TypeError, ValueError):
        week_offset = 0

    df = load_activities()
    secs = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    # Beitragende Aktivitäten je Zone (für Hover-Tooltip)
    zone_acts = {1: [], 2: [], 3: [], 4: [], 5: []}
    total_km = 0.0
    n_runs   = 0

    if not df.empty:
        today    = pd.Timestamp(datetime.now().date())
        this_mon = today - pd.Timedelta(days=today.weekday())
        wk_start = this_mon + pd.Timedelta(weeks=week_offset)
        wk_end   = wk_start + pd.Timedelta(days=7)

        runs = df[(df['date'] >= wk_start) & (df['date'] < wk_end) &
                  (df['type'].astype(str).str.contains('Run', case=False, na=False))]

        for _, r in runs.iterrows():
            try:
                act_id = int(r['id'])
            except (TypeError, ValueError):
                continue
            if act_id < 0:               # manuelle Einträge ohne Stream
                continue
            total_km += float(r.get('distance_km', 0) or 0)
            n_runs   += 1
            stream    = _get_activity_stream(act_id)
            run_secs  = _zone_seconds_from_stream(stream)
            date_val  = r.get('date')
            date_fmt  = (date_val.strftime('%a, %d. %b %Y')
                         if hasattr(date_val, 'strftime') else str(date_val)[:10])
            name      = '' if pd.isna(r.get('name')) else str(r.get('name'))
            for z, s in run_secs.items():
                secs[z] += s
                if s >= 1:               # nur nennenswerte Beiträge auflisten
                    zone_acts[z].append({
                        'name':    name or 'Aktivität',
                        'date':    date_fmt,
                        'type':    str(r.get('type', '')),
                        'icon':    ACTIVITY_ICONS.get(str(r.get('type', '')), '🏃'),
                        'seconds': int(round(s)),
                    })

    total_s = sum(secs.values())
    zones = []
    for z in (1, 2, 3, 4, 5):
        s = secs[z]
        lo, hi = ZONE_BPM[z]
        zones.append({
            'zone':       z,
            'seconds':    int(round(s)),
            'pct':        round(100 * s / total_s, 1) if total_s > 0 else 0.0,
            'hr_lo':      lo,
            'hr_hi':      hi,
            'activities': sorted(zone_acts[z], key=lambda a: -a['seconds']),
        })

    return jsonify({
        'zones':         zones,
        'total_seconds': int(round(total_s)),
        'total_km':      round(total_km, 2),
        'n_runs':        n_runs,
        'week_offset':   week_offset,
    })


@app.route('/api/hr_histogram')
def api_hr_histogram():
    """HF-Histogramm: Sekunden pro einzelnem bpm-Wert (90–210) über die letzten 7 Tage
    (rollierend, alle Sporttypen)."""
    bin_width  = 1
    bpm_range  = (90, 210)
    buckets    = {}

    df = load_activities()
    if not df.empty:
        cutoff = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=6)   # heute + 6 Tage zurück = 7 Tage
        recent = df[df['date'] >= cutoff]

        for _, r in recent.iterrows():
            try:
                act_id = int(r['id'])
            except (TypeError, ValueError):
                continue
            if act_id < 0:               # manuelle Einträge ohne Stream
                continue
            stream = _get_activity_stream(act_id)
            if not isinstance(stream, dict):
                continue
            times = stream.get('time') or []
            hrs   = stream.get('heartrate') or []
            n = min(len(times), len(hrs))
            for i in range(n - 1):
                dt = times[i + 1] - times[i]
                if dt <= 0 or dt > 600:   # Pausen/Lücken (>10 min) ignorieren
                    continue
                hr = hrs[i]
                if not hr or hr <= 0:
                    continue
                b = int(hr // bin_width) * bin_width
                buckets[b] = buckets.get(b, 0.0) + dt

    bins = []
    for b in range(bpm_range[0], bpm_range[1] + 1, bin_width):
        secs = buckets.get(b, 0.0)
        bins.append({
            'bpm_lo':  b,
            'bpm_hi':  b + bin_width - 1,
            'seconds': int(round(secs)),
            'zone':    _zone_for_hr(b + bin_width / 2),
        })

    return jsonify({'bins': bins, 'bin_width': bin_width, 'days': 7})


# ═════════════════════════════════════════════════════════════════════════════
# Best Efforts (Zwischenzeiten pro Aktivität, berechnet von backend/compute_best_efforts.py)
# ═════════════════════════════════════════════════════════════════════════════

def _load_best_efforts():
    if not os.path.exists(BEST_EFFORTS_PATH):
        return pd.DataFrame(columns=BEST_EFFORTS_COLS)
    try:
        df = pd.read_csv(BEST_EFFORTS_PATH, dtype={'activity_id': 'Int64'})
        for col in BEST_EFFORTS_COLS:
            if col not in df.columns:
                df[col] = ''
        return df
    except Exception:
        return pd.DataFrame(columns=BEST_EFFORTS_COLS)


def _save_best_efforts(df):
    df.to_csv(BEST_EFFORTS_PATH, index=False)


def _format_best_effort_time(seconds):
    seconds = round(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m}:{s:02d}'


@app.route('/api/best_efforts/<int:activity_id>/<category>', methods=['PUT'])
def api_update_best_effort(activity_id, category):
    data = request.get_json() or {}
    try:
        time_s = float(data.get('time_s'))
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid time_s'}), 400
    if time_s <= 0:
        return jsonify({'error': 'invalid time_s'}), 400

    df = _load_best_efforts()
    mask = (df['activity_id'] == activity_id) & (df['category'] == category)
    if not mask.any():
        return jsonify({'error': 'not found'}), 404

    df.loc[mask, 'time_s']    = round(time_s, 1)
    df.loc[mask, 'value_str'] = _format_best_effort_time(time_s)
    # Manuell korrigierte Zeit passt nicht mehr zum automatisch erkannten
    # GPS-Segment -> Segment-Grenzen verwerfen statt eine falsche Strecke zu zeigen.
    df.loc[mask, 'start_km']     = None
    df.loc[mask, 'end_km']       = None
    df.loc[mask, 'start_time_s'] = None
    _save_best_efforts(df)
    return jsonify({'status': 'ok', 'value_str': _format_best_effort_time(time_s)})


@app.route('/api/best_efforts_summary')
def api_best_efforts_summary():
    df = _load_best_efforts()
    result = {}
    for cat in BEST_EFFORT_DISTANCES:
        label = PR_CATS.get(cat, {}).get('label', cat)
        cat_df = df[df['category'] == cat].copy() if not df.empty else pd.DataFrame()
        if cat_df.empty:
            result[cat] = {'label': label, 'top3': [], 'progression': [], 'all_times': []}
            continue

        cat_df['time_s'] = pd.to_numeric(cat_df['time_s'], errors='coerce')
        cat_df = cat_df.dropna(subset=['time_s'])
        # GPS-Ausreißer (physiologisch unmögliche Geschwindigkeit) ausschließen
        cat_df = cat_df[cat_df['time_s'] >= BEST_EFFORT_MIN_TIME_S.get(cat, 0)]
        cat_df['date_dt'] = pd.to_datetime(cat_df['date'], errors='coerce')

        if cat_df.empty:
            result[cat] = {'label': label, 'top3': [], 'progression': [], 'all_times': []}
            continue

        all_times = [float(t) for t in cat_df['time_s'].tolist()]

        top3 = []
        for i, (_, r) in enumerate(cat_df.sort_values('time_s').head(3).iterrows()):
            top3.append({
                'rank':        i + 1,
                'activity_id': int(r['activity_id']),
                'date':        r['date_dt'].strftime('%Y-%m-%d'),
                'value_str':   str(r['value_str']),
                'time_s':      float(r['time_s']),
            })

        progression = []
        best_so_far = None
        for _, r in cat_df.sort_values('date_dt').iterrows():
            t = float(r['time_s'])
            if best_so_far is None or t < best_so_far:
                best_so_far = t
                progression.append({
                    'date':        r['date_dt'].strftime('%Y-%m-%d'),
                    'value_str':   str(r['value_str']),
                    'time_s':      t,
                    'activity_id': int(r['activity_id']),
                })

        result[cat] = {'label': label, 'top3': top3, 'progression': progression, 'all_times': all_times}

    return jsonify(result)


@app.route('/api/best_efforts/<int:activity_id>')
def api_best_efforts(activity_id):
    df = _load_best_efforts()
    if df.empty:
        return jsonify([])

    df = df.copy()
    df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    df['min_time'] = df['category'].map(BEST_EFFORT_MIN_TIME_S)
    df = df.dropna(subset=['time_s'])
    df = df[df['time_s'] >= df['min_time'].fillna(0)]
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')

    rows = df[df['activity_id'] == activity_id].copy()
    if rows.empty:
        return jsonify([])
    # In Reihenfolge der BEST_EFFORT_DISTANCES sortieren (kurz -> lang)
    order = {cat: i for i, cat in enumerate(BEST_EFFORT_DISTANCES)}
    rows['_order'] = rows['category'].map(order).fillna(99)
    rows = rows.sort_values('_order')
    result = []
    for _, r in rows.iterrows():
        cat = r['category']
        time_s = float(r['time_s'])
        # Aktueller Allzeit-Bestwert für diese Distanz (über alle Aktivitäten).
        cat_all  = df[df['category'] == cat]
        pr_row   = cat_all.loc[cat_all['time_s'].idxmin()] if not cat_all.empty else None
        is_pb    = pr_row is not None and time_s <= float(pr_row['time_s'])
        start_km = pd.to_numeric(r.get('start_km'), errors='coerce')
        end_km   = pd.to_numeric(r.get('end_km'), errors='coerce')
        start_t  = pd.to_numeric(r.get('start_time_s'), errors='coerce')
        result.append({
            'category':      cat,
            'label':         PR_CATS.get(cat, {}).get('label', cat),
            'time_s':        time_s,
            'value_str':     str(r['value_str']),
            'is_pb':         is_pb,
            'start_km':      None if pd.isna(start_km) else round(float(start_km), 2),
            'end_km':        None if pd.isna(end_km) else round(float(end_km), 2),
            'start_time_s':  None if pd.isna(start_t) else round(float(start_t), 1),
            'end_time_s':    None if pd.isna(start_t) else round(float(start_t) + time_s, 1),
            'pr_time_s':     None if pr_row is None else float(pr_row['time_s']),
            'pr_value_str':  None if pr_row is None else str(pr_row['value_str']),
        })
    return jsonify(result)


# ═════════════════════════════════════════════════════════════════════════════
# Garmin Sync (background)
# ═════════════════════════════════════════════════════════════════════════════

_garmin_sync: dict = {'running': False, 'last_result': '—', 'finished_at': ''}


@app.route('/api/garmin_sync', methods=['POST'])
def api_garmin_sync():
    if _garmin_sync['running']:
        return jsonify({'status': 'already_running'})

    def _run():
        _garmin_sync['running'] = True
        try:
            script = os.path.join(BACKEND_DIR, 'sync_garmin_csv.py')
            subprocess.run(['python3', script], check=True,
                           capture_output=True, text=True)
            # Neue Garmin-Aktivitäten in activities.csv einspeisen (Strava-Ersatz),
            # damit Trainingslast/CTL/ATL/Volumen weiterlaufen.
            merge_garmin_activities_into_csv()
            _garmin_sync['last_result'] = 'ok'
            _update_widget_csv()
        except Exception as e:
            _garmin_sync['last_result'] = str(e)
        finally:
            _garmin_sync['running']     = False
            _garmin_sync['finished_at'] = datetime.now().strftime('%H:%M')

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({'status': 'started'})


@app.route('/api/garmin_sync_status')
def api_garmin_sync_status():
    return jsonify(_garmin_sync)


# ═════════════════════════════════════════════════════════════════════════════
# Dashboard-Layout (Karten-Reihenfolge, per Drag & Drop im Edit-Modus gesetzt)
# ═════════════════════════════════════════════════════════════════════════════

DASHBOARD_LAYOUT_PATH = os.path.join(DB_DIR, 'dashboard_layout.json')


@app.route('/api/dashboard_layout', methods=['GET'])
def api_get_dashboard_layout():
    try:
        with open(DASHBOARD_LAYOUT_PATH, encoding='utf-8') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({})


@app.route('/api/dashboard_layout', methods=['POST'])
def api_save_dashboard_layout():
    data = request.get_json() or {}
    # Nur Spalten mit Listen von Widget-IDs übernehmen (Schutz vor kaputten Payloads).
    layout = {k: v for k, v in data.items() if isinstance(v, list)}
    os.makedirs(DB_DIR, exist_ok=True)
    with open(DASHBOARD_LAYOUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(layout, f)
    return jsonify({'status': 'ok'})


# ═════════════════════════════════════════════════════════════════════════════
# Formel-Workout → Garmin
# ═════════════════════════════════════════════════════════════════════════════

def _default_rest_s(work_dur_s: int) -> int:
    """Plausible Pause zwischen Wiederholungen, wenn die Formel keine eigene
    angibt: skaliert mit der Intervalldauer, aber zwischen 1–3 Minuten."""
    return max(60, min(180, round(work_dur_s * 0.5)))


def _parse_workout_formula(s: str) -> list:
    """Parst eine Intervall-Formel wie '5min@z1+3x10min@z4/2min@z1+5min@z1'.

    Jeder '+'-getrennte Teil ist ein Block: '[NxN]Dauer@zZone[/Dauer@zZone]',
    wobei der optionale Teil nach '/' die Pause zwischen den Wiederholungen
    beschreibt. Fehlt sie bei einem Wiederholungs-Block (reps > 1), wird eine
    plausible Standardpause ergänzt, damit auf Garmin nie Reps ohne Pause
    aneinandergereiht werden."""
    s   = s.strip().strip('()')
    pat = re.compile(
        r'(?:(\d+)[x×])?(\d+)(min|sek|h|s)@z(\d)'
        r'(?:/(\d+)(min|sek|h|s)@z(\d))?',
        re.IGNORECASE,
    )
    unit_s = {'min': 60, 'sek': 1, 's': 1, 'h': 3600}
    blocks = []
    for part in s.split('+'):
        m = pat.search(part.strip())
        if not m:
            continue
        reps  = int(m.group(1)) if m.group(1) else 1
        dur   = int(m.group(2))
        unit  = m.group(3).lower()
        zone  = max(1, min(5, int(m.group(4))))
        dur_s = dur * unit_s.get(unit, 60)
        block = {'reps': reps, 'duration_s': dur_s, 'zone': zone}
        if reps > 1:
            if m.group(5):
                rest_dur       = int(m.group(5))
                rest_unit      = m.group(6).lower()
                block['rest_duration_s'] = rest_dur * unit_s.get(rest_unit, 60)
                block['rest_zone']      = max(1, min(5, int(m.group(7))))
            else:
                block['rest_duration_s'] = _default_rest_s(dur_s)
                block['rest_zone']      = 1
        blocks.append(block)
    return blocks


@app.route('/api/push_workout_formula', methods=['POST'])
def api_push_workout_formula():
    try:
        data     = request.get_json() or {}
        formula   = data.get('formula', '').strip()
        date_str  = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        name      = data.get('name') or f'Formel-Workout {datetime.now().strftime("%d.%m.%Y")}'
        intention = (data.get('intention', '') or '').strip()

        built = _build_formula_workout(name, formula)
        if not built:
            return jsonify({'ok': False, 'status': 'error',
                            'error': 'Formel nicht erkannt. Beispiel: 5min@z1+3x10min@z4/2min@z1+20x10sek@z5/1min@z1+5min@z1'}), 400
        payload, description, total_s = built
        description = _apply_intention(payload, description, intention)

        api = _login_garmin()
        workout_id = _upload_and_schedule(api, payload, name, date_str)
        if not workout_id:
            return jsonify({'ok': False, 'status': 'error', 'error': 'Keine workout_id erhalten'}), 500

        gcal_id = _gcal_add_safe(date_str, name, total_s // 60, description)
        try:
            _add_planned_internal(date_str, name, 'run', formula, str(workout_id),
                                  gcal_id=gcal_id, intention=intention)
        except Exception:
            pass
        return jsonify({
            'ok':            True,
            'status':        'ok',
            'workout_id':    workout_id,
            'name':          name,
            'date':          date_str,
            'total_minutes': round(total_s / 60, 1),
            'description':   description,
        })

    except Exception as e:
        return jsonify({'ok': False, 'status': 'error', 'error': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# Ziele (Goals) – CSV-backed
# ═════════════════════════════════════════════════════════════════════════════

GOALS_COLS = ['id', 'name', 'loc', 'date', 'selected', 'website_url']


def _load_goals_csv():
    if not os.path.exists(GOALS_PATH):
        defaults = pd.DataFrame([
            {'id': 1, 'name': '100k Ultramarathon', 'loc': 'Elsass', 'date': '2027-05-13', 'selected': 'False', 'website_url': ''},
            {'id': 2, 'name': 'Sub 3 Marathon',     'loc': 'Berlin', 'date': '2028-08-15', 'selected': 'True',  'website_url': ''},
        ])
        defaults.to_csv(GOALS_PATH, index=False)
        return defaults
    try:
        df = pd.read_csv(GOALS_PATH, dtype=str)
        for col in GOALS_COLS:
            if col not in df.columns:
                df[col] = ''
        return df
    except Exception:
        return pd.DataFrame(columns=GOALS_COLS)


def _save_goals_csv(df):
    df.to_csv(GOALS_PATH, index=False)


@app.route('/api/goals', methods=['GET'])
def api_get_goals():
    df = _load_goals_csv()
    records = []
    for _, row in df.iterrows():
        records.append({
            'id':       int(row['id']) if str(row['id']).isdigit() else 0,
            'name':     str(row.get('name', '')),
            'loc':      '' if str(row.get('loc', '')).lower() in ('nan', '', 'none') else str(row['loc']),
            'date':     str(row.get('date', '')),
            'selected': str(row.get('selected', 'False')).strip().lower() in ('true', '1', 'yes'),
            'website_url': '' if str(row.get('website_url', '')).lower() in ('nan', '', 'none') else str(row['website_url']),
        })
    return jsonify(records)


@app.route('/api/goals', methods=['POST'])
def api_add_goal():
    data = request.get_json() or {}
    df   = _load_goals_csv()
    try:
        new_id = int(df['id'].astype(float).max()) + 1 if not df.empty else 1
    except Exception:
        new_id = len(df) + 1
    if data.get('selected'):
        df['selected'] = 'False'
    new_row = {
        'id':          str(new_id),
        'name':        data.get('name', ''),
        'loc':         data.get('loc', ''),
        'date':        data.get('date', ''),
        'selected':    'True' if data.get('selected') else 'False',
        'website_url': data.get('website_url', ''),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_goals_csv(df)
    return jsonify({'status': 'ok', 'goal': {**new_row, 'id': int(new_id), 'selected': bool(data.get('selected'))}})


@app.route('/api/goals/<int:goal_id>', methods=['PATCH'])
def api_update_goal(goal_id):
    data = request.get_json() or {}
    df   = _load_goals_csv()
    df['id'] = df['id'].astype(str)
    mask = df['id'] == str(goal_id)
    if not mask.any():
        return jsonify({'status': 'error', 'message': 'Nicht gefunden'}), 404
    if data.get('selected') is True:
        df['selected'] = 'False'
    for field in ('name', 'loc', 'date', 'website_url'):
        if field in data:
            df.loc[mask, field] = str(data[field])
    if 'selected' in data:
        df.loc[mask, 'selected'] = 'True' if data['selected'] else 'False'
    _save_goals_csv(df)
    return jsonify({'status': 'ok'})


@app.route('/api/goals/<int:goal_id>', methods=['DELETE'])
def api_delete_goal(goal_id):
    df   = _load_goals_csv()
    df['id'] = df['id'].astype(str)
    df   = df[df['id'] != str(goal_id)]
    _save_goals_csv(df)
    return jsonify({'status': 'ok'})


# ═════════════════════════════════════════════════════════════════════════════
# Geplante Einheiten – CSV-backed
# ═════════════════════════════════════════════════════════════════════════════

PLANNED_COLS = ['id', 'date', 'name', 'type', 'formula', 'garmin_id', 'gcal_id',
                'duration_s', 'zone', 'intention', 'created_at']


def _load_planned():
    if not os.path.exists(PLANNED_PATH):
        return pd.DataFrame(columns=PLANNED_COLS)
    try:
        df = pd.read_csv(PLANNED_PATH, dtype=str)
        for col in PLANNED_COLS:
            if col not in df.columns:
                df[col] = ''
        return df
    except Exception:
        return pd.DataFrame(columns=PLANNED_COLS)


def _save_planned(df):
    df.to_csv(PLANNED_PATH, index=False)


def _add_planned_internal(date_str, name, ptype='run', formula='', garmin_id='',
                          gcal_id='', duration_s='', zone='', intention=''):
    df = _load_planned()
    try:
        new_id = int(df['id'].astype(float).max()) + 1 if not df.empty else 1
    except Exception:
        new_id = len(df) + 1
    new_row = {
        'id':         str(new_id),
        'date':       str(date_str),
        'name':       str(name),
        'type':       str(ptype),
        'formula':    str(formula),
        'garmin_id':  str(garmin_id),
        'gcal_id':    str(gcal_id),
        'duration_s': str(duration_s),
        'zone':       str(zone),
        'intention':  str(intention or ''),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_planned(df)
    return new_row


@app.route('/api/planned_sessions', methods=['GET'])
def api_get_planned():
    df = _load_planned()
    if df.empty:
        return jsonify([])
    return jsonify(df.fillna('').to_dict(orient='records'))


@app.route('/api/planned_sessions', methods=['POST'])
def api_add_planned():
    data = request.get_json() or {}
    row  = _add_planned_internal(
        data.get('date', datetime.now().strftime('%Y-%m-%d')),
        data.get('name', ''),
        data.get('type', 'run'),
        data.get('formula', ''),
        data.get('garmin_id', ''),
        intention=data.get('intention', ''),
    )
    return jsonify({'status': 'ok', 'session': row})


@app.route('/api/planned_sessions/<int:session_id>', methods=['DELETE'])
def api_delete_planned(session_id):
    df   = _load_planned()
    df['id'] = df['id'].astype(str)
    rows = df[df['id'] == str(session_id)]
    if not rows.empty:
        row = rows.iloc[0]
        # Remove from Garmin Connect
        try:
            api = _login_garmin()
            _garmin_delete_safe(api, row.get('garmin_id', ''))
        except Exception:
            pass
        # Remove from Google Calendar
        _gcal_delete_safe(row.get('gcal_id', ''))
    df = df[df['id'] != str(session_id)]
    _save_planned(df)
    return jsonify({'status': 'ok'})


@app.route('/api/planned_sessions/<int:session_id>', methods=['PUT'])
def api_update_planned(session_id):
    """Edit / move a planned workout: recreate it on Garmin + Google Calendar."""
    data = request.get_json() or {}
    df   = _load_planned()
    df['id'] = df['id'].astype(str)
    rows = df[df['id'] == str(session_id)]
    if rows.empty:
        return jsonify({'status': 'error', 'message': 'Nicht gefunden'}), 404
    row = rows.iloc[0]

    def _val(key, default=''):
        v = row.get(key, default)
        return default if (v is None or str(v) in ('', 'nan', 'None')) else v

    new_date  = data.get('date', _val('date'))
    new_name  = data.get('name', _val('name'))
    intention = (data.get('intention', _val('intention')) or '').strip()
    formula  = data.get('formula', _val('formula')).strip() if isinstance(data.get('formula', _val('formula')), str) else ''
    try:
        duration_s = int(data.get('duration_s', _val('duration_s', 1800)) or 1800)
    except Exception:
        duration_s = 1800
    try:
        zone = int(data.get('zone', _val('zone', 2)) or 2)
    except Exception:
        zone = 2

    # Build the new workout (formula takes precedence if present)
    if formula:
        built = _build_formula_workout(new_name, formula)
        if not built:
            return jsonify({'status': 'error', 'message': 'Formel nicht erkannt'}), 400
    else:
        built = _build_simple_workout(new_name, duration_s, zone)
    payload, description, est_s = built
    description = _apply_intention(payload, description, intention)

    try:
        api = _login_garmin()
        # Remove the old Garmin workout + calendar event, then create fresh ones
        _garmin_delete_safe(api, _val('garmin_id'))
        _gcal_delete_safe(_val('gcal_id'))
        workout_id = _upload_and_schedule(api, payload, new_name, new_date)
        if not workout_id:
            return jsonify({'status': 'error', 'message': 'Keine workout_id erhalten'}), 500
        gcal_id = _gcal_add_safe(new_date, new_name, est_s // 60, description)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

    df.loc[df['id'] == str(session_id),
           ['date', 'name', 'formula', 'garmin_id', 'gcal_id', 'duration_s', 'zone', 'intention']] = \
        [new_date, new_name, formula, str(workout_id), gcal_id,
         str(duration_s if not formula else ''), str(zone if not formula else ''), intention]
    _save_planned(df)
    return jsonify({'status': 'ok', 'workout_id': workout_id, 'date': new_date,
                    'name': new_name, 'description': description})


# ═════════════════════════════════════════════════════════════════════════════
# Trainingskalender – Daily TSS/CTL/ATL (ist + geplant) für Trainingsplan-Reiter
# ═════════════════════════════════════════════════════════════════════════════

# Grobe Intensity-Factor-Schätzung pro Zone, um aus einer Formel/Dauer ohne
# tatsächliche HF-/Power-Messung einen plausiblen TSS-Schätzwert für geplante
# (noch nicht absolvierte) Einheiten zu erzeugen.
ZONE_IF_ESTIMATE = {1: 0.65, 2: 0.78, 3: 0.95, 4: 1.15, 5: 1.35}


def _estimate_planned_session(row):
    """Schätzt TSS, Gesamtdauer (s) und Zonen-Blöcke einer geplanten Einheit."""
    formula = str(row.get('formula', '') or '').strip()
    blocks = _parse_workout_formula(formula) if formula and formula.lower() != 'nan' else []
    if blocks:
        tss, total_s, zone_blocks = 0.0, 0, []
        for b in blocks:
            work_s = b['reps'] * b['duration_s']
            IF = ZONE_IF_ESTIMATE.get(b['zone'], 0.8)
            tss     += (work_s / 3600) * IF ** 2 * 100
            total_s += work_s
            zone_blocks.append({'zone': b['zone'], 'duration_s': work_s})
            rest_s = b.get('rest_duration_s')
            if rest_s:
                rest_total = b['reps'] * rest_s
                rest_zone  = b.get('rest_zone', 1)
                IF_r = ZONE_IF_ESTIMATE.get(rest_zone, 0.65)
                tss     += (rest_total / 3600) * IF_r ** 2 * 100
                total_s += rest_total
                zone_blocks.append({'zone': rest_zone, 'duration_s': rest_total})
        return round(tss, 1), total_s, zone_blocks

    try:
        dur_s = int(float(row.get('duration_s') or 0))
    except (TypeError, ValueError):
        dur_s = 0
    try:
        zone = int(float(row.get('zone') or 2))
    except (TypeError, ValueError):
        zone = 2
    if dur_s > 0:
        IF = ZONE_IF_ESTIMATE.get(zone, 0.8)
        tss = round((dur_s / 3600) * IF ** 2 * 100, 1)
        return tss, dur_s, [{'zone': zone, 'duration_s': dur_s}]
    return 0.0, 0, []


@app.route('/api/training_calendar')
def api_training_calendar():
    """
    Tagesgenaue Serie für den Trainingskalender im Trainingsplan-Reiter:
    - ist-TSS/Dauer/Distanz pro Tag (vergangene Tage, inkl. heute)
    - geplant-TSS/Dauer/Zonen-Blöcke pro Tag (zukünftige Tage)
    - CTL/ATL: historisch (compute_load) für die Vergangenheit,
      in die Zukunft fortgeschrieben anhand der geplanten TSS (gestrichelt im Chart)
    """
    df         = load_activities()
    planned_df = _load_planned()

    today       = pd.Timestamp(datetime.now().date())
    past_days   = 56   # 8 Wochen zurück
    future_days = 21   # 3 Wochen voraus
    start = today - pd.Timedelta(days=past_days)
    end   = today + pd.Timedelta(days=future_days)

    load = compute_load(df) if not df.empty else pd.DataFrame(columns=['date', 'tss', 'ctl', 'atl'])
    load_map = {}
    for _, r in load.iterrows():
        d = r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])[:10]
        load_map[d] = r

    df = df.copy()
    if not df.empty:
        df['tss'] = df.apply(calc_tss, axis=1)
    acts_by_day = {}
    if not df.empty:
        for _, r in df.iterrows():
            d = r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])[:10]
            intention = str(r.get('intention', '') or '')
            intention_met = str(r.get('intention_met', '') or '')
            t = str(r.get('type', '') or '')
            acts_by_day.setdefault(d, []).append({
                'id':            r.get('id'),
                'name':          str(r.get('name', '') or ''),
                'type':          t,
                'icon':          ACTIVITY_ICONS.get(t, '🏃'),
                'intention':     '' if intention.lower() in ('nan', 'none', '') else intention,
                'intention_met': '' if intention_met.lower() in ('nan', 'none', '') else intention_met,
                'tss':           round(float(r.get('tss', 0) or 0), 1),
                'duration_s':    int(r.get('moving_time', 0) or 0),
                'distance_km':   round(float(r.get('distance_km', 0) or 0), 2),
            })

    planned_by_day = {}
    planned_est    = {}
    for _, row in planned_df.iterrows():
        d = str(row.get('date', ''))[:10]
        if not d or len(d) != 10:
            continue
        tss, dur_s, zone_blocks = _estimate_planned_session(row)
        intention = str(row.get('intention', '') or '')
        planned_by_day.setdefault(d, []).append({
            'id':          row.get('id'),
            'name':        str(row.get('name', '') or ''),
            'type':        str(row.get('type', '') or ''),
            'intention':   '' if intention.lower() in ('nan', 'none', '') else intention,
            'tss':         tss,
            'duration_s':  dur_s,
            'zone_blocks': zone_blocks,
        })
        prev = planned_est.get(d, (0.0, 0))
        planned_est[d] = (prev[0] + tss, prev[1] + dur_s)

    # CTL/ATL in die Zukunft fortschreiben (EMA mit geschätzter geplanter TSS)
    k_ctl = 2 / (CTL_TC + 1)
    k_atl = 2 / (ATL_TC + 1)
    if not load.empty:
        ctl = float(load.iloc[-1]['ctl'])
        atl = float(load.iloc[-1]['atl'])
    else:
        ctl = atl = 0.0
    proj_map = {}
    cur = today + pd.Timedelta(days=1)
    while cur <= end:
        d   = cur.strftime('%Y-%m-%d')
        tss = planned_est.get(d, (0.0, 0))[0]
        ctl = tss * k_ctl + ctl * (1 - k_ctl)
        atl = tss * k_atl + atl * (1 - k_atl)
        proj_map[d] = {'ctl': round(ctl, 1), 'atl': round(atl, 1)}
        cur += pd.Timedelta(days=1)

    days_out = []
    cur = start
    while cur <= end:
        d         = cur.strftime('%Y-%m-%d')
        is_future = cur > today
        if is_future:
            ctl_v = proj_map.get(d, {}).get('ctl')
            atl_v = proj_map.get(d, {}).get('atl')
        else:
            row   = load_map.get(d)
            ctl_v = round(float(row['ctl']), 1) if row is not None else None
            atl_v = round(float(row['atl']), 1) if row is not None else None

        acts  = acts_by_day.get(d, [])
        plans = planned_by_day.get(d, [])
        days_out.append({
            'date':               d,
            'is_future':          bool(is_future),
            'tss':                round(sum(a['tss'] for a in acts), 1),
            'duration_s':         sum(a['duration_s'] for a in acts),
            'distance_km':        round(sum(a['distance_km'] for a in acts), 2),
            'planned_tss':        round(sum(p['tss'] for p in plans), 1),
            'planned_duration_s': sum(p['duration_s'] for p in plans),
            'ctl':                ctl_v,
            'atl':                atl_v,
            'activities':         acts,
            'planned':            plans,
        })
        cur += pd.Timedelta(days=1)

    return jsonify({'days': days_out, 'today': today.strftime('%Y-%m-%d')})


# ═════════════════════════════════════════════════════════════════════════════
# Makrotrainingsplan – CSV-backed
# ═════════════════════════════════════════════════════════════════════════════

MAKROPLAN_COLS = ['week', 'start_date', 'end_date', 'block_name', 'phase', 'tss_target']


def _load_makroplan():
    if not os.path.exists(MAKROPLAN_PATH):
        return pd.DataFrame(columns=MAKROPLAN_COLS)
    try:
        df = pd.read_csv(MAKROPLAN_PATH, dtype=str)
        for col in MAKROPLAN_COLS:
            if col not in df.columns:
                df[col] = ''
        return df
    except Exception:
        return pd.DataFrame(columns=MAKROPLAN_COLS)


@app.route('/api/makroplan', methods=['GET'])
def api_makroplan():
    df = _load_makroplan()

    # Tatsächlich geleistete TSS pro Tag (für vergangene + laufende Wochen).
    acts = load_activities()
    if not acts.empty:
        acts = acts.copy()
        acts['tss'] = acts.apply(calc_tss, axis=1)
    today = datetime.now().date().isoformat()

    records = []
    for _, row in df.iterrows():
        try:
            tss = float(row.get('tss_target', 0))
        except (ValueError, TypeError):
            tss = 0
        start = str(row.get('start_date', ''))
        end   = str(row.get('end_date', ''))

        # tss_actual nur für Wochen, die bereits begonnen haben (start <= heute);
        # für rein zukünftige Wochen bleibt es None.
        tss_actual = None
        if start and start <= today:
            if acts.empty:
                tss_actual = 0.0
            else:
                mask = (acts['date'] >= pd.Timestamp(start)) & \
                       (acts['date'] <= pd.Timestamp(end))
                tss_actual = round(float(acts.loc[mask, 'tss'].sum()), 1)

        records.append({
            'week':       str(row.get('week', '')),
            'start_date': start,
            'end_date':   end,
            'block_name': str(row.get('block_name', '')),
            'phase':      str(row.get('phase', '')),
            'tss_target': tss,
            'tss_actual': tss_actual,
        })
    return jsonify(records)


def _get_current_makro_block():
    """Return the current week's makroplan row as a dict, or a pre/post-plan fallback."""
    df = _load_makroplan()
    if df.empty:
        return None
    today = datetime.now().date().isoformat()
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        start = str(row.get('start_date', ''))
        end   = str(row.get('end_date', ''))
        if start <= today <= end:
            try:
                tss = float(row.get('tss_target', 0))
            except (ValueError, TypeError):
                tss = 0.0
            return {
                'week':       str(row.get('week', '')),
                'week_num':   i + 1,
                'total_weeks': total,
                'block_name': str(row.get('block_name', '')),
                'phase':      str(row.get('phase', '')),
                'tss_target': tss,
                'start_date': start,
                'end_date':   end,
            }
    # Before plan starts
    first = df.iloc[0]
    if today < str(first.get('start_date', '')):
        return {
            'week': 'Pre',
            'week_num': 0,
            'total_weeks': total,
            'block_name': 'Vor dem Plan',
            'phase': 'Pre',
            'tss_target': 0.0,
            'start_date': str(first.get('start_date', '')),
            'end_date':   str(first.get('start_date', '')),
        }
    # After plan ends
    last = df.iloc[-1]
    try:
        tss = float(last.get('tss_target', 0))
    except (ValueError, TypeError):
        tss = 0.0
    return {
        'week':       str(last.get('week', '')),
        'week_num':   total,
        'total_weeks': total,
        'block_name': str(last.get('block_name', '')),
        'phase':      str(last.get('phase', '')),
        'tss_target': tss,
        'start_date': str(last.get('start_date', '')),
        'end_date':   str(last.get('end_date', '')),
    }


def _makroplan_summary_str():
    """Return a compact human-readable summary of the full makroplan."""
    df = _load_makroplan()
    if df.empty:
        return 'Kein Makroplan verfügbar.'
    lines = []
    for _, row in df.iterrows():
        try:
            tss = float(row.get('tss_target', 0))
        except (ValueError, TypeError):
            tss = 0.0
        lines.append(
            f"  {str(row.get('week','?')):3s} | {str(row.get('block_name','?')):<14s} | "
            f"{str(row.get('phase','?')):<8s} | TSS-Ziel {tss:.0f} | "
            f"{row.get('start_date','')} – {row.get('end_date','')}"
        )
    return '\n'.join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Morning Check-ins – CSV-backed
# ═════════════════════════════════════════════════════════════════════════════

MORNING_COLS = ['id', 'date', 'hrv', 'rhr', 'sleep_quality',
                'readiness', 'mental_clarity', 'intention', 'recovery_index']


def _load_morning():
    if not os.path.exists(MORNING_PATH):
        return pd.DataFrame(columns=MORNING_COLS)
    try:
        df = pd.read_csv(MORNING_PATH, dtype=str)
        for col in MORNING_COLS:
            if col not in df.columns:
                df[col] = ''
        return df
    except Exception:
        return pd.DataFrame(columns=MORNING_COLS)


def _save_morning(df):
    df.to_csv(MORNING_PATH, index=False)


@app.route('/api/morning_checkins', methods=['GET'])
def api_get_morning():
    df = _load_morning()
    if df.empty:
        return jsonify([])

    _EMPTY = {'', 'nan', 'NaN', 'None', 'none'}

    def _num(v):
        if v is None:
            return None
        s = str(v).strip()
        if s in _EMPTY:
            return None
        try:
            f = float(s)
            return None if math.isnan(f) else f
        except Exception:
            return None

    def _str(v):
        s = str(v).strip() if v is not None else ''
        return '' if s in _EMPTY else s

    records = []
    for _, r in df.sort_values('date', ascending=False).iterrows():
        records.append({
            'id':             _str(r.get('id', '')),
            'date':           _str(r.get('date', '')),
            'hrv':            _num(r.get('hrv')),
            'rhr':            _num(r.get('rhr')),
            'sleep_quality':  _num(r.get('sleep_quality')),
            'readiness':      _num(r.get('readiness')),
            'mental_clarity': _num(r.get('mental_clarity')),
            'intention':      _str(r.get('intention', '')),
            'recovery_index': _num(r.get('recovery_index')),
        })
    return jsonify(records)


@app.route('/api/morning_checkins', methods=['POST'])
def api_add_morning():
    data = request.get_json() or {}
    df   = _load_morning()
    today  = data.get('date', datetime.now().strftime('%Y-%m-%d'))

    # Bestehenden Eintrag für dieses Datum (falls vorhanden) als Basis nehmen –
    # neue Werte überschreiben, leere/fehlende Felder behalten den alten Wert.
    existing = df[df['date'].astype(str) == str(today)]
    prev     = existing.iloc[0].to_dict() if not existing.empty else {}
    df = df[df['date'].astype(str) != str(today)]

    if 'id' in prev and str(prev.get('id', '')).strip() not in ('', 'nan'):
        new_id = prev['id']
    else:
        new_id = int(df['id'].astype(float).max()) + 1 if not df.empty and df['id'].notna().any() else 1

    def _merge_num(key):
        v = data.get(key, None)
        if v is None or v == '':
            return prev.get(key, '')
        return v

    def _merge_str(key):
        v = str(data.get(key, '') or '').strip()
        if v == '':
            return str(prev.get(key, '') or '').strip()
        return v

    # Auto-compute Recovery Index from Garmin data for today
    garmin_row, past_df = _get_garmin_row_for_date(today)
    ri_val = None
    if garmin_row:
        atl_today_ri, atl_7d_ri = _atl_for_date(today)
        ri_result = compute_recovery_index(garmin_row, past_df, atl_today_ri, atl_7d_ri)
        ri_val    = ri_result['recovery_index']
    if ri_val is None:
        ri_val_out = prev.get('recovery_index', '')
    else:
        ri_val_out = ri_val

    row = {
        'id':             new_id,
        'date':           today,
        'hrv':            _merge_num('hrv'),
        'rhr':            _merge_num('rhr'),
        'sleep_quality':  _merge_num('sleep_quality'),
        'readiness':      _merge_num('readiness'),
        'mental_clarity': _merge_num('mental_clarity'),
        'intention':      _merge_str('intention'),
        'recovery_index': ri_val_out,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_morning(df)
    return jsonify({'status': 'ok', 'entry': row, 'recovery_index': ri_val})


@app.route('/api/morning_checkins/prefill')
def api_morning_prefill():
    """Return today's Garmin body data for pre-filling the morning check-in form."""
    today = datetime.now().strftime('%Y-%m-%d')
    result = {'date': today, 'hrv': None, 'rhr': None, 'sleep_score': None,
              'body_battery_highest': None, 'body_battery_lowest': None,
              'avg_stress': None, 'hrv_status': None}
    if os.path.exists(GARMIN_BODY_PATH):
        try:
            body = pd.read_csv(GARMIN_BODY_PATH, dtype=str)
            row  = body[body['date'].astype(str).str[:10] == today]
            if row.empty:
                row = body.sort_values('date', ascending=False).head(1)
            if not row.empty:
                r = row.iloc[0]
                def _f(k):
                    try:
                        v = float(r[k])
                        return None if math.isnan(v) else v
                    except Exception:
                        return None
                # hrv_last_night is often NaN; fall back to weekly_avg then 5min_high
                result['hrv']                = (_f('hrv_last_night')
                                                or _f('hrv_weekly_avg')
                                                or _f('hrv_5min_high'))
                result['rhr']                = _f('resting_hr')
                result['sleep_score']        = _f('sleep_score')
                result['body_battery_highest'] = _f('body_battery_highest')
                result['body_battery_lowest']  = _f('body_battery_lowest')
                result['avg_stress']           = _f('sleep_avg_stress') or _f('avg_stress')
                result['hrv_status']           = str(r.get('hrv_status') or '').strip() or None
        except Exception:
            pass
    # Derived prefill values
    if result['sleep_score'] is not None:
        result['sleep_quality_prefill'] = max(1, min(5, round(result['sleep_score'] / 20)))
    # Readiness from body battery: 100→5, 80→4, 60→3, 40→2, <40→1
    bb = result['body_battery_highest']
    if bb is not None:
        result['readiness_prefill'] = max(1, min(5, round(bb / 20)))
    return jsonify(result)


@app.route('/api/hrv_today')
def api_hrv_today():
    """Heutiger HRV-Wert (direkt von Garmin gescraped) + 30-Tage-Statistik
    für die Dashboard-Kachel."""
    result = {'today': None, 'avg30': None, 'sigma': None, 'delta': None,
              'date': None, 'spark': []}
    if not os.path.exists(GARMIN_BODY_PATH):
        return jsonify(result)
    try:
        df = pd.read_csv(GARMIN_BODY_PATH)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')

        # Pro Tag: hrv_last_night → hrv_weekly_avg → hrv_5min_high (wie im Rest der App)
        for col in ('hrv_last_night', 'hrv_weekly_avg', 'hrv_5min_high'):
            if col not in df.columns:
                df[col] = float('nan')
        hrv = (pd.to_numeric(df['hrv_last_night'], errors='coerce')
               .fillna(pd.to_numeric(df['hrv_weekly_avg'], errors='coerce'))
               .fillna(pd.to_numeric(df['hrv_5min_high'], errors='coerce')))
        series = pd.Series(hrv.values, index=df['date'].values).dropna()
        if series.empty:
            return jsonify(result)

        today_val  = float(series.iloc[-1])
        today_date = pd.Timestamp(series.index[-1])
        last30     = series.tail(30)
        avg30      = float(last30.mean())
        sigma      = float(last30.std()) if len(last30) >= 2 else 0.0

        result['today'] = round(today_val, 1)
        result['date']  = today_date.strftime('%Y-%m-%d')
        result['avg30'] = round(avg30, 1)
        result['sigma'] = round(sigma, 1)
        result['delta'] = round(today_val - avg30, 1)
        result['spark'] = [round(float(v), 1) for v in series.tail(7)]
    except Exception:
        pass
    return jsonify(result)


@app.route('/api/load_history')
def api_load_history():
    """Daily CTL/ATL/TSB/Ratio/HRV for the last N days (default 90)."""
    try:
        days = int(request.args.get('days', 90))
        df   = load_activities()
        df['tss'] = df.apply(calc_tss, axis=1)
        load = compute_load(df)
        recent = load.tail(days)
        dates  = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in recent['date']]
        ctl    = [round(float(v), 1) for v in recent['ctl']]
        atl    = [round(float(v), 1) for v in recent['atl']]
        tsb    = [round(float(v), 1) for v in recent['tsb']]
        ratio  = [round(float(a / c), 3) if c > 0 else 0.0
                  for a, c in zip(recent['atl'], recent['ctl'])]

        hrv_series = [None] * len(dates)
        if os.path.exists(GARMIN_BODY_PATH):
            gdf = pd.read_csv(GARMIN_BODY_PATH)
            gdf['date'] = pd.to_datetime(gdf['date'], errors='coerce')
            gdf = gdf.dropna(subset=['date']).sort_values('date')
            for col in ('hrv_last_night', 'hrv_weekly_avg', 'hrv_5min_high'):
                if col not in gdf.columns:
                    gdf[col] = float('nan')
            hrv = (pd.to_numeric(gdf['hrv_last_night'], errors='coerce')
                   .fillna(pd.to_numeric(gdf['hrv_weekly_avg'], errors='coerce'))
                   .fillna(pd.to_numeric(gdf['hrv_5min_high'], errors='coerce')))
            hrv_map = {d: v for d, v in
                       zip(gdf['date'].dt.strftime('%Y-%m-%d'), hrv.tolist())
                       if not (isinstance(v, float) and math.isnan(v))}
            hrv_series = [round(hrv_map[d], 1) if d in hrv_map else None for d in dates]

        return jsonify({'dates': dates, 'ctl': ctl, 'atl': atl,
                        'tsb': tsb, 'ratio': ratio, 'hrv': hrv_series})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# Harada Grid (Mandala Chart) – Sub-3-Marathon-Zielsystem, 9×9-Raster
# ═════════════════════════════════════════════════════════════════════════════

HARADA_CHECKINS_PATH  = os.path.join(DB_DIR, 'harada_checkins.csv')
HARADA_OVERRIDES_PATH = os.path.join(DB_DIR, 'harada_grid_overrides.json')
HARADA_CHECKLIST_PATH = os.path.join(DB_DIR, 'harada_checklist.csv')

# 9×9-Text-Raster, exakt wie im Harada_Grid_Sub3_Marathon.xlsx: Zentrum (4,4) =
# Hauptziel, die 8 umliegenden Block-Mitten = Kategorien, alle übrigen 72
# Zellen = tägliche Handlungs-Items (inkl. der 8 "Mantra"-Zellen rund ums Ziel).
HARADA_GRID_TEXT = [
    ['80 km / Woche', '30 km Long Run', '80% lockeres Tempo', 'Wöchentl. Tempolauf', 'VO2max-Intervalle', 'MP-Long-Run-Passagen', '2x schwer, wenig Reps', 'Tägl. Wadenheben', 'Rumpfstabilität'],
    ['Progressive Steigerung', 'Trainingsumfang', 'Cutback jede 4. Wo.', '1000m-Wiederholungen', 'Tempo & Schwelle', 'Negative Splits', 'Mobility nach dem Lauf', 'Kraft & Prävention', 'Foamrolling'],
    ['Wochenkilometer tracken', '100 km Peak-Woche', '2 Wo. Taper', 'Schwellentest', 'Bergsprints', 'Renntempo-Läufe', 'Einbein-Stabilität', 'Glute Activation', 'Physio-Check monatlich'],
    ['Carb-Load Woche vorher', 'Renn-Gels im Training', 'Tägl. Eiweißziel', 'Disziplin in frühen Km', 'Plan vertrauen', 'Atmung kontrollieren', '8h Schlaf täglich', 'Ein Ruhetag pro Woche', 'HRV beobachten'],
    ['Hydration pro Einheit', 'Ernährung', 'Eisen-Check', 'Keinen Adrenalin-Sprung', 'SUB-3 MARATHON', 'Gleichmäßige Splits', 'Eisbad nach Key-Run', 'Erholung & Schlaf', 'Deload alle 4 Wochen'],
    ['Alle 45min fueln im Race', 'Recovery-Meal', 'Kein Alkohol im Renn-Monat', 'Auf Anstrengung achten, nicht Pace', 'Präsent bleiben', 'An Training glauben', 'Belastung balancieren', 'Schlaf-Konsistenz', 'Keine 2 harten Tage hintereinander'],
    ['Zieleinlauf visualisieren', 'Trainingstagebuch lesen', 'Selbstgespräch-Cues', 'Gleichmäßiges Pacing', 'Streckenprofil studieren', 'Pace-Band pro 5km', 'Rennschuhe einlaufen', 'Renn-Outfit testen', 'GPS-Uhr kalibrieren'],
    ['Mantra für harte Km', 'Mentale Stärke', 'Splits wöchentl. reviewen', 'Negative-Split üben', 'Rennstrategie', 'Zielgruppe/Pacer wählen', 'Anreise & Unterkunft planen', 'Ausrüstung & Logistik', 'Drop-Bag Checkliste'],
    ['Sub-3 Reports lesen', 'Monatl. Mini-Ziele', 'Kleine Erfolge feiern', 'Mentale Checkpoints', 'Plan für km 30', 'Renntag-Routine proben', 'Wetter-Plan B', 'Massage-Termine', 'Gear-Check Woche vorher'],
]

# Pastellfarben je Block (Hintergrund + Textfarbe), an die Excel-Vorlage angelehnt.
HARADA_CATEGORIES = {
    (0, 0): {'key': 'volume',      'color': '#dcfce7', 'text': '#14532d'},
    (0, 1): {'key': 'tempo',       'color': '#fde2e7', 'text': '#9d174d'},
    (0, 2): {'key': 'kraft',       'color': '#ede9fe', 'text': '#5b21b6'},
    (1, 0): {'key': 'ernaehrung',  'color': '#ffedd5', 'text': '#9a3412'},
    (1, 1): {'key': 'goal',        'color': '#fef9c3', 'text': '#854d0e'},
    (1, 2): {'key': 'erholung',    'color': '#dbeafe', 'text': '#1e40af'},
    (2, 0): {'key': 'mental',      'color': '#ede9fe', 'text': '#5b21b6'},
    (2, 1): {'key': 'strategie',   'color': '#dcfce7', 'text': '#14532d'},
    (2, 2): {'key': 'ausruestung', 'color': '#fde2e7', 'text': '#9d174d'},
}


def _build_harada_cells():
    cells = []
    for r in range(9):
        for c in range(9):
            br, bc     = r // 3, c // 3
            is_center  = (r % 3 == 1) and (c % 3 == 1)
            cat        = HARADA_CATEGORIES[(br, bc)]
            if (br, bc) == (1, 1) and is_center:
                kind = 'goal'
            elif is_center:
                kind = 'category'
            else:
                kind = 'item'
            cells.append({
                'id':         r * 9 + c,
                'row':        r,
                'col':        c,
                'text':       HARADA_GRID_TEXT[r][c],
                'kind':       kind,
                'category':   cat['key'],
                'color':      cat['color'],
                'text_color': cat['text'],
            })
    return cells


HARADA_CELLS = _build_harada_cells()


def _load_harada_checkins():
    if not os.path.exists(HARADA_CHECKINS_PATH):
        return pd.DataFrame(columns=['date', 'cell_id'])
    try:
        return pd.read_csv(HARADA_CHECKINS_PATH)
    except Exception:
        return pd.DataFrame(columns=['date', 'cell_id'])


def _save_harada_checkins(df):
    df.to_csv(HARADA_CHECKINS_PATH, index=False)


def _load_harada_overrides():
    if not os.path.exists(HARADA_OVERRIDES_PATH):
        return {}
    try:
        with open(HARADA_OVERRIDES_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_harada_overrides(overrides):
    with open(HARADA_OVERRIDES_PATH, 'w', encoding='utf-8') as f:
        json.dump(overrides, f, ensure_ascii=False)


@app.route('/api/harada_grid')
def api_harada_grid():
    overrides = _load_harada_overrides()
    if not overrides:
        return jsonify(HARADA_CELLS)
    cells = []
    for cell in HARADA_CELLS:
        cell = dict(cell)
        custom = overrides.get(str(cell['id']))
        if custom:
            cell['text'] = custom
        cells.append(cell)
    return jsonify(cells)


@app.route('/api/harada_grid/edit', methods=['POST'])
def api_edit_harada_cell():
    data = request.get_json() or {}
    try:
        cell_id = int(data.get('cell_id'))
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'message': 'invalid cell_id'}), 400
    if not any(c['id'] == cell_id for c in HARADA_CELLS):
        return jsonify({'status': 'error', 'message': 'unknown cell_id'}), 400

    text = str(data.get('text', '')).strip()
    overrides = _load_harada_overrides()
    default_text = next(c['text'] for c in HARADA_CELLS if c['id'] == cell_id)
    if not text or text == default_text:
        overrides.pop(str(cell_id), None)
    else:
        overrides[str(cell_id)] = text
    _save_harada_overrides(overrides)
    return jsonify({'status': 'ok', 'text': text or default_text})


@app.route('/api/harada_checkins', methods=['GET'])
def api_get_harada_checkins():
    """Alle Checkins, gruppiert nach Datum -> Liste von cell_ids (für Grid-Status
    von heute + das GitHub-Style-Streak-Kalender)."""
    df = _load_harada_checkins()
    result = {}
    for _, row in df.iterrows():
        try:
            cell_id = int(row['cell_id'])
        except (TypeError, ValueError):
            continue
        result.setdefault(str(row['date']), []).append(cell_id)
    return jsonify(result)


@app.route('/api/harada_checkins/toggle', methods=['POST'])
def api_toggle_harada_checkin():
    data = request.get_json() or {}
    date = str(data.get('date') or datetime.now().strftime('%Y-%m-%d'))
    try:
        cell_id = int(data.get('cell_id'))
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'message': 'invalid cell_id'}), 400

    if not any(c['id'] == cell_id and c['kind'] == 'item' for c in HARADA_CELLS):
        return jsonify({'status': 'error', 'message': 'Zelle nicht anklickbar (Überschrift)'}), 400

    df   = _load_harada_checkins()
    mask = (df['date'].astype(str) == date) & (pd.to_numeric(df['cell_id'], errors='coerce') == cell_id)
    if mask.any():
        df      = df[~mask]
        checked = False
    else:
        df      = pd.concat([df, pd.DataFrame([{'date': date, 'cell_id': cell_id}])], ignore_index=True)
        checked = True
    _save_harada_checkins(df)
    return jsonify({'status': 'ok', 'checked': checked})


# ── Einmalige Checkliste (ergänzt das tägliche Grid, zählt ebenfalls in die
# Streak-Heatmap ein – aber nur am Tag des Abhakens, nicht wiederkehrend) ────

HARADA_CHECKLIST_COLS = ['id', 'text', 'done', 'completed_at', 'created_at']


def _load_harada_checklist():
    if not os.path.exists(HARADA_CHECKLIST_PATH):
        return pd.DataFrame(columns=HARADA_CHECKLIST_COLS)
    try:
        df = pd.read_csv(HARADA_CHECKLIST_PATH, dtype={'id': int})
        for col in HARADA_CHECKLIST_COLS:
            if col not in df.columns:
                df[col] = ''
        # pandas leitet bei reinen True/False- bzw. rein-leeren Spalten sonst
        # bool- bzw. float64-dtype ab -> spätere Zuweisung eines Strings/Bool
        # würde dann mit LossySetitemError/TypeError abbrechen.
        df['done']         = df['done'].astype(object)
        df['completed_at'] = df['completed_at'].astype(object).where(df['completed_at'].notna(), '')
        return df
    except Exception:
        return pd.DataFrame(columns=HARADA_CHECKLIST_COLS)


def _save_harada_checklist(df):
    df.to_csv(HARADA_CHECKLIST_PATH, index=False)


def _harada_checklist_json(df):
    records = []
    for _, r in df.iterrows():
        records.append({
            'id':           int(r['id']),
            'text':         str(r.get('text', '')),
            'done':         str(r.get('done', '')).strip().lower() in ('true', '1'),
            'completed_at': '' if pd.isna(r.get('completed_at')) else str(r.get('completed_at', '')),
            'created_at':   '' if pd.isna(r.get('created_at')) else str(r.get('created_at', '')),
        })
    return records


@app.route('/api/harada_checklist', methods=['GET'])
def api_get_harada_checklist():
    return jsonify(_harada_checklist_json(_load_harada_checklist()))


@app.route('/api/harada_checklist', methods=['POST'])
def api_add_harada_checklist():
    data = request.get_json() or {}
    text = str(data.get('text', '')).strip()
    if not text:
        return jsonify({'status': 'error', 'message': 'Text fehlt'}), 400
    df     = _load_harada_checklist()
    new_id = int(df['id'].max()) + 1 if not df.empty else 1
    row = {'id': new_id, 'text': text, 'done': False, 'completed_at': '',
           'created_at': datetime.now().isoformat(timespec='seconds')}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_harada_checklist(df)
    return jsonify({'status': 'ok', 'item': _harada_checklist_json(pd.DataFrame([row]))[0]})


@app.route('/api/harada_checklist/<int:item_id>/toggle', methods=['POST'])
def api_toggle_harada_checklist(item_id):
    df   = _load_harada_checklist()
    mask = df['id'] == item_id
    if not mask.any():
        return jsonify({'status': 'error', 'message': 'not found'}), 404
    currently_done = str(df.loc[mask, 'done'].iloc[0]).strip().lower() in ('true', '1')
    if currently_done:
        df.loc[mask, 'done']         = False
        df.loc[mask, 'completed_at'] = ''
    else:
        df.loc[mask, 'done']         = True
        df.loc[mask, 'completed_at'] = datetime.now().isoformat(timespec='seconds')
    _save_harada_checklist(df)
    return jsonify({'status': 'ok', 'item': _harada_checklist_json(df[mask])[0]})


@app.route('/api/harada_checklist/<int:item_id>', methods=['DELETE'])
def api_delete_harada_checklist(item_id):
    df = _load_harada_checklist()
    df = df[df['id'] != item_id]
    _save_harada_checklist(df)
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print('═' * 50)
    print('  Dashboard_Claude Backend')
    print('  → http://localhost:5001')
    print('═' * 50)
    if not os.path.exists(CSV_PATH):
        print('\n⚠  Keine activities.csv gefunden.')
        print('   Zuerst Strava-Daten laden:')
        print('   curl -X POST http://localhost:5001/api/sync\n')
    _port = int(os.environ.get('PORT', 5001))
    # Auto-open browser (only on first start, not on reloader respawn or under PORT override)
    if not os.environ.get('WERKZEUG_RUN_MAIN') and not os.environ.get('PORT'):
        threading.Timer(1.2, lambda: webbrowser.open(f'http://localhost:{_port}')).start()
    app.run(debug=True, port=_port, threaded=True)
