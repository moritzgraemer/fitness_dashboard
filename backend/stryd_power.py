"""
Stryd-Laufpower aus Garmin-Connect-Aktivitätsdetails
====================================================

Ist Garmins eigene Laufleistung an der Uhr deaktiviert, bleibt das native
Powerfeld (avgPower / directPower) leer. Der Stryd-Pod schreibt seine Leistung
trotzdem in die FIT-Datei – als Connect-IQ-Entwicklerfeld seines Datenfelds:

    App-ID  18fb2cf0-1a4b-430d-ad66-988c847421f4
    Feld 0  Power [W]          (pro Sekunde)
    Feld 8  Form Power [W]     Feld 9  Leg Spring Stiffness [kN/m]
    Feld 11 Air Power [W]      Feld 24 Impact Loading Rate [bw/s]
    Feld 10 Lap Power [W]      (Rundenmitteln, nur in der FIT-Datei)
    Feld 99 CP [W]             (Critical Power aus dem Stryd-Profil, Session)

Garmin Connect liefert dieselben Felder in `get_activity_details()` als
`metricDescriptors` mit `appID` + `developerFieldNumber`; der Schlüssel
(`connectIQDeveloperField-NN`) wechselt je Aktivität, App-ID + Feldnummer
sind stabil. Dieses Modul zieht daraus die Leistungsreihe, verdichtet sie zu
Ø/Max/NP und verteilt sie auf Runden. Reines Rechenmodul plus CSV-Anreicherung
(sync + Backfill) – kein Flask.
"""

import math
from pathlib import Path

import pandas as pd

STRYD_APP_ID      = '18fb2cf0-1a4b-430d-ad66-988c847421f4'
STRYD_FIELD_POWER = 0

DB_DIR          = Path(__file__).parent.parent / 'datenbanken'
ACTIVITIES_CSV  = DB_DIR / 'GarminConnectData_Aktivities.csv'
LAPS_CSV        = DB_DIR / 'GarminConnectData_Laps.csv'
APP_ACTIVITIES  = DB_DIR / 'activities.csv'

RUN_SPORTS = ('running', 'trail_running', 'treadmill_running', 'track_running',
              'ultra_run', 'virtual_run')


# ═════════════════════════════════════════════════════════════════════════════
# Extraktion
# ═════════════════════════════════════════════════════════════════════════════

def _descriptor_index(details, key=None, app_id=None, field_no=None):
    for d in (details.get('metricDescriptors') or []):
        if key is not None and d.get('key') == key:
            return d.get('metricsIndex')
        if app_id is not None and d.get('appID') == app_id \
                and d.get('developerFieldNumber') == field_no:
            return d.get('metricsIndex')
    return None


def _column(details, idx):
    rows = details.get('activityDetailMetrics') or []
    if idx is None:
        return []
    return [(r['metrics'][idx] if idx < len(r.get('metrics') or []) else None)
            for r in rows]


def stryd_power_series(details):
    """(timer_s, watts) aus den Aktivitätsdetails – oder ([], []) ohne Stryd.

    timer_s ist Garmins `sumDuration` (Timer-Sekunden seit Start, Pausen
    ausgenommen), dieselbe Zeitbasis wie `duration` der Runden.
    """
    idx = _descriptor_index(details, app_id=STRYD_APP_ID, field_no=STRYD_FIELD_POWER)
    if idx is None:
        return [], []
    watts = _column(details, idx)
    if not any(w is not None and w > 0 for w in watts):
        return [], []
    t_idx = _descriptor_index(details, key='sumDuration')
    timer = _column(details, t_idx) if t_idx is not None else list(range(len(watts)))
    # Zeitlücken mit letztem Wert füllen, damit jede Probe eine Zeit hat
    last = 0.0
    filled = []
    for t in timer:
        if t is not None:
            last = float(t)
        filled.append(last)
    return filled, watts


def has_stryd_power(details) -> bool:
    return bool(stryd_power_series(details)[0])


# ═════════════════════════════════════════════════════════════════════════════
# Verdichtung
# ═════════════════════════════════════════════════════════════════════════════

def power_summary(timer_s, watts) -> dict:
    """Ø (nur Proben > 0 W, also ohne Standzeit), Max und Normalized Power.

    NP nach Coggan: 30-s-gleitendes Mittel, davon die 4. Potenz gemittelt,
    4. Wurzel. Die Proben werden dafür auf ein 1-s-Raster gebracht.
    """
    pairs = [(t, float(w)) for t, w in zip(timer_s, watts) if w is not None]
    if not pairs:
        return {'avg': None, 'max': None, 'np': None}
    moving = [w for _, w in pairs if w > 0]
    avg = sum(moving) / len(moving) if moving else 0.0
    mx  = max(w for _, w in pairs)

    # 1-s-Raster (Garmin liefert je nach maxchart auch gröbere Proben)
    t_end = int(pairs[-1][0])
    grid  = [None] * (t_end + 1)
    for t, w in pairs:
        i = int(t)
        if 0 <= i <= t_end:
            grid[i] = w
    last = 0.0
    for i, v in enumerate(grid):
        if v is None:
            grid[i] = last
        else:
            last = v
    n = len(grid)
    np_val = None
    if n >= 30:
        window = 30
        s = sum(grid[:window])
        fourth = []
        for i in range(window, n + 1):
            fourth.append((s / window) ** 4)
            if i < n:
                s += grid[i] - grid[i - window]
        if fourth:
            np_val = (sum(fourth) / len(fourth)) ** 0.25
    elif moving:
        np_val = avg
    return {
        'avg': round(avg, 1),
        'max': round(mx, 1),
        'np':  round(np_val, 1) if np_val is not None else None,
    }


def lap_powers(timer_s, watts, lap_durations_s) -> list:
    """Ø Power je Runde (Proben > 0 W); Rundengrenzen = kumulierte `duration`."""
    out, t0 = [], 0.0
    pairs = [(t, float(w)) for t, w in zip(timer_s, watts) if w is not None and w > 0]
    for dur in lap_durations_s:
        try:
            d = float(dur)
        except (TypeError, ValueError):
            d = 0.0
        t1 = t0 + max(0.0, d)
        seg = [w for t, w in pairs if t0 <= t < t1]
        out.append(round(sum(seg) / len(seg), 1) if seg else None)
        t0 = t1
    return out


# ═════════════════════════════════════════════════════════════════════════════
# CSV-Anreicherung (Sync + Backfill)
# ═════════════════════════════════════════════════════════════════════════════

def _is_run(sport) -> bool:
    return str(sport or '').strip().lower() in RUN_SPORTS


def _isnan(v) -> bool:
    try:
        return v is None or math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def enrich_stryd_power(api, activity_ids=None, since=None, log=print,
                       sleep_s=0.4, max_calls=None) -> dict:
    """Fehlende Laufpower aus Stryd-Feldern in die drei CSVs schreiben.

    activity_ids: nur diese Garmin-IDs prüfen (Sync: die neu geladenen);
    since:        sonst alle Läufe ab diesem Datum (Backfill), 'YYYY-MM-DD'.
    Angefasst werden nur Läufe ohne avg_power_w. Aktivitäten ohne Stryd-Feld
    werden übersprungen und beim nächsten Lauf erneut geprüft.
    """
    import time

    if not ACTIVITIES_CSV.exists():
        return {'checked': 0, 'filled': 0, 'no_stryd': 0, 'errors': 0}
    acts = pd.read_csv(ACTIVITIES_CSV)
    for col in ('avg_power_w', 'max_power_w', 'norm_power_w'):
        if col not in acts.columns:
            acts[col] = float('nan')

    mask = acts['sport'].apply(_is_run) & acts['avg_power_w'].isna()
    if activity_ids is not None:
        ids = {int(i) for i in activity_ids}
        mask &= acts['activity_id'].astype(int).isin(ids)
    elif since:
        mask &= pd.to_datetime(acts['start_time'], errors='coerce') >= pd.Timestamp(since)
    todo = acts[mask].sort_values('start_time', ascending=False)
    if max_calls:
        todo = todo.head(int(max_calls))

    laps = pd.read_csv(LAPS_CSV) if LAPS_CSV.exists() else pd.DataFrame()
    if not laps.empty and 'avg_power_w' not in laps.columns:
        laps['avg_power_w'] = float('nan')
    app_acts = pd.read_csv(APP_ACTIVITIES) if APP_ACTIVITIES.exists() else pd.DataFrame()

    stats = {'checked': 0, 'filled': 0, 'no_stryd': 0, 'errors': 0}
    filled_ids = []
    for _, row in todo.iterrows():
        aid = int(row['activity_id'])
        stats['checked'] += 1
        try:
            details = api.get_activity_details(aid, maxchart=4000, maxpoly=1)
        except Exception as e:
            stats['errors'] += 1
            log(f'     ! Details {aid}: {e}')
            continue
        timer, watts = stryd_power_series(details or {})
        if not timer:
            stats['no_stryd'] += 1
            continue
        summ = power_summary(timer, watts)
        acts.loc[acts['activity_id'] == aid, ['avg_power_w', 'max_power_w', 'norm_power_w']] = \
            [summ['avg'], summ['max'], summ['np']]

        if not laps.empty:
            lm = laps['activity_id'] == aid
            if lm.any():
                grp = laps[lm].sort_values('lap_number')
                per_lap = lap_powers(timer, watts, grp['duration_s'].tolist())
                laps.loc[grp.index, 'avg_power_w'] = per_lap

        if not app_acts.empty and 'id' in app_acts.columns:
            am = app_acts['id'] == aid
            if am.any():
                app_acts.loc[am, 'avg_watts'] = summ['avg']
                app_acts.loc[am, 'np_watts']  = summ['np'] if summ['np'] is not None else summ['avg']

        stats['filled'] += 1
        filled_ids.append(aid)
        log(f"     Stryd {aid} {str(row.get('start_time', ''))[:10]}: "
            f"Ø {summ['avg']:.0f} W · NP {summ['np']:.0f} W · max {summ['max']:.0f} W")
        if sleep_s:
            time.sleep(sleep_s)

    if filled_ids:
        acts.to_csv(ACTIVITIES_CSV, index=False)
        if not laps.empty:
            laps.to_csv(LAPS_CSV, index=False)
        if not app_acts.empty:
            app_acts.to_csv(APP_ACTIVITIES, index=False)
    stats['ids'] = filled_ids
    return stats
