"""Berechnet Best-Effort-Zwischenzeiten (100m, 400m, 1km, Meile, 5km, 10km, HM, Marathon)
für Run / TrailRun Aktivitäten aus den Garmin-GPS-Streams (get_activity_details, volle
1Hz-Auflösung) und gleicht sie mit den persönlichen Rekorden (prs.csv) ab.

RUN_FULL_BACKFILL = True  -> einmaliger Durchlauf über ALLE Run/TrailRun-Aktivitäten
                             (chronologisch, älteste zuerst).
RUN_FULL_BACKFILL = False -> nur neue Aktivitäten, die noch keine Zeile in
                             best_efforts.csv haben (inkrementeller Modus, z.B. nach
                             jedem Garmin-Sync).

Bereits verarbeitete Aktivitäten (vorhanden in best_efforts.csv) werden in beiden
Modi übersprungen -> das Script kann jederzeit unterbrochen und neu gestartet werden.

Hinweis: activities.csv enthält sowohl alte Strava- als auch neue Garmin-IDs (siehe
app.py: merge_garmin_activities_into_csv). Aktivitäten, deren ID direkt in
GarminConnectData_Aktivities.csv vorkommt, werden darüber abgerufen. Für Strava-IDs
ohne direkten Treffer (Übergangszeitraum, in dem Strava- und Garmin-Sync parallel
liefen) wird per Datum+Distanz die Garmin-Zwillings-Aktivität gematcht
(_resolve_garmin_id). Nur wenn auch das fehlschlägt (kein Garmin-Zwilling vorhanden),
wird die Aktivität übersprungen statt zu crashen – die Strava-API ist deaktiviert.
"""

import json
import time
from pathlib import Path

import pandas as pd
from garminconnect import Garmin

RUN_FULL_BACKFILL = False

# ---------------------------------------------------
# PFADE
# ---------------------------------------------------

BASE_DIR    = Path(__file__).parent.parent       # dashboard 26/
DB_DIR      = BASE_DIR / "datenbanken"
CONFIG_PATH = BASE_DIR / "config.json"

ACTIVITIES_PATH    = DB_DIR / "activities.csv"
BEST_EFFORTS_PATH  = DB_DIR / "best_efforts.csv"
PR_PATH            = DB_DIR / "prs.csv"
RAW_STREAMS_DIR    = DB_DIR / "streams_raw"
PROGRESS_PATH      = DB_DIR / "best_efforts_progress.json"
GARMIN_ACTS_PATH   = DB_DIR / "GarminConnectData_Aktivities.csv"

BEST_EFFORTS_COLS = ["activity_id", "date", "category", "time_s", "value_str", "is_pb",
                      "start_km", "end_km", "start_time_s"]
PR_COLS           = ["id", "date", "category", "value_str", "value_s", "is_pb", "notes", "activity_id"]

# Zieldistanzen in Metern (Reihenfolge = Anzeigereihenfolge)
TARGET_DISTANCES = {
    "100m":     100,
    "400m":     400,
    "1km":      1000,
    "mile":     1609.34,
    "5km":      5000,
    "10km":     10000,
    "hm":       21097.5,
    "marathon": 42195,
}

# Schnellste physiologisch plausible Geschwindigkeit (etwas über Usain Bolts WR-Schnitt),
# um GPS-Ausreißer (Sprünge im Stream) als Best Effort zu verwerfen.
MAX_PLAUSIBLE_SPEED_MS = 10.5
MIN_TIME_S = {cat: dist / MAX_PLAUSIBLE_SPEED_MS for cat, dist in TARGET_DISTANCES.items()}

# Nur diese Aktivitätstypen werden verarbeitet
RUN_TYPES = {"Run", "TrailRun"}

RATE_LIMIT_SLEEP_S = 1.0   # Schonpause zwischen Garmin-Detail-Requests


# ---------------------------------------------------
# GARMIN AUTH + STREAMS
# ---------------------------------------------------

with open(CONFIG_PATH, encoding="utf-8") as f:
    _garmin_cfg = json.load(f).get("garmin", {})

_garmin_client = None


def _client():
    global _garmin_client
    if _garmin_client is None:
        _garmin_client = Garmin(_garmin_cfg.get("username", ""), _garmin_cfg.get("password", ""))
        _garmin_client.login()
    return _garmin_client


def _garmin_activity_ids() -> set:
    """IDs, für die eine Garmin-Aktivität existiert (alte Strava-IDs sind nicht
    darin enthalten und lassen sich nicht mehr abrufen)."""
    if not GARMIN_ACTS_PATH.exists():
        return set()
    df = pd.read_csv(GARMIN_ACTS_PATH, usecols=["activity_id"])
    return set(pd.to_numeric(df["activity_id"], errors="coerce").dropna().astype("int64"))


def _resolve_garmin_id(act_id, act_date, act_km, garmin_df):
    """Matcht eine Strava-Aktivität ohne eigene Garmin-ID auf ihre Garmin-
    Zwillings-Aktivität (Übergangszeitraum, in dem Strava- und Garmin-Sync
    parallel liefen) über Datum + Distanz. None, falls kein Zwilling existiert."""
    if garmin_df.empty:
        return None
    same_day = garmin_df[garmin_df["start_time"].dt.date == act_date]
    if same_day.empty:
        return None
    diffs = (same_day["distance_km"] - act_km).abs()
    best_idx = diffs.idxmin()
    if diffs.loc[best_idx] > max(0.15, act_km * 0.03):
        return None
    return int(same_day.loc[best_idx, "activity_id"])


def _fetch_raw_stream(activity_id):
    """Holt time + distance (volle Auflösung, 1Hz) für eine Garmin-Aktivität."""
    cache_path = RAW_STREAMS_DIR / f"{activity_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    details = _client().get_activity_details(activity_id, maxchart=999999, maxpoly=1)
    descs = {d["key"]: d["metricsIndex"] for d in (details.get("metricDescriptors") or [])}
    rows  = details.get("activityDetailMetrics") or []

    def col(key):
        idx = descs.get(key)
        if idx is None:
            return []
        return [(r["metrics"][idx] if idx < len(r.get("metrics") or []) else None)
                for r in rows]

    ts_ms = col("directTimestamp")
    t0    = next((t for t in ts_ms if t is not None), None)
    dist  = col("sumDistance")

    if t0 is None or not dist:
        data = {"time": [], "distance": []}
    else:
        time_s, dist_m, last_d = [], [], 0.0
        for t, d in zip(ts_ms, dist):
            if t is None:
                continue
            last_d = d if d is not None else last_d
            time_s.append((t - t0) / 1000)
            dist_m.append(last_d)
        data = {"time": time_s, "distance": dist_m}

    RAW_STREAMS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)

    time.sleep(RATE_LIMIT_SLEEP_S)
    return data


# ---------------------------------------------------
# BEST-EFFORT-ALGORITHMUS
# ---------------------------------------------------

def _best_effort_time(time_arr, dist_arr, target_m):
    """Kürzeste Zeit (s) und zugehöriges Segment (Start-/Endkilometer, Start-
    Zeitpunkt seit Aktivitätsbeginn), in der `target_m` Meter zurückgelegt
    wurden. None, falls die Aktivität kürzer als target_m ist. Rückgabe:
    (dt_s, start_m, end_m, start_t_s) oder None."""
    n = len(dist_arr)
    if n < 2 or dist_arr[-1] < target_m:
        return None

    best = None
    j = 0
    for i in range(n):
        if j < i:
            j = i
        while j < n and (dist_arr[j] - dist_arr[i]) < target_m:
            j += 1
        if j >= n:
            break
        if j == i:
            continue

        d0, d1 = dist_arr[j - 1], dist_arr[j]
        t0, t1 = time_arr[j - 1], time_arr[j]
        if d1 > d0:
            frac = (target_m - (d0 - dist_arr[i])) / (d1 - d0)
            t_interp = t0 + frac * (t1 - t0)
        else:
            t_interp = t1

        dt = t_interp - time_arr[i]
        if best is None or dt < best[0]:
            best = (dt, dist_arr[i], dist_arr[i] + target_m, time_arr[i])

    return best


def _write_progress(done, total, current_date="", current_name="", started_at="", finished=False):
    PROGRESS_PATH.write_text(json.dumps({
        "done":         done,
        "total":        total,
        "current_date": current_date,
        "current_name": current_name,
        "started_at":   started_at,
        "updated_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished":     finished,
    }))


def _format_time(seconds):
    seconds = round(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    print("=" * 60)
    print("Best Efforts: Berechnung Run/TrailRun-Aktivitäten")
    print("=" * 60)

    activities = pd.read_csv(ACTIVITIES_PATH)
    activities = activities[activities["type"].isin(RUN_TYPES)].copy()
    activities["date"] = pd.to_datetime(activities["date"])
    activities = activities.sort_values("date")  # chronologisch, älteste zuerst

    # Bereits verarbeitete Aktivitäten überspringen
    if BEST_EFFORTS_PATH.exists():
        existing_be = pd.read_csv(BEST_EFFORTS_PATH, dtype={"activity_id": "Int64"})
        done_ids = set(existing_be["activity_id"].dropna().astype(int))
    else:
        existing_be = pd.DataFrame(columns=BEST_EFFORTS_COLS)
        done_ids = set()

    if not RUN_FULL_BACKFILL:
        # Inkrementeller Modus: nur Aktivitäten, die noch nie verarbeitet wurden
        activities = activities[~activities["id"].isin(done_ids)]
    todo = activities[~activities["id"].isin(done_ids)]

    print(f"{len(activities)} Run/TrailRun-Aktivitäten insgesamt | "
          f"{len(done_ids)} bereits verarbeitet | {len(todo)} offen")

    if todo.empty:
        print("Nichts zu tun.")
        return

    # ── PRs laden + aktuellen Bestwert pro Kategorie initialisieren ───────────
    if PR_PATH.exists():
        prs = pd.read_csv(PR_PATH)
        for col in PR_COLS:
            if col not in prs.columns:
                prs[col] = ""
    else:
        prs = pd.DataFrame(columns=PR_COLS)

    current_best = {}  # category -> seconds
    for cat in TARGET_DISTANCES:
        cat_df = prs[prs["category"] == cat]
        if not cat_df.empty:
            vals = pd.to_numeric(cat_df["value_s"], errors="coerce").dropna()
            if not vals.empty:
                current_best[cat] = float(vals.min())

    next_pr_id = int(prs["id"].max()) + 1 if not prs.empty else 1

    garmin_ids = _garmin_activity_ids()
    garmin_df  = pd.read_csv(GARMIN_ACTS_PATH) if GARMIN_ACTS_PATH.exists() else pd.DataFrame()
    if not garmin_df.empty:
        garmin_df["start_time"] = pd.to_datetime(garmin_df["start_time"], errors="coerce")

    total      = len(todo)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    be_df      = existing_be.copy()
    pr_df      = prs.copy()
    new_be_count = 0
    new_pr_count = 0
    skipped_no_garmin = 0

    for n, (_, act) in enumerate(todo.iterrows(), 1):
        act_id   = int(act["id"])
        date_str = act["date"].strftime("%Y-%m-%d")
        name     = str(act.get("name", ""))[:40]
        print(f"[{n}/{total}] {date_str} | {name} (id={act_id})")
        _write_progress(n - 1, total, date_str, name, started_at)

        garmin_id = (act_id if act_id in garmin_ids else
                     _resolve_garmin_id(act_id, act["date"].date(), float(act.get("distance_km", 0)), garmin_df))
        if garmin_id is None:
            # Alte Strava-Aktivität ohne Garmin-Zwilling – Strava-API ist
            # deaktiviert, kann nicht mehr nachberechnet werden.
            skipped_no_garmin += 1
            continue

        try:
            stream = _fetch_raw_stream(garmin_id)
        except Exception as e:
            print(f"     ! Fehler beim Stream-Abruf: {e}")
            continue

        time_arr = stream.get("time") or []
        dist_arr = stream.get("distance") or []
        if len(time_arr) < 2 or len(dist_arr) < 2:
            print("     -> keine GPS-Streamdaten, übersprungen")
            continue

        new_rows = []
        for cat, target_m in TARGET_DISTANCES.items():
            result = _best_effort_time(time_arr, dist_arr, target_m)
            if result is None:
                continue
            bt, start_m, end_m, start_t = result
            if bt < MIN_TIME_S[cat]:
                continue  # GPS-Ausreißer (physiologisch unmögliche Geschwindigkeit)

            value_str = _format_time(bt)
            is_pb = cat not in current_best or bt < current_best[cat]

            new_rows.append({
                "activity_id":  act_id,
                "date":         date_str,
                "category":     cat,
                "time_s":       round(bt, 1),
                "value_str":    value_str,
                "is_pb":        is_pb,
                "start_km":     round(start_m / 1000, 3),
                "end_km":       round(end_m / 1000, 3),
                "start_time_s": round(start_t, 1),
            })

            if is_pb:
                current_best[cat] = bt
                pr_df = pd.concat([pr_df, pd.DataFrame([{
                    "id":          next_pr_id,
                    "date":        date_str,
                    "category":    cat,
                    "value_str":   value_str,
                    "value_s":     round(bt, 1),
                    "is_pb":       True,
                    "notes":       f"Automatisch erkannt (Aktivität {act_id})",
                    "activity_id": act_id,
                }])], ignore_index=True)
                next_pr_id += 1
                new_pr_count += 1

        if new_rows:
            be_df = pd.concat([be_df, pd.DataFrame(new_rows)], ignore_index=True)
            new_be_count += len(new_rows)

        # Nach jeder Aktivität persistieren, damit Fortschritt bei Unterbrechung erhalten bleibt
        be_df.to_csv(BEST_EFFORTS_PATH, index=False)
        pr_df.to_csv(PR_PATH, index=False)

    _write_progress(total, total, "", "", started_at, finished=True)
    print(f"\n{new_be_count} neue Best-Effort-Zeilen -> {BEST_EFFORTS_PATH.name}")
    print(f"{new_pr_count} neue PR-Einträge -> {PR_PATH.name}")
    if skipped_no_garmin:
        print(f"{skipped_no_garmin} alte Strava-Aktivität(en) übersprungen (keine Garmin-ID, API deaktiviert)")
    print("\nFertig.")


if __name__ == "__main__":
    main()
