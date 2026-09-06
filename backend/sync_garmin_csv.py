import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from garminconnect import Garmin

# ---------------------------------------------------
# PFADE
# ---------------------------------------------------

BASE_DIR    = Path(__file__).parent          # dashboard 26/backend/
DB_DIR      = BASE_DIR.parent / "datenbanken"
CONFIG_PATH = BASE_DIR.parent / "config.json"   # zentrale config.json (NICHT in Git)

OUTPUT_ACTIVITIES = DB_DIR / "GarminConnectData_Aktivities.csv"
OUTPUT_LAPS       = DB_DIR / "GarminConnectData_Laps.csv"
OUTPUT_BODY       = DB_DIR / "GarminConnectData_Koerperdaten.csv"

# ---------------------------------------------------
# LOGIN
# ---------------------------------------------------

# Zugangsdaten aus der Umgebung (Server: Fly-Secret APP_CONFIG_JSON) oder,
# lokal, aus config.json. So liegt auf dem Server nie eine Datei mit Passwort.
_raw = os.environ.get("APP_CONFIG_JSON", "").strip()
if _raw:
    config = json.loads(_raw).get("garmin", {})
else:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f).get("garmin", {})
if not config.get("username"):
    raise SystemExit("⚠  Keine Garmin-Zugangsdaten (APP_CONFIG_JSON oder config.json).")

# Gemeinsamer Token-Store mit app.py (datenbanken/.garmin_tokens): eine
# gespeicherte Session statt jedes Mal Passwort-Login – schont das Garmin-
# Ratenlimit und funktioniert auch auf einem Server ohne MFA-Dialog.
GARMIN_TOKENSTORE = str(DB_DIR / ".garmin_tokens")

print("=" * 60)
print("Verbinde mit Garmin Connect...")
api = Garmin(config["username"], config["password"])
api.login(GARMIN_TOKENSTORE)
print("Login erfolgreich.\n")

# ============================================================
# 1. AKTIVITÄTEN – Zusammenfassung
# ============================================================

print("─" * 60)
print("1/3  Aktivitäten laden...")

all_activities = []
start, batch_size = 0, 100
while True:
    batch = api.get_activities(start, batch_size)
    if not batch:
        break
    all_activities.extend(batch)
    if len(batch) < batch_size:
        break
    start += batch_size

print(f"     {len(all_activities)} Aktivitäten von Garmin Connect empfangen.")


def _parse_activity(a):
    at = a.get("activityType") or {}
    et = a.get("eventType") or {}
    return {
        # Identifikation
        "activity_id":               a.get("activityId"),
        "activity_name":             a.get("activityName"),
        "description":               a.get("description"),
        "sport":                     at.get("typeKey"),
        "sub_sport_id":              at.get("parentTypeId"),
        "event_type":                et.get("typeKey"),
        "location":                  a.get("locationName"),
        "favorite":                  a.get("favorite"),
        "personal_record":           a.get("pr"),
        # Zeit
        "start_time":                a.get("startTimeLocal"),
        "start_time_gmt":            a.get("startTimeGMT"),
        "duration_s":                a.get("duration"),
        "elapsed_duration_s":        a.get("elapsedDuration"),
        "moving_duration_s":         a.get("movingDuration"),
        "duration_hours":            (a.get("duration") or 0) / 3600,
        # GPS
        "start_lat":                 a.get("startLatitude"),
        "start_lon":                 a.get("startLongitude"),
        "end_lat":                   a.get("endLatitude"),
        "end_lon":                   a.get("endLongitude"),
        "has_polyline":              a.get("hasPolyline"),
        # Distanz & Geschwindigkeit
        "distance_m":                a.get("distance"),
        "distance_km":               (a.get("distance") or 0) / 1000,
        "avg_speed_ms":              a.get("averageSpeed"),
        "max_speed_ms":              a.get("maxSpeed"),
        # Kalorien
        "calories":                  a.get("calories"),
        # Herzrate
        "avg_hr":                    a.get("averageHR"),
        "max_hr":                    a.get("maxHR"),
        # Höhe
        "ascent_m":                  a.get("elevationGain"),
        "descent_m":                 a.get("elevationLoss"),
        # Training
        "aerobic_training_effect":   a.get("aerobicTrainingEffect"),
        "anaerobic_training_effect": a.get("anaerobicTrainingEffect"),
        "training_effect_label":     a.get("trainingEffectLabel"),
        "training_load":             a.get("activityTrainingLoad"),
        "vo2max":                    a.get("vO2MaxValue"),
        # Laufen
        "avg_run_cadence_spm":       a.get("averageRunningCadenceInStepsPerMinute"),
        "max_run_cadence_spm":       a.get("maxRunningCadenceInStepsPerMinute"),
        "steps":                     a.get("steps"),
        "avg_stride_length_m":       a.get("avgStrideLength"),
        "avg_vertical_oscillation":  a.get("avgVerticalOscillation"),
        "avg_ground_contact_ms":     a.get("avgGroundContactTime"),
        "avg_vertical_ratio":        a.get("avgVerticalRatio"),
        # Radfahren
        "avg_bike_cadence_rpm":      a.get("averageBikingCadenceInRevPerMinute"),
        "max_bike_cadence_rpm":      a.get("maxBikingCadenceInRevPerMinute"),
        "avg_power_w":               a.get("avgPower"),
        "max_power_w":               a.get("maxPower"),
        "norm_power_w":              a.get("normPower"),
        "left_balance_pct":          a.get("leftBalance"),
        "right_balance_pct":         a.get("rightBalance"),
        "avg_left_torque_eff":       a.get("avgLeftTorqueEffectiveness"),
        "avg_right_torque_eff":      a.get("avgRightTorqueEffectiveness"),
        "avg_left_pedal_smooth":     a.get("avgLeftPedalSmoothness"),
        "avg_right_pedal_smooth":    a.get("avgRightPedalSmoothness"),
        # Schwimmen
        "avg_swim_cadence_spm":      a.get("averageSwimCadenceInStrokesPerMinute"),
        "avg_swolf":                 a.get("avgSwolf"),
        "avg_stroke_distance_m":     a.get("avgStrokeDistance"),
        # Temperatur
        "avg_temp_c":                a.get("averageTemperature"),
        "min_temp_c":                a.get("minTemperature"),
        "max_temp_c":                a.get("maxTemperature"),
        # Laps / Splits
        "lap_count":                 a.get("lapCount"),
        "has_splits":                a.get("hasSplits"),
        # Gerät
        "device_id":                 a.get("deviceId"),
    }


new_act_df = pd.DataFrame([_parse_activity(a) for a in all_activities])

if OUTPUT_ACTIVITIES.exists():
    existing_act_df = pd.read_csv(OUTPUT_ACTIVITIES)
    known_ids = set(existing_act_df["activity_id"])
    added_act_df = new_act_df[~new_act_df["activity_id"].isin(known_ids)]
    final_act_df = pd.concat([existing_act_df, added_act_df], ignore_index=True)
    act_added = len(added_act_df)
else:
    final_act_df = new_act_df
    act_added = len(new_act_df)

final_act_df = final_act_df.sort_values("start_time", ascending=False)
final_act_df.to_csv(OUTPUT_ACTIVITIES, index=False)
print(f"     {act_added} neue | {len(final_act_df)} gesamt  →  {OUTPUT_ACTIVITIES.name}")

# ============================================================
# 2. LAP-DATEN
# ============================================================

print("\n─" * 60)
print("2/3  Lap-Daten laden...")

if OUTPUT_LAPS.exists():
    existing_laps_df = pd.read_csv(OUTPUT_LAPS)
    laps_done_ids = set(existing_laps_df["activity_id"])
else:
    existing_laps_df = pd.DataFrame()
    laps_done_ids = set()

to_fetch_laps = [
    a for a in all_activities
    if a.get("hasSplits") and a["activityId"] not in laps_done_ids
]
print(f"     {len(to_fetch_laps)} neue Aktivitäten mit Splits werden abgefragt...")

new_lap_rows = []
for i, a in enumerate(to_fetch_laps, 1):
    act_id = a["activityId"]
    try:
        splits = api.get_activity_splits(act_id)
        laps = (splits or {}).get("lapDTOs") or []
        for lap_num, lap in enumerate(laps, 1):
            new_lap_rows.append({
                "activity_id":        act_id,
                "lap_number":         lap_num,
                "start_time":         lap.get("startTimeLocal") or lap.get("startTimeGMT"),
                "duration_s":         lap.get("duration"),
                "moving_duration_s":  lap.get("movingDuration"),
                "distance_m":         lap.get("distance"),
                "avg_speed_ms":       lap.get("averageSpeed"),
                "max_speed_ms":       lap.get("maxSpeed"),
                "calories":           lap.get("calories"),
                "avg_hr":             lap.get("averageHR"),
                "max_hr":             lap.get("maxHR"),
                "ascent_m":           lap.get("elevationGain"),
                "descent_m":          lap.get("elevationLoss"),
                "avg_run_cadence":    lap.get("averageRunCadence"),
                "avg_bike_cadence":   lap.get("averageBikeCadence"),
                "avg_power_w":        lap.get("averagePower"),
                "avg_stride_m":       lap.get("avgStrideLength"),
                "avg_temp_c":         lap.get("averageTemperature"),
                "intensity":          lap.get("intensity"),
                "lap_trigger":        lap.get("lapTrigger"),
                "start_lat":          lap.get("startLatitude"),
                "start_lon":          lap.get("startLongitude"),
                "end_lat":            lap.get("endLatitude"),
                "end_lon":            lap.get("endLongitude"),
            })
    except Exception as e:
        print(f"     ! Aktivität {act_id}: {e}")

    if i % 5 == 0:
        print(f"     {i}/{len(to_fetch_laps)} abgefragt...")
        time.sleep(1)

if new_lap_rows:
    new_laps_df = pd.DataFrame(new_lap_rows)
    if not existing_laps_df.empty:
        final_laps_df = pd.concat([existing_laps_df, new_laps_df], ignore_index=True)
    else:
        final_laps_df = new_laps_df
    final_laps_df.to_csv(OUTPUT_LAPS, index=False)
    print(f"     {len(new_lap_rows)} neue Lap-Zeilen | {len(final_laps_df)} gesamt  →  {OUTPUT_LAPS.name}")
else:
    print("     Keine neuen Lap-Daten.")

# ============================================================
# 2b. STRYD-LAUFPOWER – Connect-IQ-Feld statt (abgeschalteter) Garmin-Power
# ============================================================

print("\n─" * 60)
print("2b/3 Stryd-Laufpower nachtragen...")
try:
    import sys as _sys
    _sys.path.insert(0, str(BASE_DIR))
    import stryd_power as _sp
    _new_ids = added_act_df["activity_id"].dropna().astype(int).tolist()
    _st = _sp.enrich_stryd_power(api, activity_ids=_new_ids) if _new_ids else \
          {"checked": 0, "filled": 0, "no_stryd": 0, "errors": 0}
    print(f"     {_st['filled']} Läufe mit Stryd-Power ergänzt "
          f"({_st['checked']} geprüft, {_st['no_stryd']} ohne Stryd-Feld, {_st['errors']} Fehler)")
except Exception as e:
    print(f"     ! Stryd-Power: {e}")

# ============================================================
# 3. KÖRPERDATEN – täglich
# ============================================================

print("\n─" * 60)
print("3/3  Körperdaten laden...")


def _local_iso(ms):
    """Garmin liefert 'Local'-Zeitstempel als ms-Epoch, dessen UTC-Interpretation
    bereits die lokale Wanduhrzeit ergibt (kein weiterer TZ-Shift nötig)."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None).isoformat()

# Datumsbereich: früheste Aktivität bis heute
if not final_act_df.empty:
    min_date = pd.to_datetime(final_act_df["start_time"]).min().date()
else:
    min_date = date.today() - timedelta(days=365)
today = date.today()

if OUTPUT_BODY.exists():
    existing_body_df = pd.read_csv(OUTPUT_BODY)
    fetched_dates = set(existing_body_df["date"])
else:
    existing_body_df = pd.DataFrame()
    fetched_dates = set()

all_dates = [
    min_date + timedelta(days=i)
    for i in range((today - min_date).days + 1)
]
# Heute & gestern immer neu abfragen – Schlaf-/HRV-Daten der letzten Nacht
# trudeln oft erst später am Tag bei Garmin ein und würden sonst dauerhaft
# als "leer" stehen bleiben (siehe fetched_dates-Check).
refresh_dates = {str(today), str(today - timedelta(days=1))}
dates_to_fetch = [d for d in all_dates if str(d) not in fetched_dates or str(d) in refresh_dates]
print(f"     {len(dates_to_fetch)} Tage werden abgefragt (Zeitraum: {min_date} – {today})...")

new_body_rows = []
for i, d in enumerate(dates_to_fetch, 1):
    d_str = str(d)
    row = {"date": d_str}

    # Tages-Stats: Schritte, Kalorien, HR, Stress, Body Battery, SpO2, Atmung
    try:
        s = api.get_stats(d_str)
        row.update({
            "steps":                  s.get("totalSteps"),
            "step_goal":              s.get("dailyStepGoal"),
            "distance_m":             s.get("totalDistanceMeters"),
            "active_calories":        s.get("activeKilocalories"),
            "bmr_calories":           s.get("bmrKilocalories"),
            "total_calories":         s.get("totalKilocalories"),
            "floors_ascended":        s.get("floorsAscended"),
            "floors_descended":       s.get("floorsDescended"),
            "intensity_min_moderate": s.get("moderateIntensityMinutes"),
            "intensity_min_vigorous": s.get("vigorousIntensityMinutes"),
            "avg_hr":                 s.get("averageHeartRate"),
            "min_hr":                 s.get("minHeartRate"),
            "max_hr":                 s.get("maxHeartRate"),
            "resting_hr":             s.get("restingHeartRate"),
            "avg_stress":             s.get("averageStressLevel"),
            "max_stress":             s.get("maxStressLevel"),
            "avg_spo2":               s.get("averageSpo2"),
            "avg_respiration":        s.get("avgWakingRespirationValue") or s.get("averageBreathingRate"),
            "body_battery_charged":   s.get("bodyBatteryChargedValue"),
            "body_battery_drained":   s.get("bodyBatteryDrainedValue"),
            "body_battery_highest":   s.get("bodyBatteryHighestValue"),
            "body_battery_lowest":    s.get("bodyBatteryLowestValue"),
        })
    except Exception:
        pass

    # Schlafdaten: Phasen, Score, SpO2, Atmung, Stress
    try:
        sleep = api.get_sleep_data(d_str)
        sd = (sleep or {}).get("dailySleepDTO") or {}
        scores = sd.get("sleepScores") or {}
        overall = scores.get("overall") or {}
        need = sd.get("sleepNeed") or {}
        baseline_min = need.get("baseline")
        actual_min   = need.get("actual")
        total_s      = sd.get("sleepTimeSeconds")
        row.update({
            "sleep_score":          overall.get("value") if isinstance(overall, dict) else None,
            "sleep_total_s":        total_s,
            "sleep_deep_s":         sd.get("deepSleepSeconds"),
            "sleep_light_s":        sd.get("lightSleepSeconds"),
            "sleep_rem_s":          sd.get("remSleepSeconds"),
            "sleep_awake_s":        sd.get("awakeSleepSeconds"),
            "sleep_avg_spo2":       sd.get("averageSpO2Value"),
            "sleep_avg_respiration":sd.get("averageRespirationValue"),
            "sleep_avg_stress":     sd.get("averageStressLevel"),
            "sleep_start_local":    _local_iso(sd.get("sleepStartTimestampLocal")),
            "sleep_end_local":      _local_iso(sd.get("sleepEndTimestampLocal")),
            "sleep_need_baseline_s": baseline_min * 60 if baseline_min is not None else None,
            "sleep_need_actual_s":   actual_min * 60 if actual_min is not None else None,
            "sleep_debt_s": (actual_min * 60 - total_s) if (actual_min is not None and total_s is not None) else None,
        })
    except Exception:
        pass

    # HRV
    try:
        hrv = api.get_hrv_data(d_str)
        hrv_sum = (hrv or {}).get("hrvSummary") or {}
        row.update({
            "hrv_weekly_avg":      hrv_sum.get("weeklyAvg"),
            "hrv_last_night":      hrv_sum.get("lastNight"),
            "hrv_5min_high":       hrv_sum.get("lastNight5MinHigh"),
            "hrv_baseline_low":    hrv_sum.get("baselineLowUpper"),
            "hrv_baseline_high":   hrv_sum.get("baselineHighUpper"),
            "hrv_status":          hrv_sum.get("status"),
        })
    except Exception:
        pass

    new_body_rows.append(row)

    if i % 20 == 0:
        print(f"     {i}/{len(dates_to_fetch)} Tage verarbeitet...")
        time.sleep(1)

if new_body_rows:
    new_body_df = pd.DataFrame(new_body_rows)
    if not existing_body_df.empty:
        # Alte Zeilen für neu abgefragte Tage entfernen (werden durch new_body_df ersetzt)
        existing_body_df = existing_body_df[
            ~existing_body_df["date"].astype(str).isin(new_body_df["date"].astype(str))
        ]
        final_body_df = pd.concat([existing_body_df, new_body_df], ignore_index=True)
    else:
        final_body_df = new_body_df
    final_body_df = final_body_df.sort_values("date", ascending=False)
    final_body_df.to_csv(OUTPUT_BODY, index=False)
    print(f"     {len(new_body_rows)} Tage | {len(final_body_df)} gesamt  →  {OUTPUT_BODY.name}")
else:
    print("     Keine neuen Körperdaten.")

# ---------------------------------------------------
print("\n" + "=" * 60)
print("Sync abgeschlossen.")
print("=" * 60)
