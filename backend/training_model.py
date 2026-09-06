"""
Trainings-Zustandsmodell – energiesystem-spezifische Fitness (deterministisch)
=============================================================================

Erweitert das eindimensionale CTL/ATL/TSB-Modell (Bannister) um eine
Zerlegung der Fitness in DREI physiologische Systeme:

    aerobic_base   – aerobe Grundlage   (Z1/Z2)
    threshold      – Laktatschwelle      (Z3/Z4)
    vo2            – VO2max / anaerob     (Z5)
    strength       – Kraft / neuromuskulär (Krafttraining, Bouldern)

Hintergrund / Literatur
-----------------------
* 3-Zonen-Intensitätsverteilung (Z1<LT1 · Z2 LT1–LT2 · Z3>LT2) ist der
  etablierte Rahmen für polarisiertes vs. pyramidales Training
  (Seiler 80/20; Filipas et al. 2022 – Pyramidal→Polarisiert ist die beste
  Saison-Progression).
* Mehrere getrennte Impuls-Antworten statt einer Last-Zahl entspricht dem
  dreidimensionalen Impuls-Antwort-Modell (Stöggl et al. 2025).
* Zonen-Gewichtung physiologisch (Bannister-TRIMP, exponentiell) ist bereits
  im TSS via IF² enthalten – deshalb wird der TSS einer Einheit einfach nach
  HF-Zone auf die Systeme aufgeteilt, ohne zusätzliche Gewichtung.

Bewusst NICHT modelliert: economy / neuromuscular. Ihre reale Belastung
(z.B. Hügelsprints) steckt bereits über HF/TSS in threshold/vo2 – ein
separater "weicher" Faktor wäre eine erfundene Größe ohne Literaturanker.

Das Modul ist rein (kein Flask, keine app.py-Importe). app.py liefert die
bereits berechneten Inputs (TSS, CTL/ATL/TSB, Recovery Index) und ruft hier
die Formeln auf.
"""

import math
import pandas as pd

SYSTEMS = ('aerobic', 'threshold', 'vo2', 'strength')

# ── TSS-Aufteilung pro Einheit nach HF-Zone ───────────────────────────────────
# Spalten summieren je auf 1.0. 'low' = unter LT1, 'mid' = LT1–LT2, 'high' = über LT2.
# 'strength' ist KEINE HF-Zone, sondern ein Modalitäts-Bucket: Kraft/Bouldern
# tragen ihre (sRPE-basierte) Last zu 100% ins strength-System und 0% in die
# kardiovaskulären Systeme – HF wäre hier ein irreführender Treiber (s. calc_tss).
ZONE_SPLIT = {
    'low':      {'aerobic': 0.85, 'threshold': 0.15, 'vo2': 0.00, 'strength': 0.00},
    'mid':      {'aerobic': 0.35, 'threshold': 0.60, 'vo2': 0.05, 'strength': 0.00},
    'high':     {'aerobic': 0.10, 'threshold': 0.25, 'vo2': 0.65, 'strength': 0.00},
    'strength': {'aerobic': 0.00, 'threshold': 0.00, 'vo2': 0.00, 'strength': 1.00},
}

# ── Zeitkonstanten je System (Tage) ───────────────────────────────────────────
# Bannister: Fitness τ≈42, Fatigue τ≈14. Schnellere Systeme adaptieren/verfallen
# rascher → kleineres τ. aerob = träge Basis, vo2 = schnell auf- und abbaubar.
# strength: Kraftadaptation ist träge und hält lange – zwischen vo2 und aerob.
SYSTEM_TAU = {'aerobic': 42, 'threshold': 21, 'vo2': 14, 'strength': 35}

# ── Gewichte der Composite-Fitness ────────────────────────────────────────────
# Aus outline.calc_fitness (0.30/0.25/0.20 für aerob/Schwelle/vo2), auf 3 Systeme
# renormiert. Kraft ist für Ausdauerziele (Ultra/Marathon) unterstützend, daher
# nur 0.10; die kardiovaskulären Gewichte sind entsprechend auf Summe 0.90 skaliert.
FITNESS_WEIGHTS = {'aerobic': 0.36, 'threshold': 0.30, 'vo2': 0.24, 'strength': 0.10}

# ── Phasen-Sollverteilung (Last-Anteil, Summe 1.0) ────────────────────────────
# Last-Anteil (nicht Zeit-Anteil): hochintensive Einheiten tragen via IF² viel
# TSS/Minute, daher liegt der vo2-Lastanteil über seinem Zeitanteil (5–12%).
# Progression Base→Peak verschiebt Last von aerob/Schwelle zu vo2 (polarisierend),
# Peak hält Schwelle bewusst hoch (Marathon-Spezifik). Deload = fast nur aerob.
# strength-Sollanteil: in der Base hoch (Maximalkraft-Aufbau), zum Wettkampf hin
# auf reines Erhalten reduziert (polarisierende Periodisierung). Summe je 1.0.
PHASE_EMPHASIS = {
    'Base':   {'aerobic': 0.62, 'threshold': 0.20, 'vo2': 0.06, 'strength': 0.12},
    'Build':  {'aerobic': 0.55, 'threshold': 0.21, 'vo2': 0.16, 'strength': 0.08},
    'Peak':   {'aerobic': 0.50, 'threshold': 0.25, 'vo2': 0.20, 'strength': 0.05},
    'Taper':  {'aerobic': 0.54, 'threshold': 0.24, 'vo2': 0.16, 'strength': 0.06},
    'Deload': {'aerobic': 0.80, 'threshold': 0.12, 'vo2': 0.04, 'strength': 0.04},
    'Race':   {'aerobic': 0.60, 'threshold': 0.24, 'vo2': 0.12, 'strength': 0.04},
}

# Gewicht des Entwicklungs-Rückstands relativ zum Dosis-Defizit bei der
# Zielsystem-Wahl (freier Parameter, an Daten kalibrierbar).
LAMBDA = 0.3


def phase_emphasis(phase: str) -> dict:
    """Sollverteilung für eine Phase; Fallback auf Base (z.B. 'Pre', leer)."""
    return PHASE_EMPHASIS.get(str(phase), PHASE_EMPHASIS['Base'])


def _zone_bucket(avg_hr, hr_lt1, hr_lt2, is_strength=False) -> str:
    """Ordnet eine Einheit einer Zonen-/Modalitätsgruppe zu.

    is_strength überschreibt die HF-Logik: Kraft/Bouldern → 'strength', damit die
    Last nicht über die (irreführende) Herzfrequenz in die Ausdauersysteme fließt.
    """
    if is_strength:
        return 'strength'
    try:
        hr = float(avg_hr)
    except (TypeError, ValueError):
        return 'low'
    if math.isnan(hr) or hr <= 0:
        return 'low'          # ohne HF: als lockere aerobe Einheit behandeln
    if hr < hr_lt1:
        return 'low'
    if hr < hr_lt2:
        return 'mid'
    return 'high'


def split_tss(avg_hr, tss, hr_lt1, hr_lt2, is_strength=False) -> dict:
    """Verteilt den TSS einer Einheit auf die vier Systeme."""
    w = ZONE_SPLIT[_zone_bucket(avg_hr, hr_lt1, hr_lt2, is_strength)]
    try:
        t = float(tss)
    except (TypeError, ValueError):
        t = 0.0
    if math.isnan(t) or t <= 0:
        t = 0.0
    return {s: t * w[s] for s in SYSTEMS}


def build_system_timeseries(acts: pd.DataFrame, hr_lt1, hr_lt2,
                            today=None, window_days=730) -> pd.DataFrame:
    """
    Tägliche EMA je System über die letzten `window_days` Tage.

    acts: DataFrame mit Spalten 'date' (datetime), 'avg_hr', 'tss' und optional
          'is_strength' (bool) – Kraft/Bouldern-Einheiten.
    Rückgabe: DataFrame (aufsteigend nach Datum) mit Spalten
              date, aerobic, threshold, vo2, strength  (= EMA-Zustände).
    """
    if today is None:
        today = pd.Timestamp(pd.Timestamp.now().date())
    else:
        today = pd.Timestamp(today)

    # System-Beiträge je Einheit
    contrib = {s: [] for s in SYSTEMS}
    dates = []
    for _, r in acts.iterrows():
        parts = split_tss(r.get('avg_hr'), r.get('tss'), hr_lt1, hr_lt2,
                          bool(r.get('is_strength', False)))
        dates.append(pd.Timestamp(r['date']).normalize())
        for s in SYSTEMS:
            contrib[s].append(parts[s])

    daily = pd.DataFrame({'date': dates, **contrib})
    # pro Kalendertag summieren (mehrere Einheiten am selben Tag = additiv)
    daily = daily.groupby('date', as_index=False).sum()

    # lückenloser Tagesindex inkl. Ruhetagen (Beitrag 0) → EMA korrekt
    full = pd.DataFrame({'date': pd.date_range(today - pd.Timedelta(days=window_days), today)})
    merged = full.merge(daily, on='date', how='left').fillna(0.0)

    for s in SYSTEMS:
        k = 2.0 / (SYSTEM_TAU[s] + 1)
        state, out = 0.0, []
        for v in merged[s]:
            state = v * k + state * (1 - k)
            out.append(state)
        merged[s] = out
    return merged


def normalize_scores(ts: pd.DataFrame, ref_days=90, offset=0) -> dict:
    """
    System-Zustände auf 0–100 normieren – relativ zum eigenen `ref_days`-Maximum
    ("100 = dein Bestniveau in diesem System").

    offset>0 liefert die Scores von vor `offset` Tagen, normiert auf DASSELBE
    aktuelle Maximum → direkt mit den heutigen Scores vergleichbar (Geist-Polygon).
    """
    idx = -1 - offset
    scores = {}
    for s in SYSTEMS:
        peak = float(ts[s].tail(ref_days).max())
        if peak <= 0 or len(ts[s]) <= offset:
            scores[s] = 0.0
            continue
        cur = float(ts[s].iloc[idx])
        scores[s] = round(max(0.0, min(100.0, 100.0 * cur / peak)), 1)
    return scores


def recent_distribution(acts: pd.DataFrame, hr_lt1, hr_lt2,
                        days=14, today=None) -> dict:
    """Last-Anteil je System über die letzten `days` Tage (Summe 1.0)."""
    if today is None:
        today = pd.Timestamp(pd.Timestamp.now().date())
    else:
        today = pd.Timestamp(today)
    cutoff = today - pd.Timedelta(days=days)
    totals = {s: 0.0 for s in SYSTEMS}
    for _, r in acts.iterrows():
        d = pd.Timestamp(r['date']).normalize()
        if d < cutoff or d > today:
            continue
        parts = split_tss(r.get('avg_hr'), r.get('tss'), hr_lt1, hr_lt2,
                          bool(r.get('is_strength', False)))
        for s in SYSTEMS:
            totals[s] += parts[s]
    grand = sum(totals.values())
    if grand <= 0:
        return {s: 0.0 for s in SYSTEMS}
    return {s: round(totals[s] / grand, 3) for s in SYSTEMS}


def choose_target_system(scores: dict, dist: dict, phase: str,
                         lam: float = LAMBDA) -> tuple:
    """
    Zielsystem = größtes Defizit relativ zum Phasen-Soll, NICHT absolutes Minimum.

        gap_s      = e_s − a_s                  (Dosis-Defizit vs. Phasen-Soll)
        develop_s  = e_s · (1 − score_s/100)    (Entwicklungs-Rückstand, phasen-gewichtet)
        priority_s = gap_s + λ · develop_s

    Liefert (target, ranking) mit ranking = sortierte Liste (system, priority, gap).
    Hat ein bereits ausreichend dosiertes System (gap≤0) niedrige Priorität,
    fällt es automatisch zurück → "eine Einheit deckt den Wochenreiz".
    """
    e = phase_emphasis(phase)
    rows = []
    for s in SYSTEMS:
        gap = e[s] - dist.get(s, 0.0)
        develop = e[s] * (1 - scores.get(s, 0.0) / 100.0)
        prio = gap + lam * develop
        rows.append({'system': s, 'priority': round(prio, 3),
                     'gap': round(gap, 3), 'emphasis': e[s],
                     'share': round(dist.get(s, 0.0), 3)})
    rows.sort(key=lambda x: x['priority'], reverse=True)
    target = rows[0]['system'] if rows and rows[0]['gap'] > 0 else None
    if target is None and rows:
        target = rows[0]['system']     # alle über Soll → höchste Restpriorität
    return target, rows


def composite_fitness(scores: dict) -> float:
    """Gewichtete Gesamt-Fitness aus den drei System-Scores (0–100)."""
    return round(sum(FITNESS_WEIGHTS[s] * scores.get(s, 0.0) for s in SYSTEMS), 1)


def momentum(ts: pd.DataFrame, days_back=42, ref_days=90) -> float:
    """
    Veränderung der normierten Composite-Fitness ggü. vor `days_back` Tagen.
    Positiv = Fitness baut auf.
    """
    comp = sum(FITNESS_WEIGHTS[s] * ts[s] for s in SYSTEMS)
    peak = float(comp.tail(ref_days).max())
    if peak <= 0 or len(comp) <= days_back:
        return 0.0
    norm = (100.0 * comp / peak).clip(0, 100)
    return round(float(norm.iloc[-1] - norm.iloc[-1 - days_back]), 1)


# ── Abgeleitete Scores (reine Formeln, Inputs liefert app.py) ─────────────────

def _clip(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def injury_risk(fatigue, monotony, volume_jump, pain) -> float:
    """0.35·Fatigue + 0.25·Monotonie + 0.20·Volumensprung + 0.20·Schmerz."""
    return round(_clip(0.35 * fatigue + 0.25 * monotony
                       + 0.20 * volume_jump + 0.20 * pain), 1)


def confidence(motivation, consistency, recent_success) -> float:
    """0.4·Motivation + 0.3·Konstanz + 0.3·jüngste Erfolge (alle 0–100)."""
    return round(_clip(0.4 * motivation + 0.3 * consistency
                       + 0.3 * recent_success), 1)


def race_readiness(fitness, recovery, conf, fatigue, inj_risk) -> float:
    """
    0.40·Fitness + 0.20·Recovery + 0.15·Confidence − 0.15·Fatigue − 0.10·Injury.
    Hohe Fitness bei hoher Ermüdung wird abgewertet (outline.calc_race_readiness).
    """
    return round(_clip(0.40 * fitness + 0.20 * recovery + 0.15 * conf
                       - 0.15 * fatigue - 0.10 * inj_risk), 1)
