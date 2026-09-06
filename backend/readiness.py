"""
Zustand & Kaskade – Readiness-Score und Anpassung der Tagesempfehlung
=====================================================================

Teil 2 und 3 des Lastmodells (Teil 1 = unit_load.py).

ZUSTAND
-------
    CTL_t = CTL_{t-1} + α_c·(L_t − CTL_{t-1})        α_c = 1/42
    ATL_t = ATL_{t-1} + α_a·(L_t − ATL_{t-1})        α_a = 1/7
    TSB_t = CTL_{t-1} − ATL_{t-1}     ← bewusst die Werte von GESTERN:
            die Form des Tages ist das Ergebnis des bisherigen Trainings,
            nicht der Einheit, die heute erst noch kommt.

    Mech-Last_k = rollender 28-Tage-Mittelwert der täglichen L_mech,
                  auf eine Wochenmenge hochgerechnet (×7).

    z_x = (x − μ_60(x)) / σ_60(x)

    Readiness:
        HRV → ln des rollenden 7-Tage-Mittels, dann z gegen 60-Tage-Historie
        RHR → z gegen 60-Tage-Historie
        c̃  = (c − 3)/2  – Check-in-Item (1–5) auf [−1, 1]

        R = 0,35·Ẽ + 0,25·M̃ − 0,20·Q̃ + 0,15·z_HRV − 0,05·z_RHR

    Belegung der Check-in-Items (morning_checkins.csv):
        Ẽ ← readiness        (Energie)
        M̃ ← mental_clarity   (Motivation)
        Q̃ ← 6 − sleep_quality (schlechter Schlaf als Belastungssignal)

KASKADE
-------
    1. Verletzt + Schweregrad ≥ 3 → Modalität sperren, Alternative / Rad
    2. Mech-Last_k + L̂_mech > Mech-Last_{k-1}·(1+r_mech) → Lauf → Rad
    3. TSB < −0,35·CTL → Intensität auf Z2, Dauer bleibt
    4. niedrige HRV / Motivation / Schlaf oder hohe Ermüdung → r_aer = r_mech = 0
    5. sonst → f = clip(1 + 0,25·R, 0,5, 1,1), Dauer × f

    Steigerung pro Woche:
        r_aer  = 0,15 (Schlafwoche) sonst 0,10
        r_mech = 0,10 (Kraftwoche)  sonst 0,05

Reines Rechenmodul – kein Flask, kein Datei-I/O.
"""

import math

import numpy as np
import pandas as pd

# ── Zeitkonstanten ────────────────────────────────────────────────────────────
ALPHA_CTL = 1.0 / 42
ALPHA_ATL = 1.0 / 7
MECH_WINDOW_DAYS = 28      # Fenster des rollenden Mech-Last-Mittels
Z_WINDOW_DAYS    = 60      # Referenzfenster der z-Scores
HRV_SMOOTH_DAYS  = 7       # Glättung der HRV vor der z-Bildung

# ── Readiness-Gewichte ────────────────────────────────────────────────────────
W_ENERGY, W_MOTIVATION, W_QUALITY = 0.35, 0.25, -0.20
W_HRV, W_RHR = 0.15, -0.05

# ── Wochensteigerung ──────────────────────────────────────────────────────────
R_AER_SLEEP_WEEK,  R_AER_DEFAULT  = 0.15, 0.10
R_MECH_STRENGTH_WEEK, R_MECH_DEFAULT = 0.10, 0.05

# ── Schwellen der Kaskade ─────────────────────────────────────────────────────
INJURY_BLOCK_SEVERITY = 3      # ab hier Modalität sperren
TSB_CTL_FACTOR        = -0.35  # TSB < −0,35·CTL  → Intensität kappen
GATE_HRV_Z            = -1.0   # z_HRV darunter = "low HRV"
GATE_CHECKIN          = 2      # Slider-Wert ≤ 2 (von 5) = "low"
GATE_LOAD_RATIO       = 1.3    # ATL/CTL darüber = "high Fatigue"
SLEEP_WEEK_MIN        = 4.0    # Ø sleep_quality ≥ 4 → "Schlafwoche"
DURATION_F_MIN, DURATION_F_MAX = 0.5, 1.1
DURATION_F_SLOPE = 0.25


# ═════════════════════════════════════════════════════════════════════════════
# Bausteine
# ═════════════════════════════════════════════════════════════════════════════

def ema_state(daily_load, alpha_ctl=ALPHA_CTL, alpha_atl=ALPHA_ATL) -> pd.DataFrame:
    """CTL/ATL/TSB aus einer lückenlosen Tagesreihe der Last L_t.

    TSB_t = CTL_{t-1} − ATL_{t-1} (Vortageswerte, s. Modul-Docstring).
    """
    ctl = atl = 0.0
    ctls, atls, tsbs = [], [], []
    for L in daily_load:
        tsbs.append(ctl - atl)                 # noch die Werte von gestern
        ctl += alpha_ctl * (float(L) - ctl)
        atl += alpha_atl * (float(L) - atl)
        ctls.append(ctl)
        atls.append(atl)
    return pd.DataFrame({'ctl': np.round(ctls, 1),
                         'atl': np.round(atls, 1),
                         'tsb': np.round(tsbs, 1)})


def mech_weekly(daily_mech, window=MECH_WINDOW_DAYS) -> pd.Series:
    """Mech-Last als Wochenmenge: rollender `window`-Tage-Mittelwert × 7.

    Der 28-Tage-Mittelwert glättet einzelne lange Läufe weg, sodass der
    Wochenvergleich in Regel 2 nicht an einem einzigen Ausreißer hängt.
    """
    s = pd.Series(list(daily_mech), dtype='float64').fillna(0.0)
    return (s.rolling(window, min_periods=1).mean() * 7).round(2)


def rolling_z(series, window=Z_WINDOW_DAYS) -> pd.Series:
    """z_x = (x − μ_window(x)) / σ_window(x), NaN-tolerant.

    Das Fenster endet auf dem jeweiligen Tag (inklusive), σ = 0 → z = 0.
    """
    s = pd.Series(list(series), dtype='float64')
    mu = s.rolling(window, min_periods=10).mean()
    sd = s.rolling(window, min_periods=10).std()
    z  = (s - mu) / sd.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def hrv_z(hrv_series, smooth=HRV_SMOOTH_DAYS, window=Z_WINDOW_DAYS) -> pd.Series:
    """HRV: ln des rollenden 7-Tage-Mittels, dann z gegen 60 Tage.

    Der Logarithmus, weil HRV (RMSSD) rechtsschief verteilt ist – ohne ihn
    ziehen einzelne hohe Nächte den Mittelwert und damit die z-Skala hoch.
    """
    s = pd.Series(list(hrv_series), dtype='float64')
    smoothed = s.rolling(smooth, min_periods=1).mean()
    ln = np.log(smoothed.where(smoothed > 0))
    return rolling_z(ln, window)


def scale_checkin(value, default=3.0) -> float:
    """c̃ = (c − 3)/2 → Slider 1–5 auf [−1, 1]. Fehlend = neutral (0)."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        c = default
    if math.isnan(c):
        c = default
    return max(-1.0, min(1.0, (c - 3.0) / 2.0))


def readiness(energy=None, motivation=None, sleep_quality=None,
              z_hrv=0.0, z_rhr=0.0) -> dict:
    """R = 0,35·Ẽ + 0,25·M̃ − 0,20·Q̃ + 0,15·z_HRV − 0,05·z_RHR.

    Q wird aus sleep_quality gebildet: Q = 6 − sleep_quality (guter Schlaf → Q
    klein → der negative Term zieht kaum). Rückgabe enthält alle Terme, damit
    die Empfehlung begründbar bleibt.
    """
    e = scale_checkin(energy)
    m = scale_checkin(motivation)
    q_raw = (6.0 - float(sleep_quality)) if _isnum(sleep_quality) else 3.0
    q = scale_checkin(q_raw)

    zh = _num(z_hrv)
    zr = _num(z_rhr)

    terms = {
        'energy':     W_ENERGY * e,
        'motivation': W_MOTIVATION * m,
        'quality':    W_QUALITY * q,
        'hrv':        W_HRV * zh,
        'rhr':        W_RHR * zr,
    }
    R = sum(terms.values())
    return {
        'R': round(R, 3),
        'terms': {k: round(v, 3) for k, v in terms.items()},
        'scaled': {'energy': round(e, 3), 'motivation': round(m, 3),
                   'quality': round(q, 3), 'z_hrv': round(zh, 3), 'z_rhr': round(zr, 3)},
    }


def growth_rates(sleep_week: bool, strength_week: bool) -> dict:
    """Erlaubte Wochensteigerung. Guter Schlaf erlaubt mehr aerobe Steigerung,
    begleitendes Krafttraining mehr mechanische (Sehnen/Knochen toleranter)."""
    return {
        'r_aer':  R_AER_SLEEP_WEEK if sleep_week else R_AER_DEFAULT,
        'r_mech': R_MECH_STRENGTH_WEEK if strength_week else R_MECH_DEFAULT,
        'sleep_week': bool(sleep_week),
        'strength_week': bool(strength_week),
    }


def duration_factor(R: float) -> float:
    """f = clip(1 + 0,25·R, 0,5, 1,1) – asymmetrisch: schlechte Tage dürfen die
    Einheit halbieren, gute sie höchstens um 10 % verlängern."""
    return round(max(DURATION_F_MIN,
                     min(DURATION_F_MAX, 1.0 + DURATION_F_SLOPE * _num(R))), 3)


# ═════════════════════════════════════════════════════════════════════════════
# Kaskade
# ═════════════════════════════════════════════════════════════════════════════

def cascade(*, planned: dict, R: float, ctl: float, tsb: float,
            mech_week: float, mech_week_prev: float, mech_forecast: float,
            load_ratio: float = 0.0, z_hrv: float = 0.0,
            motivation=None, sleep_quality=None,
            injuries: list = None, rates: dict = None) -> dict:
    """Passt die geplante Einheit an den Tageszustand an.

    planned: {'type', 'zone', 'duration_min', 'workout', ...} – die rohe
             Empfehlung (Algorithmus oder LLM), die angepasst werden soll.
    mech_forecast: L̂_mech der geplanten Einheit (Prognose, s. forecast_mech()).

    Die fünf Regeln werden in fester Reihenfolge ausgewertet und wirken
    kumulativ; `rules` protokolliert jede, die gegriffen hat.
    """
    planned = dict(planned or {})
    rules, notes = [], []
    out = {
        'type':         planned.get('type', 'Run'),
        'zone':         planned.get('zone'),
        'duration_min': _num(planned.get('duration_min'), 0.0),
    }
    original = dict(out)

    # ── Regel 4 zuerst: das Gate bestimmt r_mech, das Regel 2 gleich braucht ──
    gate = _gate_flags(z_hrv, motivation, sleep_quality, load_ratio)
    rates = dict(rates or growth_rates(False, False))
    if gate['any']:
        rates = {**rates, 'r_aer': 0.0, 'r_mech': 0.0}
        rules.append({'rule': 'gate', 'action': 'r_aer = r_mech = 0',
                      'reason': gate['reasons']})

    # ── Regel 1: Verletzung ───────────────────────────────────────────────────
    blocking = [i for i in (injuries or [])
                if _num(i.get('severity')) >= INJURY_BLOCK_SEVERITY]
    if blocking and _is_impact(out['type']):
        out['type'] = 'Ride'
        rules.append({
            'rule': 'injury',
            'action': f"Modalität gesperrt → {out['type']}",
            'reason': [f"{i.get('body_part', '?')} (Schweregrad {int(_num(i.get('severity')))})"
                       for i in blocking],
        })
        notes.append('Laufbelastung wegen aktiver Verletzung ausgesetzt.')

    # ── Regel 2: mechanisches Wochenbudget ────────────────────────────────────
    budget = mech_week_prev * (1.0 + rates['r_mech'])
    projected = mech_week + _num(mech_forecast)
    mech_over = mech_week_prev > 0 and projected > budget
    if mech_over and _is_impact(out['type']):
        out['type'] = 'Ride'
        rules.append({
            'rule': 'mech_budget',
            'action': 'Lauf → Rad',
            'reason': [f"Mech-Last {mech_week:.1f} + Prognose {_num(mech_forecast):.1f} "
                       f"= {projected:.1f} > Budget {budget:.1f} "
                       f"(Vorwoche {mech_week_prev:.1f} × {1 + rates['r_mech']:.2f})"],
        })

    # ── Regel 3: Form zu tief → Intensität kappen ─────────────────────────────
    tsb_limit = TSB_CTL_FACTOR * ctl
    if ctl > 0 and tsb < tsb_limit:
        if _num(out['zone'], 0) > 2:
            out['zone'] = 2
            rules.append({'rule': 'tsb', 'action': 'Intensität → Z2 (Dauer bleibt)',
                          'reason': [f"TSB {tsb:+.1f} < {tsb_limit:+.1f} (−0,35 × CTL {ctl:.1f})"]})
            notes.append('Umfang bleibt, nur die Intensität wird zurückgenommen.')
        else:
            rules.append({'rule': 'tsb', 'action': 'bereits Z2 – keine Änderung',
                          'reason': [f"TSB {tsb:+.1f} < {tsb_limit:+.1f}"]})

    # ── Regel 5: Dauer über Readiness skalieren ───────────────────────────────
    f = duration_factor(R)
    if out['duration_min'] > 0:
        out['duration_min'] = int(round(out['duration_min'] * f))
        rules.append({'rule': 'duration', 'action': f'Dauer × {f:.2f}',
                      'reason': [f"R = {_num(R):+.2f} → f = clip(1 + 0,25·R; 0,5; 1,1)"]})

    return {
        'original':      original,
        'adjusted':      out,
        'rules':         rules,
        'rates':         rates,
        'gate':          gate,
        'duration_factor': f,
        'mech_budget':   {'week': round(mech_week, 2),
                          'week_prev': round(mech_week_prev, 2),
                          'forecast': round(_num(mech_forecast), 2),
                          'budget': round(budget, 2),
                          'exceeded': bool(mech_over)},
        'tsb_limit':     round(tsb_limit, 1),
        'notes':         notes,
    }


def forecast_mech(duration_min: float, sport: str, speed_kmh: float = None,
                  ascent_m: float = 0.0, descent_m: float = 0.0,
                  intensity_factor: float = 0.75) -> float:
    """L̂_mech einer noch nicht gelaufenen Einheit.

    Für Läufe wird die Einheit als eine einzige Runde mit konstantem Tempo
    modelliert (Kadenz aus dem Tempo geschätzt), für Räder direkt über λ·t·IF².
    Bewusst grob – die Prognose entscheidet nur, ob das Wochenbudget reißt.
    """
    import unit_load as ul   # flach, wie training_model – backend/ liegt auf sys.path

    dur_min = max(0.0, _num(duration_min))
    if dur_min <= 0:
        return 0.0
    if ul.is_bike(sport):
        return ul.mech_load_bike(dur_min / 60.0, intensity_factor,
                                 ul.lambda_for(sport))['l_mech']
    if not ul.is_run(sport):
        return 0.0
    v = _num(speed_kmh) or 10.0
    lap = ul.normalize_lap({
        'duration_s': dur_min * 60.0,
        'distance_m': v * 1000.0 * (dur_min / 60.0),
        'avg_speed_ms': v / 3.6,
        'ascent_m': ascent_m, 'descent_m': descent_m,
    })
    return ul.mech_load_run([lap])['l_mech']


# ═════════════════════════════════════════════════════════════════════════════
# Hilfen
# ═════════════════════════════════════════════════════════════════════════════

IMPACT_TYPES = {'run', 'trailrun', 'virtualrun', 'running', 'trail_running'}


def _is_impact(t) -> bool:
    """Modalität mit Bodenkontakt-Belastung – nur die wird auf Rad umgelenkt."""
    return str(t or '').strip().lower() in IMPACT_TYPES


def _isnum(v) -> bool:
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _num(v, default=0.0) -> float:
    return float(v) if _isnum(v) else default


def _gate_flags(z_hrv, motivation, sleep_quality, load_ratio) -> dict:
    """Regel 4: Bei einem dieser Signale wird die Wochensteigerung ausgesetzt."""
    reasons = []
    if _num(z_hrv) < GATE_HRV_Z:
        reasons.append(f"HRV niedrig (z = {_num(z_hrv):+.2f})")
    if _isnum(motivation) and float(motivation) <= GATE_CHECKIN:
        reasons.append(f"Motivation niedrig ({int(float(motivation))}/5)")
    if _isnum(sleep_quality) and float(sleep_quality) <= GATE_CHECKIN:
        reasons.append(f"Schlaf schlecht ({int(float(sleep_quality))}/5)")
    if _num(load_ratio) > GATE_LOAD_RATIO:
        reasons.append(f"Ermüdung hoch (ATL/CTL = {_num(load_ratio):.2f})")
    return {'any': bool(reasons), 'reasons': reasons}
