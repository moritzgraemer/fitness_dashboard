"""
Last pro Einheit – aerobe Last (L_aer) und mechanische Last (L_mech)
====================================================================

Ersetzt die einwertige TSS-Berechnung durch ZWEI getrennte Lastkanäle, die
unterschiedliche Gewebe belasten und unterschiedlich schnell adaptieren:

    L_aer   – kardiovaskuläre/metabolische Last  (Intensität²  × Zeit)
    L_mech  – mechanische Last                   (Schrittzahl × Aufprallfaktor)

Warum Lap-Auflösung (Jensen'sche Ungleichung)
---------------------------------------------
L_aer ist konvex in der Intensität (IF²). Für die Mittelwerte gilt daher

    mean(IF²) >= mean(IF)²

mit Gleichheit nur bei konstanter Intensität. Ein 60-min-Intervalltraining
mit Ø-HF 150 erzeugt real deutlich mehr Last als ein 60-min-Dauerlauf mit
Ø-HF 150 – die alte Rechnung über die Ø-Herzfrequenz der GESAMTEN Aktivität
verbucht beide identisch und unterschätzt strukturierte Einheiten systematisch.
Sobald Runden (Laps) vorliegen, wird L_aer deshalb pro Runde gebildet und
summiert. Bei genau einer Runde reduziert sich die Formel exakt auf die alte.

Formeln
-------
    IF_i    = HF_i / LTHR       (bzw. NP_i / FTP, Powermeter bevorzugt)
    L_aer   = Σ_i (t_i / 60) · IF_i² · 100          → 1 h an der Schwelle = 100

    n_i     = c_i · t_i                              (Schritte in der Runde)
    g_desc  = D_i / s_i ,  g_asc = A_i / s_i         (Gefälle / Steigung)
    φ_i     = max(φ_min, (v_i / v_ref)^β) · (1 + κ_desc·g_desc + κ_asc·g_asc)
    L_mech^run  = (1/1000) · Σ_i φ_i · n_i           (nur Laps mit v_i ≥ 1,5 km/h)
    L_mech^bike = λ · t_h · IF²                      (λ = 1,5 Road / 3,0 MTB)

    t_i [min] · c_i [Schritte/min] · s_i, D_i, A_i [m] · v_i [km/h]

Das Modul ist rein (keine Flask-/app.py-Importe, kein Datei-I/O). app.py
liefert die normalisierten Runden und die Schwellenwerte.
"""

import math

# ── Parameter L_mech (Lauf) ───────────────────────────────────────────────────
V_REF        = 10.0   # km/h – Referenztempo, bei dem φ = 1,0 (ohne Steigung)
PHI_MIN      = 0.4    # Untergrenze des Tempo-Faktors (Gehen/Traben)
BETA         = 1.0    # Exponent des Tempo-Faktors
KAPPA_DESC   = 6.0    # Gewicht Gefälle  – exzentrische Bremsarbeit, teuerster Anteil
KAPPA_ASC    = 3.0    # Gewicht Steigung – konzentrisch, geringerer Gewebeschaden
MIN_SPEED_KMH = 1.5   # darunter zählt eine Runde nicht (Pause, Ampel, Foto-Stopp)

# ── Parameter L_mech (Rad) ────────────────────────────────────────────────────
# Radfahren ist konzentrisch – kaum Gewebeschaden. λ skaliert die aerobe Last
# auf ein mechanisches Äquivalent; MTB doppelt so hoch wegen Stößen/Stabiarbeit.
LAMBDA_ROAD = 1.5
LAMBDA_MTB  = 3.0

# ── Fallback-Schrittfrequenz ──────────────────────────────────────────────────
# Lineare Regression über die eigenen 1337 Lauf-Runden aus
# GarminConnectData_Laps.csv (R = 0,69):  c ≈ 101,8 + 5,95 · v[km/h].
# Wird nur benutzt, wenn Garmin für eine Runde keine Kadenz geliefert hat; das
# Ergebnis wird dann als `cadence_estimated` markiert.
CADENCE_A     = 101.8
CADENCE_B     = 5.95
CADENCE_CLIP  = (120.0, 200.0)

RUN_SPORTS  = ('run', 'trailrun', 'virtualrun', 'treadmill_running',
               'track_running', 'trail_running', 'running')
BIKE_SPORTS = ('ride', 'virtualride', 'ebikeride', 'gravelride',
               'mountainbikeride', 'cycling', 'indoor_cycling', 'mountain_biking')
MTB_SPORTS  = ('mountainbikeride', 'mountain_biking', 'gravelride')


def _f(val, default=0.0):
    """Robuster float-Cast: None / '' / NaN / 'nan' → default."""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else x


def estimate_cadence(speed_kmh: float) -> float:
    """Schrittfrequenz (spm, beidbeinig) aus dem Tempo schätzen – s. CADENCE_A/B."""
    c = CADENCE_A + CADENCE_B * speed_kmh
    return max(CADENCE_CLIP[0], min(CADENCE_CLIP[1], c))


def is_run(sport) -> bool:
    return str(sport or '').strip().lower() in RUN_SPORTS


def is_bike(sport) -> bool:
    return str(sport or '').strip().lower() in BIKE_SPORTS


def lambda_for(sport) -> float:
    """λ für L_mech^bike – MTB/Gravel rauer als Straße."""
    return LAMBDA_MTB if str(sport or '').strip().lower() in MTB_SPORTS else LAMBDA_ROAD


# ═════════════════════════════════════════════════════════════════════════════
# Runden-Normalisierung
# ═════════════════════════════════════════════════════════════════════════════

def normalize_lap(raw: dict) -> dict:
    """Eine Roh-Runde (Laps-CSV oder synthetische Ganz-Aktivitäts-Runde) auf das
    interne Schema bringen. Fehlende Geschwindigkeit wird aus Distanz/Dauer
    rekonstruiert, fehlende Kadenz aus dem Tempo geschätzt."""
    dur_s = _f(raw.get('duration_s'))
    dist  = _f(raw.get('distance_m'))

    v_ms = _f(raw.get('avg_speed_ms'))
    if v_ms <= 0 and dur_s > 0:
        v_ms = dist / dur_s
    v_kmh = v_ms * 3.6

    cad = _f(raw.get('cadence_spm'))
    cad_est = False
    if cad <= 0:
        cad = estimate_cadence(v_kmh) if v_kmh > 0 else 0.0
        cad_est = cad > 0

    return {
        'lap_number':   int(_f(raw.get('lap_number'), 0)),
        'duration_s':   dur_s,
        'duration_min': dur_s / 60.0,
        'distance_m':   dist,
        'speed_kmh':    v_kmh,
        'avg_hr':       _f(raw.get('avg_hr')),
        'avg_power_w':  _f(raw.get('avg_power_w')),
        'ascent_m':     _f(raw.get('ascent_m')),
        'descent_m':    _f(raw.get('descent_m')),
        'cadence_spm':  cad,
        'cadence_estimated': cad_est,
    }


# ═════════════════════════════════════════════════════════════════════════════
# L_aer – aerobe Last
# ═════════════════════════════════════════════════════════════════════════════

def lap_intensity(lap: dict, lthr: float, ftp: float = 0.0) -> tuple:
    """IF einer Runde. Powermeter wird bevorzugt (misst die Arbeit direkt,
    HF hinkt bei Intervallen träge hinterher) – aber nur, wenn für die
    Modalität überhaupt ein Schwellenwert konfiguriert ist.

    Rückgabe: (IF, quelle) mit quelle ∈ {'power', 'hr', None}."""
    p = lap.get('avg_power_w', 0.0)
    if ftp and ftp > 0 and p > 0:
        return p / ftp, 'power'
    hr = lap.get('avg_hr', 0.0)
    if lthr and lthr > 0 and hr > 0:
        return hr / lthr, 'hr'
    return 0.0, None


def aerobic_load(laps: list, lthr: float, ftp: float = 0.0) -> dict:
    """L_aer = Σ_i (t_i/60) · IF_i² · 100.

    Rückgabe: {'l_aer', 'laps' (Detail je Runde), 'if_mean', 'if_quad_mean',
               'jensen_gain' (Aufschlag ggü. Rechnung mit Ø-Intensität),
               'source', 'covered_min'}.
    `jensen_gain` macht sichtbar, was die Lap-Auflösung überhaupt bringt:
    0 % bei konstanter Intensität, deutlich >0 % bei Intervallen.
    """
    detail, total = [], 0.0
    sum_t = sum_t_if = sum_t_if2 = 0.0
    sources = set()

    for lap in laps:
        t_min = lap['duration_min']
        if t_min <= 0:
            continue
        IF, src = lap_intensity(lap, lthr, ftp)
        if src is None or IF <= 0:
            detail.append({**{k: lap[k] for k in ('lap_number', 'duration_min')},
                           'if': None, 'l_aer': 0.0, 'source': None})
            continue
        contrib = (t_min / 60.0) * IF ** 2 * 100.0
        total  += contrib
        sum_t     += t_min
        sum_t_if  += t_min * IF
        sum_t_if2 += t_min * IF ** 2
        sources.add(src)
        detail.append({'lap_number': lap['lap_number'],
                       'duration_min': round(t_min, 2),
                       'if': round(IF, 3),
                       'l_aer': round(contrib, 1),
                       'source': src})

    if_mean  = (sum_t_if / sum_t) if sum_t > 0 else 0.0
    if_quad  = (sum_t_if2 / sum_t) if sum_t > 0 else 0.0
    # Was eine Rechnung auf Ø-Intensität (= alte Logik) ergeben hätte:
    flat = (sum_t / 60.0) * if_mean ** 2 * 100.0
    gain = ((total / flat - 1.0) * 100.0) if flat > 0 else 0.0

    return {
        'l_aer':        round(total, 1),
        'laps':         detail,
        'if_mean':      round(if_mean, 3),
        'if_quad_mean': round(if_quad, 3),
        'jensen_gain':  round(gain, 1),      # % Mehrlast durch Lap-Auflösung
        'source':       'power' if sources == {'power'} else
                        ('mixed' if len(sources) > 1 else ('hr' if sources else None)),
        'covered_min':  round(sum_t, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# L_mech – mechanische Last
# ═════════════════════════════════════════════════════════════════════════════

def lap_phi(lap: dict) -> dict:
    """Aufprallfaktor φ einer Lauf-Runde inkl. seiner beiden Anteile."""
    v = lap['speed_kmh']
    s = lap['distance_m']
    pace_factor = max(PHI_MIN, (v / V_REF) ** BETA) if v > 0 else PHI_MIN
    g_desc = (lap['descent_m'] / s) if s > 0 else 0.0
    g_asc  = (lap['ascent_m']  / s) if s > 0 else 0.0
    grade_factor = 1.0 + KAPPA_DESC * g_desc + KAPPA_ASC * g_asc
    return {
        'pace_factor':  pace_factor,
        'g_desc':       g_desc,
        'g_asc':        g_asc,
        'grade_factor': grade_factor,
        'phi':          pace_factor * grade_factor,
    }


def mech_load_run(laps: list) -> dict:
    """L_mech^run = (1/1000) · Σ φ_i · n_i über alle Runden mit v ≥ 1,5 km/h."""
    detail, total, steps = [], 0.0, 0.0
    estimated_min = 0.0
    for lap in laps:
        t_min = lap['duration_min']
        if t_min <= 0:
            continue
        if lap['speed_kmh'] < MIN_SPEED_KMH:
            detail.append({'lap_number': lap['lap_number'], 'skipped': 'v < 1,5 km/h'})
            continue
        p = lap_phi(lap)
        n = lap['cadence_spm'] * t_min          # Schritte in dieser Runde
        contrib = p['phi'] * n / 1000.0
        total += contrib
        steps += n
        if lap['cadence_estimated']:
            estimated_min += t_min
        detail.append({
            'lap_number':   lap['lap_number'],
            'duration_min': round(t_min, 2),
            'speed_kmh':    round(lap['speed_kmh'], 2),
            'cadence_spm':  round(lap['cadence_spm'], 1),
            'cadence_estimated': lap['cadence_estimated'],
            'steps':        round(n),
            'pace_factor':  round(p['pace_factor'], 3),
            'grade_factor': round(p['grade_factor'], 3),
            'phi':          round(p['phi'], 3),
            'l_mech':       round(contrib, 2),
        })
    return {
        'l_mech':  round(total, 2),
        'steps':   round(steps),
        'laps':    detail,
        'cadence_estimated_min': round(estimated_min, 1),
    }


def mech_load_bike(duration_h: float, intensity_factor: float, lam: float) -> dict:
    """L_mech^bike = λ · t_h · IF². Konzentrisch, daher pauschal statt pro Schritt."""
    val = lam * max(0.0, duration_h) * max(0.0, intensity_factor) ** 2
    return {'l_mech': round(val, 2), 'lambda': lam,
            'duration_h': round(duration_h, 3),
            'if': round(intensity_factor, 3)}


# ═════════════════════════════════════════════════════════════════════════════
# Gesamteinstieg
# ═════════════════════════════════════════════════════════════════════════════

def unit_load(raw_laps: list, sport: str, lthr: float,
              ftp: float = 0.0, run_ftp: float = 0.0) -> dict:
    """L_aer und L_mech einer Einheit aus ihren (Roh-)Runden.

    raw_laps: Liste mit duration_s, distance_m, avg_speed_ms, avg_hr,
              avg_power_w, ascent_m, descent_m, cadence_spm (fehlende Felder
              werden toleriert). Eine einzige Runde = ganze Aktivität.
    sport:    Aktivitätstyp (Strava- oder Garmin-Schreibweise).
    """
    laps = [normalize_lap(l) for l in (raw_laps or [])]
    run, bike = is_run(sport), is_bike(sport)

    # Powerschwelle je Modalität: Rad-FTP nur fürs Rad, Laufleistung (Stryd)
    # nur, wenn dafür eine eigene Schwelle konfiguriert ist – sonst wäre
    # Laufwatt / Rad-FTP eine Scheingenauigkeit.
    threshold_power = ftp if bike else (run_ftp if run else 0.0)

    aer = aerobic_load(laps, lthr, threshold_power)

    if run:
        mech = mech_load_run(laps)
        mech['model'] = 'run'
    elif bike:
        dur_h = sum(l['duration_min'] for l in laps) / 60.0
        mech  = mech_load_bike(dur_h, aer['if_mean'], lambda_for(sport))
        mech['model'] = 'bike'
    else:
        # Schwimmen, Wandern, Kraft … – kein validiertes mechanisches Modell.
        mech = {'l_mech': 0.0, 'model': 'none'}

    return {
        'l_aer':  aer['l_aer'],
        'l_mech': mech['l_mech'],
        'aer':    aer,
        'mech':   mech,
        'n_laps': len(laps),
    }
