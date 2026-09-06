#!/usr/bin/env python3
"""
Dashboard v2 – Server-Wrapper
=============================

Serviert das überarbeitete Frontend (frontend2/dashboard_v2.html) und die
Mobilansicht (frontend2/mobile.html) und nutzt dabei app.py UNVERÄNDERT als
Rechenkern: alle /api/*-Routen, die L_aer/L_mech-Berechnung, CTL/ATL/TSB,
Readiness und die Kaskade sind exakt dieselben wie in v1. Es gibt hier keine
zweite Logik – nur eine zweite Oberfläche.

    python3 serve_v2.py            # → http://localhost:5002
    V2_PORT=5005 python3 serve_v2.py
    V2_NO_BROWSER=1 python3 serve_v2.py

Umgebungsvariablen (für den Betrieb auf einem Server, s. README-DEPLOY.md):
    V2_HOST        Bind-Adresse (Standard 127.0.0.1; im Container 0.0.0.0)
    V2_PORT        Port (Standard 5002; im Container 8080)
    V2_DEBUG       1 = Flask-Debug mit Auto-Reload (lokal), 0 = aus (Server)
    DASH_PASSWORD  Wenn gesetzt, ist ALLES passwortgeschützt: Login-Formular
                   unter /login (Cookie, 90 Tage) oder HTTP-Basic-Auth.
    DASH_SECRET    Optionaler eigener Cookie-Schlüssel (sonst aus dem Passwort
                   abgeleitet – ein neues Passwort loggt alle Geräte aus).

Routen dieser Datei:
    /            Desktop-Oberfläche; Mobilgeräte werden nach /m umgeleitet
                 (?desktop=1 erzwingt Desktop und merkt sich das per Cookie)
    /m           Mobilansicht (iPhone, „auf einen Blick“)
    /login, /logout, /healthz

v1 läuft parallel und unbeeinflusst über app.py auf Port 5001.
"""

import hashlib
import hmac
import os
import re
import sys
import threading
import webbrowser
from urllib.parse import quote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import app as core          # registriert alle bestehenden Routen auf core.app

from flask import (Response, jsonify, make_response, redirect, request,
                   send_from_directory)
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

flask_app     = core.app
FRONTEND2_DIR = os.path.join(BASE_DIR, 'frontend2')
HTML_FILE     = 'dashboard_v2.html'
MOBILE_FILE   = 'mobile.html'

# v1 beendet den Server, sobald der Browser-Tab geschlossen wird (client_gone →
# os._exit). Für v2 soll der Prozess laufen, bis er bewusst gestoppt wird –
# sonst killt ein versehentlich geschlossener Tab den Server mitten im Sync.
core._arm_shutdown = lambda seconds: None


# ─────────────────────────────────────────────────────────────────────────────
# Passwortschutz (nur aktiv, wenn DASH_PASSWORD gesetzt ist)
# ─────────────────────────────────────────────────────────────────────────────

DASH_PASSWORD = os.environ.get('DASH_PASSWORD', '').strip()
_SECRET = os.environ.get('DASH_SECRET') or \
    hashlib.sha256(('fitness-dashboard:' + DASH_PASSWORD).encode('utf-8')).hexdigest()
flask_app.secret_key = _SECRET
_serializer   = URLSafeTimedSerializer(_SECRET, salt='dash-login')
SESSION_COOKIE = 'dash_session'
SESSION_MAX_AGE = 90 * 24 * 3600
VIEW_COOKIE    = 'dash_view'
PUBLIC_PATHS   = ('/login', '/logout', '/healthz', '/favicon.png')


def _password_ok(candidate) -> bool:
    return bool(DASH_PASSWORD) and hmac.compare_digest(str(candidate or ''), DASH_PASSWORD)


def _is_https() -> bool:
    return request.is_secure or request.headers.get('X-Forwarded-Proto', '') == 'https'


def _authenticated() -> bool:
    if not DASH_PASSWORD:
        return True
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        try:
            if _serializer.loads(token, max_age=SESSION_MAX_AGE) == 'ok':
                return True
        except (BadSignature, SignatureExpired):
            pass
    auth = request.authorization
    if auth and _password_ok(auth.password):
        return True
    return False


@flask_app.before_request
def _gate():
    if request.path in PUBLIC_PATHS or request.path.startswith('/static/'):
        return None
    if _authenticated():
        return None
    if request.path.startswith('/api/'):
        return jsonify({'error': 'auth', 'message': 'Anmeldung erforderlich (/login).'}), 401
    return redirect('/login?next=' + quote(request.full_path.rstrip('?') or '/'))


_LOGIN_HTML = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#141414"><title>Training · Anmelden</title>
<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#141414;--surface:#1B1B1B;--border:#F2EFE7;--accent:#4E7FD1;--text:#F2EFE7;--muted:#9C9C90;--red:#E2543C}
@media(prefers-color-scheme:light){:root{--bg:#EEF0E8;--surface:#F6F7F1;--border:#141414;--accent:#1E4FA0;--text:#141414;--muted:#5B5F55;--red:#A6431E}}
*{box-sizing:border-box;margin:0;padding:0}
body{min-height:100vh;display:flex;align-items:center;justify-content:center;background:var(--bg);color:var(--text);
     font-family:'Space Grotesk',system-ui,sans-serif;padding:24px env(safe-area-inset-right) 24px env(safe-area-inset-left)}
form{width:100%;max-width:360px;background:var(--surface);border:2px solid var(--border);padding:28px 24px}
h1{font-family:'Archivo Black',sans-serif;font-size:26px;margin-bottom:4px}h1 span{color:var(--accent)}
p{color:var(--muted);font-size:13px;margin-bottom:20px}
label{display:block;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
input{width:100%;font:inherit;font-size:17px;padding:12px;background:var(--bg);color:var(--text);border:1px solid var(--border);outline:none}
input:focus{box-shadow:inset 0 0 0 2px var(--accent)}
button{width:100%;margin-top:14px;padding:13px;font:inherit;font-weight:700;font-size:15px;background:var(--accent);color:#fff;border:0;cursor:pointer}
.err{color:var(--red);font-size:13px;margin-top:10px}
</style></head><body>
<form method="post" action="/login">
  <h1>Training<span>.</span></h1>
  <p>Fitness-Dashboard – bitte anmelden.</p>
  <label for="pw">Passwort</label>
  <input id="pw" name="password" type="password" autocomplete="current-password" autofocus required>
  <input type="hidden" name="next" value="{next}">
  <button type="submit">Anmelden</button>
  {error}
</form></body></html>"""


def _safe_next(value: str) -> str:
    v = str(value or '/')
    return v if (v.startswith('/') and not v.startswith('//')) else '/'


@flask_app.route('/login', methods=['GET', 'POST'])
def dash_login():
    if not DASH_PASSWORD:
        return redirect('/')
    nxt = _safe_next(request.values.get('next', '/'))
    if request.method == 'POST':
        if _password_ok(request.form.get('password')):
            resp = make_response(redirect(nxt))
            resp.set_cookie(SESSION_COOKIE, _serializer.dumps('ok'),
                            max_age=SESSION_MAX_AGE, httponly=True,
                            secure=_is_https(), samesite='Lax')
            return resp
        html = _LOGIN_HTML.replace('{next}', nxt).replace(
            '{error}', '<div class="err">Falsches Passwort.</div>')
        return Response(html, status=401, mimetype='text/html')
    if _authenticated():
        return redirect(nxt)
    return Response(_LOGIN_HTML.replace('{next}', nxt).replace('{error}', ''),
                    mimetype='text/html')


@flask_app.route('/logout')
def dash_logout():
    resp = make_response(redirect('/login'))
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@flask_app.route('/healthz')
def dash_healthz():
    return 'ok'


# ─────────────────────────────────────────────────────────────────────────────
# Oberflächen: Desktop (/) und Mobil (/m)
# ─────────────────────────────────────────────────────────────────────────────

_MOBILE_UA = re.compile(r'iPhone|iPod|Android.*Mobile|Windows Phone', re.I)


def _no_store(resp):
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp


def _v2_index():
    """Ersetzt die '/'-Ansicht: Desktop-Frontend, Mobilgeräte → /m."""
    if request.args.get('desktop'):
        resp = _no_store(send_from_directory(FRONTEND2_DIR, HTML_FILE))
        resp.set_cookie(VIEW_COOKIE, 'desktop', max_age=30 * 24 * 3600, samesite='Lax')
        return resp
    if request.cookies.get(VIEW_COOKIE) != 'desktop' and \
            _MOBILE_UA.search(request.headers.get('User-Agent', '')):
        return redirect('/m')
    return _no_store(send_from_directory(FRONTEND2_DIR, HTML_FILE))


@flask_app.route('/m')
def mobile_index():
    resp = _no_store(send_from_directory(FRONTEND2_DIR, MOBILE_FILE))
    if request.args.get('reset'):
        resp.delete_cookie(VIEW_COOKIE)
    return resp


# Die Route '/' ist in app.py bereits an die View-Funktion 'index' gebunden.
# Statt eine zweite Route zu registrieren (was Flask ablehnen würde), wird die
# hinterlegte Funktion ausgetauscht – v1 bleibt in seinem eigenen Prozess davon
# unberührt.
flask_app.view_functions['index'] = _v2_index


if __name__ == '__main__':
    port  = int(os.environ.get('V2_PORT', 5002))
    host  = os.environ.get('V2_HOST', '127.0.0.1')
    debug = os.environ.get('V2_DEBUG', '1') not in ('0', 'false', 'no', '')
    print('═' * 52)
    print('  Fitness Dashboard v2')
    print(f'  → http://{"localhost" if host in ("127.0.0.1", "0.0.0.0") else host}:{port}   (Mobil: /m)')
    print('  Rechenkern: app.py (identisch zu v1 auf Port 5001)')
    print(f'  Passwortschutz: {"aktiv" if DASH_PASSWORD else "aus (DASH_PASSWORD nicht gesetzt)"}')
    print('═' * 52)

    for fn in (HTML_FILE, MOBILE_FILE):
        if not os.path.exists(os.path.join(FRONTEND2_DIR, fn)):
            print(f'\n⚠  {fn} fehlt in {FRONTEND2_DIR}\n')

    if not os.environ.get('WERKZEUG_RUN_MAIN') and not os.environ.get('V2_NO_BROWSER'):
        threading.Timer(1.2, lambda: webbrowser.open(f'http://localhost:{port}')).start()

    flask_app.run(host=host, debug=debug, port=port, threaded=True)
