"""Google Calendar integration: adds a run event when a workout is pushed to Garmin."""
import os
import json
import datetime as dt

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Zentrale config.json im Projekt-Wurzelverzeichnis (NICHT in Git).
CONFIG_PATH = os.path.join(os.path.dirname(BACKEND_DIR), 'config.json')
SCOPES      = ['https://www.googleapis.com/auth/calendar.events']

COLOR_ORANGE = '6'  # Google Calendar colorId "Tangerine"
EVENT_HOUR   = 9
EVENT_DURATION_MIN = 60


def _load_config():
    """Wie in app.py: erst die Umgebung (Server-Secret), dann config.json."""
    raw = os.environ.get('APP_CONFIG_JSON', '').strip()
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_token(token_json_str):
    """Aktualisiertes Google-Token zurück in config.json schreiben (Rest bleibt erhalten)."""
    cfg = _load_config()
    cfg.setdefault('google', {})['token'] = json.loads(token_json_str)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _get_credentials():
    google_cfg = _load_config().get('google', {})
    token_info = google_cfg.get('token')
    creds = None
    if token_info:
        creds = Credentials.from_authorized_user_info(token_info, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_config = google_cfg.get('credentials')
            if not client_config:
                raise FileNotFoundError(
                    'Google-Zugangsdaten fehlen in config.json (Schlüssel google.credentials)'
                )
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=0)
        _save_token(creds.to_json())
    return creds


def add_run_event(date_str, session_label, duration_min=EVENT_DURATION_MIN, description=''):
    """Add a 'run - <session_label>' event at 09:00 (orange) on date_str (YYYY-MM-DD)."""
    creds = _get_credentials()
    service = build('calendar', 'v3', credentials=creds)

    start = dt.datetime.strptime(date_str, '%Y-%m-%d').replace(hour=EVENT_HOUR, minute=0)
    end   = start + dt.timedelta(minutes=duration_min)

    event = {
        'summary': f'run - {session_label}',
        'colorId': COLOR_ORANGE,
        'start': {'dateTime': start.isoformat(), 'timeZone': 'Europe/Vienna'},
        'end':   {'dateTime': end.isoformat(),   'timeZone': 'Europe/Vienna'},
    }
    if description:
        event['description'] = description
    return service.events().insert(calendarId='primary', body=event).execute()


def delete_event(event_id):
    """Delete a calendar event by its id (no-op if it no longer exists)."""
    creds = _get_credentials()
    service = build('calendar', 'v3', credentials=creds)
    try:
        service.events().delete(calendarId='primary', eventId=event_id).execute()
    except Exception:
        # 404/410 → already gone; treat as success
        pass
