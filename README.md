# Fitness-Dashboard

Selbstgehostetes Trainings-Dashboard für Ausdauersport. Liest Aktivitäten,
Schlaf und HRV aus Garmin Connect, berechnet daraus ein zweikanaliges
Lastmodell (aerobe und mechanische Last) und leitet daraus die Empfehlung für
die nächste Einheit ab. Läuft als einzelner Container, zum Beispiel auf Fly.io.

Es gibt zwei Oberflächen aus einer Codebasis: das Desktop-Dashboard unter `/`
und eine Mobilansicht unter `/m`, auf die Telefone automatisch geleitet werden.

## Was hier NICHT im Repository liegt

Dieses Repository ist öffentlich. Deshalb enthält es ausschließlich Code:

| Nicht im Repo | Wo es stattdessen liegt |
|---|---|
| Garmin-Zugangsdaten, API-Schlüssel | Fly-Secret `APP_CONFIG_JSON`, verschlüsselt |
| Passwort des Dashboards | Fly-Secret `DASH_PASSWORD` |
| Garmin-Sitzung (Token) | Volume unter `datenbanken/.garmin_tokens` |
| Aktivitäten, Herzfrequenz, Schlaf, HRV, GPS-Routen, Verletzungen | Volume unter `datenbanken/` |

`datenbanken/` ist im Repository ein leeres Verzeichnis und dient nur als
Mountpunkt. Die `.gitignore` schließt seinen Inhalt aus.

## Sicherheit

Das Dashboard zeigt Gesundheitsdaten und kann Workouts auf die Uhr schieben.
Es ist deshalb nur mit gesetztem `DASH_PASSWORD` erreichbar: jede Seite und
jede Programmierschnittstelle verlangt entweder das Login-Cookie (Formular
unter `/login`, 90 Tage gültig) oder HTTP-Basic-Auth mit demselben Passwort.
Ohne dieses Secret warnt der Start im Log.

**Der Garmin-Zugang wird nie Teil des Images oder des Repositories.** Fly.io
speichert Secrets verschlüsselt und reicht sie erst zur Laufzeit als
Umgebungsvariable in den Container. `app.py` und `backend/sync_garmin_csv.py`
lesen `APP_CONFIG_JSON` direkt aus der Umgebung; eine `config.json` auf der
Platte gibt es nur beim lokalen Entwickeln, und die steht in `.gitignore`.

## Einrichtung auf Fly.io

Voraussetzung: [flyctl](https://fly.io/docs/flyctl/install/) installiert und
`fly auth login` ausgeführt.

**1. App anlegen** und den Namen in `fly.toml` unter `app =` eintragen:

```bash
fly apps create mein-fitness-dashboard
```

**2. Volume für die Daten** anlegen, in derselben Region wie `primary_region`:

```bash
fly volumes create fitness_data --size 1 --region cdg --app mein-fitness-dashboard
```

**3. Secrets setzen.** Das Passwort für das Dashboard:

```bash
fly secrets set DASH_PASSWORD='ein-langes-passwort' --app mein-fitness-dashboard
```

Und die Zugangsdaten als ein JSON nach dem Muster von `config.example.json`
(Garmin-Login, optional Groq-Schlüssel und Google-Kalender):

```bash
fly secrets set APP_CONFIG_JSON="$(cat config.json)" --app mein-fitness-dashboard
```

**4. Deploy:**

```bash
fly deploy --app mein-fitness-dashboard
```

**5. Daten aufspielen.** Beim ersten Start ist das Volume leer, das Dashboard
zeigt „Keine Daten". Zwei Wege:

- Vorhandene Daten hochladen:

  ```bash
  scripts/upload_data.sh mein-fitness-dashboard /pfad/zu/datenbanken
  ```

- Oder bei null anfangen und im Dashboard auf Garmin-Sync tippen. Der erste
  Lauf holt die gesamte Historie und dauert einige Minuten.

Danach ist das Dashboard unter `https://<app-name>.fly.dev` erreichbar.
Auf dem iPhone landet man automatisch auf `/m`; dort über „Zum Home-Bildschirm"
hinzufügen, dann läuft es als Vollbild-App.

## Garmin-Sync im laufenden Betrieb

Der Sync-Knopf im Dashboard startet `backend/sync_garmin_csv.py` im Container.
Die Anmeldung läuft über den Token-Store auf dem Volume, das Passwort wird nur
gebraucht, wenn die Sitzung abläuft.

Verlangt Garmin dann eine Zwei-Faktor-Bestätigung, lässt sich das auf einem
Server nicht beantworten. In dem Fall einmal lokal synchronisieren und den
frischen Token hochladen:

```bash
scripts/upload_data.sh mein-fitness-dashboard
```

## Betrieb

```bash
fly logs --app mein-fitness-dashboard
```

```bash
scripts/download_data.sh mein-fitness-dashboard
```

Die Maschine schläft bei Inaktivität (`auto_stop_machines`), der erste Aufruf
danach dauert ein paar Sekunden. Ein Gigabyte Volume und `shared-cpu-1x` mit
einem Gigabyte Arbeitsspeicher liegen bei wenigen Euro im Monat.

## Lokal entwickeln

```bash
pip install -r requirements.txt
```

`config.example.json` nach `config.json` kopieren und ausfüllen, dann:

```bash
V2_PORT=5002 python3 serve_v2.py
```

Ohne `DASH_PASSWORD` läuft es lokal ohne Anmeldung. `/m` ist die Mobilansicht,
`/?desktop=1` erzwingt das Desktop-Dashboard auf schmalen Fenstern.

## Umgebungsvariablen

| Variable | Bedeutung |
|---|---|
| `DASH_PASSWORD` | schützt Oberfläche und Schnittstellen; ohne sie ist alles offen |
| `APP_CONFIG_JSON` | vollständige Zugangsdaten als JSON, ersetzt `config.json` |
| `DASH_SECRET` | eigener Schlüssel für das Login-Cookie (sonst aus dem Passwort abgeleitet) |
| `V2_HOST`, `V2_PORT` | Bind-Adresse und Port, im Container `0.0.0.0:8080` |
| `V2_DEBUG` | `1` schaltet den Flask-Reloader ein, auf dem Server `0` |

## Anpassen

Die persönlichen Schwellenwerte stehen oben in `app.py`: Laktatschwellen-Puls,
maximale Herzfrequenz, Rad-FTP und optional die Stryd-Laufleistungsschwelle.
`trainer_profile.md` beschreibt die Trainingsphilosophie und geht in die
KI-Analysen ein; die Datei kann gelöscht oder ersetzt werden.
