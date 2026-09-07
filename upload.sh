#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  upload.sh — lädt den kompletten Inhalt dieses Ordners nach GitHub hoch
#  Repository: https://github.com/moritzgraemer/fitness_dashboard
#
#  Benutzung:
#     ./upload.sh                      # Commit-Nachricht wird automatisch erzeugt
#     ./upload.sh "Meine Nachricht"    # eigene Commit-Nachricht
#     ./upload.sh -f "Nachricht"       # erzwingt den Push (überschreibt GitHub!)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

BRANCH="main"
REMOTE="origin"
FORCE=0

# ── Argumente ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "-f" || "${1:-}" == "--force" ]]; then
    FORCE=1
    shift
fi
MESSAGE="${1:-Update: $(date '+%d.%m.%Y %H:%M')}"

# ── Immer im Repo-Wurzelverzeichnis arbeiten ────────────────────────────────
cd "$(dirname "$0")"
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "FEHLER: Hier ist kein Git-Repository." >&2
    exit 1
fi

echo "→ Repository:  $(git remote get-url "$REMOTE")"
echo "→ Branch:      $BRANCH"
echo

# ── Alles vormerken (respektiert .gitignore) ────────────────────────────────
git add -A

# ── Sicherheitsnetz: keine Geheimnisse / persönlichen Daten hochladen ───────
#    (BSD-grep auf macOS kann kein -P, daher zwei einfache ERE-Prüfungen)
VERBOTEN='^(config\.json|credentials\.json|token\.json|\.env|backend/(GarminConnectConfig|google_credentials|google_token)\.json|.*\.pem)$'
GEFUNDEN=$(git diff --cached --name-only | grep -E "$VERBOTEN" || true)
DATEN=$(git diff --cached --name-only | grep -E '^datenbanken/' | grep -v '^datenbanken/\.gitkeep$' || true)
[[ -n "$DATEN" ]] && GEFUNDEN=$(printf '%s\n%s' "$GEFUNDEN" "$DATEN" | grep -v '^$')

if [[ -n "$GEFUNDEN" ]]; then
    echo "ABBRUCH: Diese Dateien enthalten Zugangsdaten oder persönliche Daten" >&2
    echo "         und dürfen nicht in ein öffentliches Repository:" >&2
    echo "$GEFUNDEN" | sed 's/^/           /' >&2
    echo >&2
    echo "         Sie wurden wieder aus der Vormerkung entfernt." >&2
    echo "         Ergänze sie in der .gitignore. Falls Git sie bereits" >&2
    echo "         verfolgt, zusätzlich:  git rm --cached <datei>" >&2
    # Nur die beanstandeten Dateien entfernen, den Rest der Vormerkung behalten
    echo "$GEFUNDEN" | while IFS= read -r f; do
        [[ -n "$f" ]] && git reset --quiet -- "$f"
    done
    exit 1
fi

# ── Warnung bei sehr großen Dateien (GitHub-Limit: 100 MB) ──────────────────
while IFS= read -r datei; do
    [[ -f "$datei" ]] || continue
    groesse=$(wc -c < "$datei")
    if (( groesse > 50000000 )); then
        echo "WARNUNG: $datei ist $(( groesse / 1048576 )) MB groß (GitHub-Limit: 100 MB)"
    fi
done < <(git diff --cached --name-only)

# ── Commit ──────────────────────────────────────────────────────────────────
if git diff --cached --quiet; then
    echo "Keine Änderungen zum Committen — es wird nur gepusht."
else
    echo "Geänderte Dateien:"
    git diff --cached --name-status | sed 's/^/  /'
    echo
    git commit -m "$MESSAGE"
    echo
fi

# ── Push ────────────────────────────────────────────────────────────────────
PUSH_ARGS=(--set-upstream "$REMOTE" "$BRANCH")
if (( FORCE )); then
    echo "→ Erzwinge Push (--force-with-lease) …"
    PUSH_ARGS=(--force-with-lease "${PUSH_ARGS[@]}")
fi

if git push "${PUSH_ARGS[@]}"; then
    echo
    echo "✓ Fertig: https://github.com/moritzgraemer/fitness_dashboard"
else
    echo >&2
    echo "Der Push wurde abgelehnt. Häufigste Ursachen:" >&2
    echo >&2
    echo "  a) Auf GitHub liegen Commits, die du lokal nicht hast." >&2
    echo "     Erst holen und zusammenführen, dann erneut hochladen:" >&2
    echo "       git pull --rebase $REMOTE $BRANCH && ./upload.sh" >&2
    echo >&2
    echo "  b) Die Historien passen nicht zusammen (z. B. GitHub hat einen" >&2
    echo "     eigenen \"Initial commit\"). Wenn dein lokaler Stand gelten soll:" >&2
    echo "       ./upload.sh -f \"$MESSAGE\"" >&2
    echo "     ACHTUNG: Das überschreibt den Stand auf GitHub unwiderruflich." >&2
    exit 1
fi
