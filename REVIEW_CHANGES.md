# Änderungen aus dem Review

- Diagnoseantworten und Logs enthalten keinen API-Schlüssel mehr.
- Fehlerhafte/limitierte Providerantworten ergeben fehlende Daten statt erfundener Ersatzwerte; Statementfehler lösen keine rekursiven Requests aus.
- Ohne `ALPHAVANTAGE_API_KEY` bleiben echte Daten nicht verfügbar. Nur `VALUE_ALLOW_DUMMY=true` aktiviert ausdrücklich Demo-Daten.
- `/lynch` und `/debug/keys` liefern die benötigten Wachstumsdaten an die Lynch-Strategie.
- Historische Snapshots mit `asof` werden vor dem Datenabruf abgewiesen (REST: HTTP 422). DIRECT und REST-Backtest-Engines dürfen aktuelle Fundamentals nicht unter historischen Daten speichern. Historische Läufe können erst nach Einführung von Point-in-time-Daten wieder aktiviert werden; alte Ergebnisse müssen erneut erzeugt werden.

Tests: `python -m unittest discover -s tests -v` (5 Regressionstests, keine echten Providerrequests).
