# Course homepage

Quarto-Website als Gerüst für die Kursseite. Noch ohne Inhalte und ohne Styling.

## Einmalig einrichten

1. Neues Repository auf GitHub anlegen, zum Beispiel `course-homepage`, leer und ohne README.
2. Diesen Ordner hineinschieben:

   ```sh
   git init -b main
   git add .
   git commit -m "Add Quarto site skeleton"
   git remote add origin git@github.com:<user>/course-homepage.git
   git push -u origin main
   ```

3. Im Repository unter Settings, Pages, bei Source `GitHub Actions` wählen. Nicht `Deploy from a branch`.
4. Unter Actions den Lauf `publish` abwarten. Danach steht die Seite unter

   ```
   https://<user>.github.io/course-homepage/
   ```

Jeder weitere Push auf `main` rendert und veröffentlicht automatisch neu.

## Lokal arbeiten

Quarto installieren (https://quarto.org/docs/get-started/), dann:

```sh
quarto preview   # lokale Vorschau mit Autoreload
quarto render    # baut nach _site/
```

`_site/` ist in `.gitignore` und wird nicht committet, das Rendern passiert in der Action.

## Python

`pyproject.toml` ist für die lokale Umgebung vorbereitet:

```sh
uv sync
```

Solange keine `.qmd`-Datei Code ausführt, braucht die Action kein Python. Sobald du
ausführbare Python-Chunks einbaust, muss der Workflow vor `quarto render` zusätzlich
`astral-sh/setup-uv` und `uv sync` ausführen, und `quarto render` läuft dann über
`uv run`.

Alternative, wenn das Ausführen in CI zu langsam oder zu fragil wird: lokal rendern,
`execute: freeze: auto` nutzen (ist bereits gesetzt) und den Ordner `_freeze/`
mitcommitten. Dann führt die Action keinen Code mehr aus, sondern verwendet die
eingefrorenen Ergebnisse.

## Was du anpassen willst

- Titel: `website.title` in `_quarto.yml` und die Überschrift in `index.qmd`
- Navigation: `website.navbar` in `_quarto.yml`
- Neue Seiten: `.qmd`-Datei anlegen und in der Navbar oder einer Sidebar eintragen
