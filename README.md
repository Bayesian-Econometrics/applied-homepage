# Applied Bayesian Econometrics

Quarto course website with 15 lecture chapters and 15 matching Python exercise sessions.
The fixed sequence, assigned literature and exercise focus are in `schedule.qmd`.

## Local build

Install Quarto, then use the existing project environment:

```sh
uv sync --extra bart
uv run --extra bart quarto preview
uv run --extra bart quarto render
```

The output goes to `_site/`. The existing GitHub Actions workflow builds and publishes
on pushes to `main`; a local render does not publish anything.

Quarto generates JavaScript libraries for navigation, mathematical notation and slides.
These live in `_site/`, `.quarto/` and the `_freeze/` execution cache; they are not
hand-written course source files. All three directories and temporary `.quarto_ipynb`
notebooks are ignored by Git. The current CI workflow rebuilds from the source, so the
local cache does not need to be committed.

## Teaching material

- `lecture-notes/`: intuition, posterior derivations, interpreted code and reading guides.
- `exercise-sessions/`: each scheduled lab, a collapsible reference implementation and
  the chapter's additional practice questions.
- `lecture-slides/`: the existing introductory deck, expanded with the fixed course
  sequence, Bayes and Beta-posterior derivations, and the Week 1 exercise.
- `literature.qmd`: assigned readings and a map of the supplementary UPM and local sources.

The Week 10 and Week 14 reference implementations execute genuine PyMC-BART
samplers. The optional `bart` dependency group installs their tested PyMC 5 / PyMC-BART
0.11 API combination; `uv.lock` records the resolved environment. Use `--extra bart`
for a complete site build. The examples use sequential chains (`cores=1`) for
cross-platform execution and report sampling diagnostics. Initial compilation and
sampling can take several minutes. Other exercises use the core numerical packages.

The pages use `freeze: auto`. Explicitly render a changed chapter to refresh its output;
the code examples simulate their own data and use fixed seeds. Sampling results remain
subject to Monte Carlo error. The course notes distinguish reference implementations
from production inference and document the assumptions required by each application.
