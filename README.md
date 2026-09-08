# Applied Bayesian Econometrics

Quarto course website with 15 lecture chapters and 15 matching Python exercise sessions.
The fixed sequence, assigned literature and exercise focus are in `schedule.qmd`.

## Local build

Install Quarto, then use the existing project environment:

```sh
uv sync
uv run quarto preview
uv run quarto render
```

The output goes to `_site/`. The existing GitHub Actions workflow builds and publishes
on pushes to `main`; a local render does not publish anything.

## Teaching material

- `lecture-notes/`: intuition, posterior derivations, interpreted code and reading guides.
- `exercise-sessions/`: each scheduled lab, a collapsible reference implementation and
  the chapter's additional practice questions.
- `lecture-slides/`: the existing introductory deck, expanded with the fixed course
  sequence, Bayes and Beta-posterior derivations, and the Week 1 exercise.
- `literature.qmd`: assigned readings and a map of the supplementary UPM and local sources.

The Week 10 and Week 14 reference implementations require `pymc` and `pymc-bart`.
Those dependencies have not been added to the project. These two blocks use
`eval: false`, so the website displays their source without claiming executed results.
All other exercise solutions use the existing numerical dependencies.

The pages use `freeze: auto`. Explicitly render a changed chapter to refresh its output;
the code examples simulate their own data and use fixed seeds. Sampling results remain
subject to Monte Carlo error. The course notes distinguish reference implementations
from production inference and document the assumptions required by each application.
