# Contributing

## Branches

Work on a branch, never commit directly to `main`:

- `feature/<name>` — new functionality
- `fix/<name>` — bug fixes
- `docs/<name>` — documentation-only changes

## Workflow

1. Branch off `main`.
2. Make your changes, with tests for new behaviour.
3. Before opening a pull request, make sure locally:
   ```
   python -m pytest -q
   python -m ruff check dms tests
   ```
   both pass.
4. Open a pull request into `main`. Describe what changed and why, and how
   you tested it. Every pull request needs at least one review before
   merging.
5. Keep pull requests focused — one logical change per PR is easier to
   review than a large mixed one.

## Commit messages

Short, imperative, present tense, under ~70 characters:

```
Add outdoor walkway network
Fix stair capacity formula for uneven risers
```

Avoid vague messages like "update" or "fixes" with no context — say what
changed.

## Code style

- No building-, room-, exit-, or stair-specific IDs or coordinates in
  `dms/` code — everything geometry- or building-specific belongs in a
  YAML config under `configs/`.
- No hard-coded physics or tuning constants in code — they belong in
  `configs/agent_defaults.yaml` or `configs/viz.yaml`, each with a unit and
  a source (a citation, or `[U]` for an unverified project default — see
  `SYSTEM_SPEC.md`).
- Every function that depends on a specific building should work for any
  valid building YAML — when practical, test against more than one fixture
  building.
- Keep simulation code free of rendering imports (`dms/scenario/`, `dms/graph/`,
  `dms/fields/`, `dms/geometry/` should never import `matplotlib`), so the
  simulation stays fast and headless.
- Videos in `out/videos/` are only re-committed at milestones, not on every
  render, to keep the repo small.
