# Contributing

Thanks for your interest in improving this project.

## Before you start

Please read the [medical disclaimer](DISCLAIMER.md). Contributions must not
add clinical claims, diagnostic logic, or treatment recommendations. Geometry
and tooling are in scope; medical advice is not.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ground rules

- **Never commit scan data or personal files.** No `.ply`, `.stl`, `.obj`,
  `.pcd`, archives of scans, PDFs, or anything derived from an identifiable
  person's foot. `.gitignore` blocks the common cases, but check `git status`
  before committing — an archive or a stray document will slip through any
  pattern list.
- **Keep the repository small.** Large sample data belongs in a GitHub Release
  asset or Git LFS, not in the git history.
- **Keep the web viewer local-first.** Do not change the `ply_viewer_web.py`
  defaults away from `host=127.0.0.1` and `debug=False`.

## Code style

- Python 3.10+, 4-space indentation, UTF-8, LF line endings — see
  `.editorconfig`.
- Some older modules are tab-indented; if you touch one, convert the whole
  file to spaces in a separate, formatting-only commit so that the functional
  diff stays readable.
- Use `argparse` for CLI entry points and type hints on new public functions,
  matching the existing modules.
- Prefer `yaml.safe_load` over `yaml.load`, and list-form `subprocess.run`
  without `shell=True`.

## Pull requests

- One logical change per pull request; separate formatting from behavior.
- Describe what you changed and how you verified it. If a change affects
  generated geometry, include a before/after preview image — most scripts
  accept a `--preview` flag.
- The `old/` directory is an archive of superseded experiments. It is kept for
  reference and is not maintained; please do not build on it.
