# Security Policy

## Reporting a vulnerability

Please report security issues privately using GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
("Report a vulnerability" under the repository's **Security** tab) rather than
opening a public issue.

Please include a description of the issue, reproduction steps, and the version
or commit you tested. Expect an initial response within a few days — this is a
volunteer-maintained project.

## Scope and threat model

This is **local-first, single-user desktop software**. It has no accounts, no
authentication, no network service intended for exposure, and it stores no
credentials.

The one network-facing component is the Dash web viewer (`ply_viewer_web.py`).
It is designed for local use only:

- It binds to `127.0.0.1` by default and runs with `debug=False` by default.
  **Keep both defaults.**
- It has **no authentication or authorization whatsoever**. Anyone who can
  reach the port can browse and load any point-cloud file (`.ply`, `.pcd`,
  `.xyz`, `.pts`) found by a recursive glob under the working directory, and
  can write files into `outputs/`.
- Passing `--host 0.0.0.0` exposes all of the above to your entire network.
  Do not do this on an untrusted network, and never expose it to the internet.
  If you need remote access, put it behind an authenticating reverse proxy or
  an SSH tunnel.
- `--debug` enables the Werkzeug interactive debugger, which permits remote
  code execution. Never combine `--debug` with a non-loopback `--host`.

## Untrusted input

Scan files are parsed by [Open3D](https://www.open3d.org/) and
[trimesh](https://trimesh.org/), and configuration files by PyYAML
(`yaml.safe_load`). Treat scan and mesh files from third parties as untrusted
input: parser vulnerabilities in these upstream libraries would be reachable
through this pipeline. Keep dependencies current, and prefer processing files
whose origin you trust.

## Privacy

Foot and insole scans are personal data, and depending on your jurisdiction may
be treated as health or biometric data. Do not commit scans, heightmaps, or
fitted presets derived from a real person to a public repository without that
person's informed consent. `scans/` and common 3D formats are excluded in
`.gitignore` for this reason.
