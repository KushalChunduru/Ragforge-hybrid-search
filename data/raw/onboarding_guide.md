# Engineering Onboarding Guide

## Week One

Get access to the internal GitHub org, the staging Kubernetes cluster, and the
`#eng-oncall` Slack channel. Run `make bootstrap` from the monorepo root to
install pinned toolchain versions and pre-commit hooks.

## Local Development

Services run locally via `docker compose up`. The API gateway listens on
`localhost:8080` and proxies to individual services by path prefix. Hot
reload is enabled by default; set `HOT_RELOAD=0` to disable it when profiling.

## Code Review Norms

Every PR needs one approval from a service owner before merge. Large
refactors (500+ lines) should be split unless explicitly agreed otherwise in
the team channel. CI must be green; flaky test retries are capped at 2.

## Who to Ask

- Gateway / auth questions: platform team, `#platform-help`.
- Data pipeline questions: data-eng team, `#data-eng-help`.
- Incident response: see the incident runbook for the current on-call rotation.
