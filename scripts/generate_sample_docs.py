"""Generates a small set of placeholder 'internal docs' to exercise the
ingestion pipeline end to end: markdown, plain text, and HTML, with specific
technical terms (function names, config keys, error codes) for BM25 to catch,
and a couple of near-duplicate paragraphs (copy-pasted across files, as real
internal wikis tend to accumulate) to exercise the dedup path.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RATE_LIMIT_PARAGRAPH = (
    "Rate limits are enforced per API key using a token bucket with a capacity of "
    "600 requests per minute. When a client exceeds the limit, the gateway returns "
    "HTTP 429 with an ERR_RATE_LIMITED error code and a Retry-After header in seconds. "
    "The bucket refill rate is controlled by the config key gateway.rate_limit.rps."
)

API_REFERENCE_MD = f"""# API Reference

## Authentication

All requests must include an `Authorization: Bearer <token>` header. Tokens are
issued via `POST /v1/auth/token` and expire after 3600 seconds. Expired tokens
return HTTP 401 with error code `ERR_TOKEN_EXPIRED`.

## Rate Limiting

{RATE_LIMIT_PARAGRAPH}

## Endpoints

### GET /v1/documents

Returns a paginated list of documents. Accepts `page`, `page_size`, and
`updated_since` query parameters. `page_size` defaults to 25 and has a hard
cap of 200 enforced by the config key `api.pagination.max_page_size`.

### POST /v1/documents

Creates a new document. The request body is validated against the
`DocumentCreateSchema`. Validation failures return HTTP 422 with error code
`ERR_VALIDATION_FAILED` and a list of field-level messages.

### DELETE /v1/documents/{{id}}

Soft-deletes a document by setting `deleted_at`. Hard deletion happens via the
nightly `purge_deleted_documents` job, controlled by config key
`jobs.purge.retention_days` (default 30).

## Error Codes

| Code | Meaning |
|---|---|
| ERR_TOKEN_EXPIRED | Bearer token has expired, re-authenticate. |
| ERR_RATE_LIMITED | Client exceeded the configured rate limit. |
| ERR_VALIDATION_FAILED | Request body failed schema validation. |
| ERR_NOT_FOUND | Requested resource does not exist or was already deleted. |
"""

CONFIG_REFERENCE_HTML = f"""<!DOCTYPE html>
<html>
<head><title>Config Reference</title></head>
<body>
<h1>Config Reference</h1>

<h2>Gateway</h2>
<p>{RATE_LIMIT_PARAGRAPH}</p>
<p>Set <code>gateway.timeout_ms</code> to control the upstream request timeout.
The default is 5000ms. Values below 500ms are rejected at startup.</p>

<h2>Pagination</h2>
<p>The config key <code>api.pagination.max_page_size</code> caps the
<code>page_size</code> query parameter across all list endpoints. Raising this
value increases memory pressure on the query planner, so changes require
sign-off from the platform team.</p>

<h2>Jobs</h2>
<p>The <code>jobs.purge.retention_days</code> key controls how long
soft-deleted documents are kept before the nightly <code>purge_deleted_documents</code>
job hard-deletes them. Setting this to 0 disables the retention window entirely,
which is only intended for staging environments.</p>
</body>
</html>
"""

ONBOARDING_GUIDE_MD = """# Engineering Onboarding Guide

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
"""

INCIDENT_RUNBOOK_TXT = """Incident Response Runbook

Severity Levels
SEV1: Full outage or data loss risk. Page on-call immediately via PagerDuty.
SEV2: Significant degradation, no data loss. Post in #incidents within 15 minutes.
SEV3: Minor, customer-visible issue with a workaround. Track in the issue tracker.

Immediate Steps for SEV1/SEV2
1. Acknowledge the page within 5 minutes.
2. Open a #incident-<date> channel and post a one-line status.
3. Check the gateway dashboard for ERR_RATE_LIMITED and ERR_VALIDATION_FAILED
   spikes -- these usually indicate a bad deploy rather than a dependency outage.
4. If a recent deploy is the suspected cause, roll back with
   `deploy rollback --service <name> --to <previous_sha>`.
5. Update the status page once the immediate impact is mitigated.

Postmortems
Every SEV1 and SEV2 gets a postmortem within 3 business days. Postmortems are
blameless and focus on the sequence of events, contributing factors, and
concrete follow-up actions with owners and due dates.

On-Call Rotation
The on-call rotation is managed in PagerDuty under the "platform-primary"
schedule. Handoff happens every Monday at 10:00 local time. Check
`#eng-oncall` for the current primary and secondary.
"""

DEPLOYMENT_NOTES_MD = """# Deployment Notes

## Rollout Strategy

Deploys use a canary rollout: 5% of traffic for 10 minutes, then a full
rollout if error rates stay within baseline. Canary failures auto-rollback
via the `deploy rollback --service <name> --to <previous_sha>` command,
the same one used during incident response.

## Watch These Signals

During rollout, watch the gateway dashboard for spikes in `ERR_VALIDATION_FAILED`
(usually a schema mismatch between client and server) and `ERR_RATE_LIMITED`
(usually a misconfigured `gateway.rate_limit.rps` value in the new release).

## Config Changes

Config changes that touch `jobs.purge.retention_days` or
`api.pagination.max_page_size` require a platform team sign-off in the PR,
since both affect data retention and query load respectively.
"""


FILES = {
    "api_reference.md": API_REFERENCE_MD,
    "config_reference.html": CONFIG_REFERENCE_HTML,
    "onboarding_guide.md": ONBOARDING_GUIDE_MD,
    "incident_runbook.txt": INCIDENT_RUNBOOK_TXT,
    "deployment_notes.md": DEPLOYMENT_NOTES_MD,
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in FILES.items():
        path = RAW_DIR / filename
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(PROJECT_ROOT)} ({len(content)} chars)")


if __name__ == "__main__":
    sys.exit(main())
