# Deployment Notes

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
