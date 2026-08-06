# API Reference

## Authentication

All requests must include an `Authorization: Bearer <token>` header. Tokens are
issued via `POST /v1/auth/token` and expire after 3600 seconds. Expired tokens
return HTTP 401 with error code `ERR_TOKEN_EXPIRED`.

## Rate Limiting

Rate limits are enforced per API key using a token bucket with a capacity of 600 requests per minute. When a client exceeds the limit, the gateway returns HTTP 429 with an ERR_RATE_LIMITED error code and a Retry-After header in seconds. The bucket refill rate is controlled by the config key gateway.rate_limit.rps.

## Endpoints

### GET /v1/documents

Returns a paginated list of documents. Accepts `page`, `page_size`, and
`updated_since` query parameters. `page_size` defaults to 25 and has a hard
cap of 200 enforced by the config key `api.pagination.max_page_size`.

### POST /v1/documents

Creates a new document. The request body is validated against the
`DocumentCreateSchema`. Validation failures return HTTP 422 with error code
`ERR_VALIDATION_FAILED` and a list of field-level messages.

### DELETE /v1/documents/{id}

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
