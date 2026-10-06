# API Usage & Rate Limits

## Rate limits
API requests are limited to 600 requests per minute per API key on the Pro
plan, and 60 requests per minute on the Free plan. Exceeding the limit
returns an HTTP 429 response with a `Retry-After` header indicating how many
seconds to wait.

## Authentication
All API requests require an `Authorization: Bearer <API_KEY>` header. API
keys can be created and revoked under Settings > Developer > API Keys. Keys
are shown only once at creation time, so store them securely.

## Pagination
List endpoints return up to 100 items per page. Use the `cursor` field from
the response to fetch the next page via the `?cursor=` query parameter.
Results are ordered by creation time, newest first, by default.

## Webhooks
Webhooks notify your endpoint of events like `invoice.paid` or
`subscription.canceled`. Configure webhook URLs under Settings > Developer >
Webhooks. We retry failed webhook deliveries with exponential backoff for up
to 24 hours, and you can verify payload authenticity using the signature in
the `X-Signature` header.
