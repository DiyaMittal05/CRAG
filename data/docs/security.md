# Account Security

## Session timeouts
Web sessions expire after 24 hours of inactivity. Mobile app sessions stay
signed in for 30 days unless you manually sign out. Signing in on a new
device does not sign you out of existing sessions; manage active sessions
under Settings > Security > Active Sessions.

## Suspicious login alerts
We email you whenever a sign-in occurs from a new device or an unrecognized
location. The alert includes an "This wasn't me" link that immediately
revokes that session and forces a password reset on next login.

## Data encryption
All data is encrypted in transit via TLS 1.2+ and at rest using AES-256.
Encryption keys are rotated automatically every 90 days; this is handled
entirely on our side and requires no action from you.

## Reporting a security issue
Security vulnerabilities should be reported to security@example.com rather
than through normal support channels. We acknowledge valid reports within
48 hours and do not require a bug bounty signup to report.
