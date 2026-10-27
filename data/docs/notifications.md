# Notifications

## Notification channels
Notifications can be delivered by email, SMS, and in-app push. Configure
each channel independently under Settings > Notifications; by default, only
email is enabled.

## What triggers a notification
Billing events (successful charge, failed charge, upcoming renewal),
security events (new-device login, password changed), and product events
(export ready, webhook delivery failure) each have their own on/off toggle.
Security notifications cannot be fully disabled, only moved between
channels, since they protect account access.

## Digest vs. real-time
Product notifications can be set to real-time or batched into a daily
digest sent at 9am in your account's timezone. Billing and security
notifications are always sent in real time regardless of this setting.

## Unsubscribing
Marketing emails include an unsubscribe link and can also be turned off
under Settings > Notifications > Marketing. Transactional notifications
(receipts, security alerts) are not affected by the marketing unsubscribe
and cannot be turned off via that link.
