# Third-Party Integrations

## Available integrations
Native integrations exist for Slack (event notifications), Zapier (trigger
workflows on billing/product events), and Salesforce (sync customer
records). All integrations are configured under Settings > Integrations and
require an account on the Pro plan or higher.

## Slack integration
Connecting Slack requires workspace admin approval during OAuth. Once
connected, you choose which event types post to which channel; a single
workspace can only be connected to one account at a time.

## Zapier integration
The Zapier integration exposes triggers for `invoice.paid`,
`subscription.canceled`, and `export.ready`, and an action to create a
support ticket. It uses the same API key you generate under Settings >
Developer > API Keys, so revoking that key also disconnects Zapier.

## Disconnecting an integration
Disconnecting an integration under Settings > Integrations stops new events
immediately but does not delete data already sent to the third party before
disconnecting; that is managed on the third party's side.
