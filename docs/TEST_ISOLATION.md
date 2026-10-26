# Offline unit-test contract

The ordinary `python -m pytest tarzan/tests -q` command is independent of
market services, API keys and SMTP. An autouse fixture blocks the suite's
known native requests, curl, urllib, socket/DNS and SMTP ports. It records
channel names only and rejects a test at teardown even when a product
fallback caught the denied call. It is a test gate, not an operating-system
network sandbox or authority for production availability.

Tests of transport/parsing explicitly install synthetic native HTTP responses
at their transport boundary. Tests of a higher-level computation declare
missing optional sources with narrowly scoped fixtures. The justETF-absence
fixture returns 404 only for justETF API URLs; unrelated requests remain
blocked. Provider ownership, parsers, positive-response tests, cancellation
and financial assertions are not globally replaced or weakened.

Known production credentials are removed before each test; authentication
tests inject their own synthetic credentials afterward. The SDK's identity
and cookie cache uses a temporary suite directory, not the developer's
persistent cache. Product input files and market caches remain untouched.

Previously, passing tests attempted unmocked network I/O and depended on
whatever sources returned that day. An offline observer recorded 42 denied
calls across 19 test/channel rows. These observations diagnose missing test
isolation, not a failed newsletter or a measured delivery-time regression.
The guard and explicit fixtures correct that defect without modifying
portfolio formulas or the delivery workflow. CI latency must be measured
separately; test-suite time is not newsletter-send time.
