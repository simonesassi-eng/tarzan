# Runtime installation

Newsletter publication installs the exact runtime projection of the single
immutable dependency lock. Test-only dependencies are excluded; all remaining
pins and hashes are unchanged. pip's resolver, hash checks, retry/time limits and
the subsequent `pip check` remain enabled. No executable virtualenv snapshot or
additional dependency-version authority is introduced.

The installer uses `--no-compile`: it does not eagerly compile every installed
dependency into bytecode. Python still compiles any source actually imported.
The workflow still compiles Tarzan and its scripts before execution. No
financial formula, source selection, market cache, input or SMTP contract is
changed by this option.

The appropriate performance boundary includes installation, Tarzan source
compilation and first operational imports, including the Google Drive
libraries. Installation alone can hide work shifted into the first import.
Fresh-environment comparisons must preserve Python, wheels, hash lock,
resolver behavior, hardware and the initial cache, and alternate the arms.

An offline local-wheel experiment measures this bounded component, not
live provider acquisition, download latency, AI or delivery time. Linux
verification and later natural sends remain separate release evidence.
No newsletter-runtime saving is claimed solely from a faster pip step.

The Linux experiment used Python 3.12.15, the same 47 runtime packages,
local verified wheels and eight fresh environments in ABBAABBA order.
Median install + Tarzan compile + first core/Drive imports was 16.259 s
with eager compilation and 6.852 s without it. The first import itself
increased from 0.826 s to 3.242 s; that cost is included, not hidden.
The four combined samples per arm ranged from 16.198–16.504 s and
6.836–6.862 s respectively. This is one runner's bounded component
experiment, excluding download, provider latency, AI and delivery.
