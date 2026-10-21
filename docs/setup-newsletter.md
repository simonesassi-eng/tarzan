# Newsletter automation — setup guide

This guide configures the supported single-user newsletter schedule and
authenticated on-demand "Update" replies. Run/deploy timing and inbox receipt
are best-effort, not guarantees.

> **What you'll have at the end:**
>
> - Scheduled sends at Italian market hours (08:00 pre-open, a first
>   digest at 09:45 then a 90-minute cadence to the 17:35 wrap-up, the
>   18:00 official-close digest, 23:00 recap;
>   weekends 08:00 / 13:05 / 23:00), driven by the Gmail Apps Script
> - A new unquoted "Update" reply from the verified trigger owner can request
>   another run; it must still pass freshness, analysis, claim and SMTP gates
> - One manual "Run workflow" button in the GitHub Actions UI
> - Your source CSVs stay in private Google Drive, with private ephemeral
>   staging on the runner; the GitHub source repo can be public

No push deploys Apps Script. Saving trigger code and updating a configured
web-app deployment are separate operator steps (section 6).

---

## 0. Architecture in one diagram

The Gmail Apps Script is the **single scheduler**: it attempts the configured
market slots (and authenticated on-demand "Update" replies) by posting a
`repository_dispatch` to GitHub. GitHub Actions is a pure **runner** —
it has no `schedule:` cron of its own. Apps Script timing is best-effort too:
the scheduler admits a slot only within 25 minutes, and Actions suppresses
requests whose oldest request/native-run time is at least 45 minutes old.
The bounded native queue permits at most 100 pending runs, not unlimited
durable event storage or burst-free/punctual delivery.

```
                                                          .──────────.
                                                         (   Gmail    )
                                                          `──────┬───'
                                                                 │ 1. SMTP send
   ┌─────────────────────────┐                                   │
   │ Private Google Drive    │                                   │
   │ folder                  │ ◄─── 2. download CSVs ───┐        │
   │   ├── order_list.csv    │      (service account)   │        │
   │   └── targets.csv       │                          │        │
   └─────────────────────────┘                          │        │
                                                ┌───────┴────────┴───┐
   manual click ────────────────────────────►   │   GitHub Actions   │
                                                │  newsletter.yml    │
                                                │  (runner only)     │
                                                │  • download Drive  │
                                                │  • run pipeline    │
                                                │  • render HTML     │
                                                │  • SMTP send       │
                                                └─────────┬──────────┘
                                                          ▲
                                       repository_dispatch│ (event: send_now)
                                                          │ label = slot / on-demand
                                                ┌─────────┴──────────┐
                                                │ Gmail Apps Script  │
                                                │ tick() every 5 min:│
                                                │ • checkSchedule()  │ ← market slots,
                                                │     (Europe/Rome)  │   idempotent per
                                                │ • processInbox()   │   (date, slot)
                                                │     ("Update")     │
                                                └────────────────────┘
```

Three things must never enter git:

- The portfolio CSVs (private Drive or ignored local inputs)
- The Google service-account key (GitHub Secret)
- The Gmail App Password (GitHub Secret)

---

## 1. Push the repo to GitHub (public is fine)

Review and stage only intended source changes. Never use a broad `git add`
that includes personal inputs, provider captures, `.private/` or output.

```bash
# from the repo root
git status --short
# Example: explicitly reviewed automation source, not the whole tree.
git add .github/workflows/newsletter.yml \
        scripts/apps_script/Code.gs scripts/apps_script/appsscript.json
git diff --cached --stat
git diff --cached
git commit -m "feat(newsletter): scheduled email automation"
git push origin main
```

> **Sanity check:** browse the repo on github.com — there should be
> **no `.private/` folder, no `order_list.csv`, no `targets.csv`** anywhere.
> If you see them, stop and remove them before continuing.

An ignore rule does not remove previously published history. Inspect staged
content itself, not just the current folder listing. Install locally from
the root hash lock in Python 3.12 as shown in [the README](../README.md);
do not use the former floating legacy dependency list.

---

## 2. Set up the Google Drive integration

You need a "service account" — a robot Google identity that can read
your folder without you sharing your personal Drive credentials.

### 2a. Create a Google Cloud project (skip if you already have one)

1. Go to <https://console.cloud.google.com/projectcreate>.
2. Project name: `tarzan-newsletter` (anything works).
3. Click **Create**, wait for the project to be ready, select it.

### 2b. Enable the Drive API

1. <https://console.cloud.google.com/apis/library/drive.googleapis.com>
2. Click **Enable**.

### 2c. Create the service account

1. <https://console.cloud.google.com/iam-admin/serviceaccounts>
2. Click **Create service account**.
3. Service account name: `tarzan-drive-reader`.
4. Click **Create and continue** (you can skip the role step).
5. Click **Done**.

### 2d. Generate the JSON key

1. Click on the service account you just created.
2. **Keys** tab → **Add Key** → **Create new key** → **JSON** → **Create**.
3. A JSON file downloads. **Keep it safe — you won't be able to download it again.**

### 2e. Share your Drive folder with the service account

1. Open the JSON file. Copy the value of `"client_email"` (looks like
   `tarzan-drive-reader@tarzan-newsletter.iam.gserviceaccount.com`).
2. Go to your Drive folder
   (`https://drive.google.com/drive/folders/<your-folder-id>`).
3. Click **Share**.
4. Paste the `client_email`, set role to **Viewer**, untick "Notify
   people", click **Share**.

### 2f. Verify your folder contains the right files

The folder MUST contain these filenames at the top level:
- `order_list.csv`
- `targets.csv`
- `targets_per_holding.csv` (optional — per-instrument rebalancing targets)

Anything else in the folder is ignored. The names are case-sensitive.
Each required name must resolve to one unambiguous uploaded CSV file, not a
native Google Sheet/shortcut. Duplicate names, partial Drive configuration or
changed size/checksum evidence reject inputs instead of selecting an older
local ledger. The optional target file may be absent; a present invalid file
is not treated as absence.

---

## 3. Create the Gmail App Password

1. Go to <https://myaccount.google.com/security>.
2. Turn on **2-Step Verification** if not already on.
3. Visit <https://myaccount.google.com/apppasswords>.
4. App name: `Tarzan` → **Create**.
5. Google shows a 16-character password (e.g. `abcd efgh ijkl mnop`).
   **Copy it now** — Google won't show it again.

---

## 4. Add GitHub Secrets

In your repo on github.com:

1. **Settings** → **Secrets and variables** → **Actions** → **New repository secret**
2. Add these five secrets exactly:

| Name                            | Value                                                  |
|---------------------------------|--------------------------------------------------------|
| `SMTP_USER`                     | Your Gmail sending account                            |
| `SMTP_PASS`                     | The 16-char App Password from step 3 (spaces don't matter) |
| `RECIPIENT_EMAIL`               | The intended inbox                                    |
| `DRIVE_FOLDER_ID`               | Native ID after `/folders/`, without slashes or URL    |
| `GOOGLE_DRIVE_CREDENTIALS_JSON` | The **entire** content of the JSON key file from step 2d |

> **For `GOOGLE_DRIVE_CREDENTIALS_JSON`:** open the JSON file in a text
> editor, select all, copy, paste into the GitHub Secret value field.
> GitHub stores it as-is — don't reformat it.

Optional `GEMINI_API_KEY` enables AI text. If using the existing durable claim
service, set both `DELIVERY_CLAIM_ENDPOINT` and `DELIVERY_CLAIM_TOKEN`.
An incomplete pair or an unavailable configured service blocks SMTP; it does
not fall back locally. With neither present, the intentional fallback is
host-local: an empty ephemeral runner cannot promise cross-run deduplication.
SMTP acknowledgment never proves inbox receipt.

Actions logs fixed status codes. Its public cache is one allowlisted
fixed-macro price snapshot, not the private runtime cache or the Drive files.
Private staging/output are ephemeral and not uploaded as artifacts.

---

## 5. Test the workflow with a manual run

1. github.com → your repo → **Actions** tab → **Tarzan Newsletter**.
2. Click **Run workflow** → leave the label as `manual` → **Run**.
3. Wait for completion; queueing, providers and native services affect duration.
4. A normal subject has the form `Portfolio Digest - HH:MM - <session date>
   ±X.XX%`. Fallback bases are labeled `1D` or `uP&L`; trigger labels and
   portfolio totals are not appended.
5. Check fixed statuses such as `SMTP_ACKNOWLEDGED`. A green run can also be
   `SKIPPED_STALE`, `SUPPRESSED_DUPLICATE` or a sanitized failure notification;
   it does not itself prove a fresh normal newsletter reached the inbox.
   Full financial diagnostics belong in a private local run, not public logs.

> **Something failed?** Jump to the troubleshooting section at the bottom.

---

## 6. Wire up the Gmail Apps Script (scheduler + "Update" replies)

This single Apps Script does two jobs: it attempts the configured daily
slots, and authenticates a new Gmail "Update" reply for an on-demand run. Both
fire the same GitHub workflow via `repository_dispatch`.

### 6a. Create a fine-grained PAT for Apps Script

1. <https://github.com/settings/personal-access-tokens/new>
2. Settings:
   - **Token name:** `tarzan-apps-script`
   - **Expiration:** a bounded lifetime (for example 90 days, within your
     organization policy); plan rotation before expiry
   - **Repository access:** "Only select repositories" → choose your Tarzan repo
   - **Repository permissions** — **Contents → Read and write**.
     Leave Actions and unrelated permissions ungranted. This script posts
     `repository_dispatch`, not workflow-editing Actions API calls.

     The native endpoint's required permission is documented in
     [GitHub's fine-grained token permissions](https://docs.github.com/en/rest/authentication/permissions-required-for-fine-grained-personal-access-tokens#repository-permissions-for-contents).
3. Click **Generate token**, copy the value (`github_pat_...`).

### 6b. Create the Apps Script project

1. Sign in as the same Gmail account and go to <https://script.google.com>.
2. **New project**.
3. Replace the contents of `Code.gs` with the contents of
   `scripts/apps_script/Code.gs` from this repo. (Just copy and paste.)
4. Rename the project to `Tarzan Scheduler` (top-left).
5. In **Services** (plus button), add **Gmail API**, identifier `Gmail`.
   For a standard Cloud project, enable Gmail API in its Cloud console too.
   Native OAuth verifies the mailbox owner and each message's own `SENT`
   label. Received messages, even with a forged owner From header, cannot
   authorize an Update. Missing API/scopes/owner fail closed; scheduled slots
   remain independently handled by `checkSchedule`.
6. Confirm the manifest uses V8, `Europe/Rome` and the Gmail advanced service,
   matching `scripts/apps_script/appsscript.json`.

> **Updating the script later.** Pushing to GitHub does NOT deploy it. Open the
> project from <https://script.google.com/home>, replace the whole of `Code.gs` with
> the repo's
> `scripts/apps_script/Code.gs`, and **Save**. The `tick()` trigger runs the
> latest *saved* code. For the authenticated-Update protocol upgrade, add Gmail
> API and run `installTrigger()` once: it migrates existing messages on old
> handled threads into per-message markers, preventing an old request from
> being dispatched again. Send a new reply after installation to verify it.
> A later reply in the same thread is eligible; its informational thread label
> no longer suppresses it. Run `validateSlots` too. A configured claim service
> uses `doPost` at the web-app `/exec` endpoint: **Manage deployments → Edit →
> New version → Deploy** updates that endpoint separately. Keep its existing
> claim authentication/property configuration. Saving code or reinstalling
> `tick()` does not update a deployed web-app version.

### 6c. Set Script Properties

In the Apps Script editor: **Project Settings** (gear icon, left
sidebar) → scroll to **Script Properties** → **Edit** → **Add property**:

| Property name | Value                                       |
|---------------|---------------------------------------------|
| `GH_OWNER`    | Your GitHub username (e.g. `simonesassi-eng`) |
| `GH_REPO`     | Repo name (e.g. `tarzan`)                   |
| `GH_TOKEN`    | The PAT from step 6a                        |

The other properties (`LABEL_NAME`, `SUBJECT_MATCH`, `WORD_MATCH`)
have sensible defaults — leave them blank.

> **Slot times** live in `_buildSlots_()` at the top of `Code.gs` (see
> the table further down for the full schedule). Edit there if you want
> different hours; no other change needed. After editing, run
> `validateSlots()` from the editor — it fails if two slots land closer
> than `MAX_LAG_MINUTES`, which would make them share a tick and send
> twice.

### 6d. Approve OAuth and install the trigger

1. In `Code.gs`, the function dropdown at the top → pick `installTrigger`
   → click **Run**.
2. Review and authorize the native scopes required by this source (mailbox
   reading/labeling, external requests and trigger management), as the intended
   Gmail owner. Missing service/authorization must not be bypassed.
3. This installs a single `tick()` trigger that runs every 5 minutes
   and handles both the schedule and "Update" replies. You can run
   `checkSchedule` or `processInbox` manually from the dropdown to test
   either path on demand.

### 6e. End-to-end test

1. Open the email you received in step 5.
2. Hit **Reply**, type literally `Update`, send.
3. On the next tick, a new unquoted reply from the verified trigger owner,
   natively recorded as `SENT`, posts to GitHub. Replies older than 45 minutes,
   automated newsletters and quoted/forwarded trigger text do not qualify.
4. **Actions** tab on GitHub → new run with event `repository_dispatch`.
5. Follow the run's fixed status codes and inbox result. There is no guaranteed
   completion time; old queued requests skip SMTP rather than sending stale.

That's it. Going forward:

- **Scheduled sends, automatic** (08:00 pre-open, the 09:45–17:35 market
  cadence and the 18:00 official close on weekdays / 13:05 on weekends,
  23:00 recap — Europe/Rome),
  guarded by per-slot dispatch markers, not a receipt guarantee.
- **Reply "Update"** → an authenticated per-message request on a later tick.
- **Click "Run workflow"** → a new manual run subject to the same gates.
- **Edit the CSVs in your Drive folder** — next run picks up the changes.

Update idempotency is per message, under the script lock. Markers are bounded;
the listener pauses instead of evicting recent idempotency evidence at capacity.
Search and native-verification work are bounded per tick. Public GitHub dispatch
payloads contain an opaque event digest and request timestamp, not message
bodies, addresses or newsletter subjects/portfolio figures.

---

## How scheduling and DST are handled

Scheduling lives entirely in the Gmail Apps Script, **not** in GitHub
Actions. The script's `tick()` trigger runs every 5 minutes and, for
each market slot, dispatches a send only when:

- the current `Europe/Rome` time is at or after the slot time, but
  within `MAX_LAG_MINUTES` (default 25) of it;
- weekend rules allow it (Sat/Sun: 08:00, 13:05 and 23:00 only); and
- no dispatch marker exists for `(date, slot)`. Script locking and markers
  avoid ordinary repeated dispatches; the GitHub HTTP call and marker write
  are not one transaction. Delivery claims separately coordinate SMTP when
  their shared durable service is configured.

Because the script formats time in `Europe/Rome`, **DST is handled
natively** — a slot's target is the same wall-clock time year-round. There
is no UTC double-cron and no DST guard to maintain.

| Slot      | Italian time                        | Days       |
|-----------|-------------------------------------|------------|
| preopen   | 08:00                               | every day  |
| wd-HHMM   | 09:45, then 10:35 → 16:35 every 90 min | weekdays |
| close     | 17:35                               | weekdays   |
| official-close | 18:00                        | weekdays   |
| weekend   | 13:05                               | weekends   |
| night     | 23:00                               | every day  |

> **Why not GitHub's built-in cron?** It does not own this project's schedule.
> One Europe/Rome scheduler makes ownership explicit. Neither scheduler timing
> nor the bounded Actions queue promises that every request executes or that
> messages never arrive close together.

---

## Updating your portfolio inputs

When you buy/sell, update your CSVs **directly in the Drive folder**
(through Google Drive, Sheets export, or by replacing the file). The
next scheduled or on-demand run picks up the new data automatically.
No git commit, push or rebuild. Replace the intended files without creating
ambiguous duplicate names. Identical legitimate fills are preserved; invalid
economic cash signs/target constraints are rejected rather than silently fixed.

For local development, keep using `input/order_list.csv` and
`input/targets.csv` (which are gitignored).

---

## Troubleshooting

### "Drive folder is missing: order_list.csv, targets.csv"

- Re-check the folder ID in the `DRIVE_FOLDER_ID` secret. It's the
  segment after `/folders/` in the URL.
- Verify the service account email (from the JSON `"client_email"`
  field) is listed as **Viewer** on the folder.
- Confirm the filenames are spelled exactly as `order_list.csv` and
  `targets.csv`. Case matters.

### "Authentication unsuccessful" / SMTP login fails

- Double-check `SMTP_PASS` is the 16-char **App Password**, not your
  regular Google account password.
- Confirm 2-Step Verification is on (App Passwords require it).

### Email never arrives

- Look for `SMTP_ACKNOWLEDGED`, then check Spam and the intended inbox.
  Server acceptance is not a receipt guarantee.
- `SKIPPED_STALE` means the operational request was too old; start a new manual
  event if you still want an update. `UNCERTAIN` never permits automatic resend.
- If a scheduled slot didn't fire at all, open Apps Script →
  **Executions** and check the recent `tick()` runs. A slot is skipped
  (logged `too-late`) if the trigger didn't run within `MAX_LAG_MINUTES`
  (25) of the slot time; trigger manually with "Run workflow" if you need that send.

### A scheduled send was missed

- Apps Script triggers are best-effort; a slot attempts dispatch
  on the first `tick()` at or after its time, within a 25-minute window.
- Check Apps Script → **Executions** for `checkSchedule` log lines like
  `Dispatched slot "wd-1205" ...` or `Slot "..." skipped: ...`.
- For a new manual request use **Run workflow**. Removing a slot marker does
  not reset a consumed durable delivery claim for that same event; do not
  delete claim state to bypass uncertain-delivery reconciliation.

### `GitHub dispatch failed: HTTP 403` ("Resource not accessible by personal access token")

The Apps Script `tick()` ran on time but GitHub rejected the
`repository_dispatch`, so no workflow started and no email was sent. The
`GH_TOKEN` is missing a required permission. The most common cause is a
token created with "Actions" only — a fine-grained PAT needs
**Contents: Read and write** to fire `repository_dispatch`.

Fix: regenerate the PAT (see "Rotating the `GH_TOKEN`" below) with
**Contents: Read and write** on the
`tarzan` repo, then update the `GH_TOKEN` Script Property.

### Apps Script doesn't dispatch

- In Apps Script: **Executions** (left sidebar). Look for failed
  `tick()` runs. The error column tells you what went wrong.
- Common: `GH_TOKEN` expired or wrongly scoped. Recreate the PAT
  (see "Rotating the `GH_TOKEN`" below) and update the Script Property.
- Verify Gmail API is enabled and `installTrigger()` was authorized by the
  intended mailbox owner. New replies require their own native SENT proof and
  exact owner From, not an arbitrary sender/received thread.
- Send a **new reply** to re-test. Removing the informational thread label does
  not erase per-message markers; a later reply in the same thread is eligible
  without removing labels.

### Rotating the `GH_TOKEN`

The fine-grained PAT expires according to its configured lifetime. GitHub
shows a token's value **only once at
creation** — it cannot be retrieved later, so when it expires you
generate a fresh one. Symptoms of an expired/invalid token: Apps Script
**Executions** show `tick()` failing with `GitHub dispatch failed with
HTTP 401` (or `403`), and scheduled sends stop arriving.

To rotate it:

1. Go to <https://github.com/settings/personal-access-tokens/new>
   (signed in as the repo owner, e.g. `simonesassi-eng`).
2. Set:
   - **Token name:** `tarzan-apps-script`
   - **Expiration:** a bounded lifetime within your organization policy
   - **Repository access:** "Only select repositories" → the `tarzan` repo
   - **Repository permissions** — **Contents: Read and write** only.
     Leave unrelated permissions ungranted.
     (Contents R/W is what makes `repository_dispatch` work; without it
     GitHub returns HTTP 403.)
3. Click **Generate token** and **copy the value** (`github_pat_...`)
   immediately — you won't see it again.
4. In Apps Script: **Project Settings** (gear icon) → **Script
   Properties** → **Edit** → replace the `GH_TOKEN` value with the new
   token → **Save**.
5. Verify an Apps Script dispatch. A manual **Run workflow** uses GitHub's own
   authorization and does not test the Apps Script PAT. After a new dispatch
   works, revoke the replaced token. A dispatch still does not prove receipt.

> Existing fine-grained PATs are listed at
> <https://github.com/settings/tokens?type=beta> with their name and
> expiry — but the token **value** is never shown there. If you don't
> have the value saved, just generate a new one with the steps above.

### "Run workflow" button is missing

The `workflow_dispatch` trigger requires the workflow file to be on the
**default branch**. Push `.github/workflows/newsletter.yml` to `main`
(or `mainline`) first.

### Pipeline fails on `enrich_holdings`

Inspect fixed failure codes and reproduce privately if necessary. Providers
can rate-limit or lack history; retries/deadlines are bounded. Do not force a
price, remove a gate or run an unbounded retry loop. A new manual event is a
new delivery request; an uncertain prior invocation requires reconciliation.

---

## Manual local test (no Drive, no email send)

For pure local development with the same CSVs you use day-to-day:

```bash
# from repo root
unset DRIVE_FOLDER_ID GOOGLE_DRIVE_CREDENTIALS_JSON
SMTP_USER=sender@example.invalid SMTP_PASS=synthetic \
RECIPIENT_EMAIL=reader@example.invalid \
DRY_RUN=1 \
TARZAN_DISABLE_AI=1 \
ORDERS_PATH=input/order_list.csv \
TARGETS_PATH=input/targets.csv \
TARGETS_PER_HOLDING_PATH=input/targets_per_holding.csv \
python scripts/send_newsletter.py
```

First clear both Drive settings if they exist in your local environment;
an incomplete pair must not select local fallback. The script prints
`DRY_RUN=1 — skipping claim and SMTP send.` and writes the owned set to
`output/<date>/<attempt-id>/`, including `newsletter.html`, `report.html`,
`summary.json`, `ledger.jsonl` and the checksum manifest.
Open `newsletter.html` locally. A dry run still validates input/semantics but
creates no delivery claim and invokes no SMTP.

For a before/after KPI comparison, freeze the input files, provider observations
and time boundary; see [frozen verification](FROZEN_KPI_VERIFICATION.md).
Two live runs in different minutes cannot isolate a code-induced change.
Old public logs/caches/history do not become confidential merely because new
code uses private staging, restricted logging and a public macro allowlist.
