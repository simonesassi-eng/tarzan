# Newsletter automation — setup guide

This guide takes you from a fresh repo to receiving the Tarzan
newsletter at your inbox three times a day at the Italian market hours,
with on-demand "Update" replies wired in.

> **Time required:** ~30 minutes the first time.
> **What you'll have at the end:**
> - 3× scheduled sends per day (09:05, 13:00, 17:35 Europe/Rome),
>   driven by the Gmail Apps Script — never duplicated, never bursty
> - Reply "Update" to any newsletter → fresh send within ~5 minutes
> - One manual "Run workflow" button in the GitHub Actions UI
> - Your portfolio data stays in your private Google Drive — the
>   GitHub repo can be public.

---

## 0. Architecture in one diagram

The Gmail Apps Script is the **single scheduler**: it fires the three
daily market slots (and on-demand "Update" replies) by posting a
`repository_dispatch` to GitHub. GitHub Actions is a pure **runner** —
it has no `schedule:` cron of its own. This is deliberate: GitHub's
built-in cron is best-effort and releases backlogged runs in a burst,
which is what previously caused several newsletters to arrive
back-to-back.

```
                                                          .──────────.
                                                         (   Gmail    )
                                                          `──────┬───'
                                                                 │ 1. SMTP send
   ┌─────────────────────────┐                                   │
   │ Private Google Drive    │                                   │
   │ folder                  │ ◄─── 2. download CSVs ───┐        │
   │   ├── holdings.csv      │      (service account)   │        │
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
                                                │ • checkSchedule()  │ ← 3 daily slots,
                                                │     (Europe/Rome)  │   idempotent per
                                                │ • processInbox()   │   (date, slot)
                                                │     ("Update")     │
                                                └────────────────────┘
```

Three things never enter git:
- The portfolio CSVs (Drive only, fetched at runtime)
- The Google service-account key (GitHub Secret)
- The Gmail App Password (GitHub Secret)

---

## 1. Push the repo to GitHub (public is fine)

```bash
# from the repo root
git status
git add .gitignore docs/ scripts/ tarzan/export/ \
        .github/workflows/ requirements.txt prototypes/
git commit -m "feat(newsletter): scheduled email automation"
git push origin mainline   # or main, whichever you use
```

> **Sanity check:** browse the repo on github.com — there should be
> **no `.private/` folder, no `holdings.csv`, no `targets.csv`** anywhere.
> If you see them, stop and remove them before continuing.

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
2. Go to your Drive folder:
   <https://drive.google.com/drive/folders/1I9BaXVO1R7cpeps-USyrpWB759YQX48a>
3. Click **Share**.
4. Paste the `client_email`, set role to **Viewer**, untick "Notify
   people", click **Share**.

### 2f. Verify your folder contains the right files

The folder MUST contain exactly these two filenames at the top level:
- `holdings.csv`
- `targets.csv`

Anything else in the folder is ignored. The names are case-sensitive.

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
| `SMTP_USER`                     | `simonesassi4@gmail.com`                               |
| `SMTP_PASS`                     | The 16-char App Password from step 3 (spaces don't matter) |
| `RECIPIENT_EMAIL`               | `simonesassi4@gmail.com`                               |
| `DRIVE_FOLDER_ID`               | `1I9BaXVO1R7cpeps-USyrpWB759YQX48a` (from your folder URL) |
| `GOOGLE_DRIVE_CREDENTIALS_JSON` | The **entire** content of the JSON key file from step 2d |

> **For `GOOGLE_DRIVE_CREDENTIALS_JSON`:** open the JSON file in a text
> editor, select all, copy, paste into the GitHub Secret value field.
> GitHub stores it as-is — don't reformat it.

---

## 5. Test the workflow with a manual run

1. github.com → your repo → **Actions** tab → **Tarzan Newsletter**.
2. Click **Run workflow** → leave the label as `manual` → **Run**.
3. Wait ~3 minutes for the green check.
4. Open your inbox. You should see an email like
   `[Tarzan] manual · €213,476 (+X.XX%) · N action(s)`.
5. Open the run logs to confirm "Drive: downloaded ..." and
   "Sent newsletter to ...".

> **Something failed?** Jump to the troubleshooting section at the bottom.

---

## 6. Wire up the Gmail Apps Script (scheduler + "Update" replies)

This single Apps Script does two jobs: it **schedules** the three daily
sends, and it turns a Gmail "Update" reply into an on-demand send. Both
fire the same GitHub workflow via `repository_dispatch`.

### 6a. Create a fine-grained PAT for Apps Script

1. <https://github.com/settings/personal-access-tokens/new>
2. Settings:
   - **Token name:** `tarzan-apps-script`
   - **Expiration:** 1 year
   - **Repository access:** "Only select repositories" → choose your Tarzan repo
   - **Repository permissions** → **Actions** → **Read and write**
     (only this scope)
3. Click **Generate token**, copy the value (`github_pat_...`).

### 6b. Create the Apps Script project

1. Sign in as `simonesassi4@gmail.com` and go to <https://script.google.com>.
2. **New project**.
3. Replace the contents of `Code.gs` with the contents of
   `scripts/apps_script/Code.gs` from this repo. (Just copy and paste.)
4. Rename the project to `Tarzan Scheduler` (top-left).

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

> **Slot times** (09:05 / 13:00 / 17:35 Europe/Rome, weekend = morning
> only) live in the `SLOTS` constant at the top of `Code.gs`. Edit
> there if you want different hours; no other change needed.

### 6d. Approve OAuth and install the trigger

1. In `Code.gs`, the function dropdown at the top → pick `installTrigger`
   → click **Run**.
2. Apps Script asks for permissions. Approve all
   (Gmail read, UrlFetch).
3. This installs a single `tick()` trigger that runs every 5 minutes
   and handles both the schedule and "Update" replies. You can run
   `checkSchedule` or `processInbox` manually from the dropdown to test
   either path on demand.

### 6e. End-to-end test

1. Open the email you received in step 5.
2. Hit **Reply**, type literally `Update`, send.
3. Within 5 minutes the `tick()` trigger sees the reply and posts to GitHub.
4. **Actions** tab on GitHub → new run with event `repository_dispatch`.
5. Within ~3 more minutes, a fresh newsletter lands in your inbox.

That's it. Going forward:

- **Three sends per day, automatic** (09:05, 13:00, 17:35 Europe/Rome),
  each fired at most once per day — no duplicates, no bursts.
- **Reply "Update"** → fresh send within ~8 minutes total.
- **Click "Run workflow"** in the Actions UI for an instant manual send.
- **Edit the CSVs in your Drive folder** — next run picks up the changes.

---

## How scheduling and DST are handled

Scheduling lives entirely in the Gmail Apps Script, **not** in GitHub
Actions. The script's `tick()` trigger runs every 5 minutes and, for
each market slot, dispatches a send only when:

- the current `Europe/Rome` time is at or after the slot time, but
  within `MAX_LAG_MINUTES` (default 90) of it;
- weekend rules allow it (Sat/Sun = morning only); and
- no send has already happened for that `(date, slot)` — an idempotency
  marker in Script Properties guarantees **at most one send per slot per
  day**, no matter how the polling jitters or retries.

Because the script formats time in `Europe/Rome`, **DST is handled
natively** — a slot fires at the same wall-clock time year-round. There
is no UTC double-cron and no DST guard to maintain.

| Slot    | Italian time | Days       |
|---------|--------------|------------|
| morning | 09:05        | every day  |
| midday  | 13:00        | weekdays   |
| close   | 17:35        | weekdays   |

> **Why not GitHub's built-in cron?** It's UTC-only and best-effort: it
> queues runs under load and then releases them in a burst, which is
> what caused multiple newsletters to arrive back-to-back. A punctual
> external scheduler plus per-slot idempotency removes that failure mode
> at the root.

---

## Updating your portfolio inputs

When you buy/sell, update your CSVs **directly in the Drive folder**
(through Google Drive, Sheets export, or by replacing the file). The
next scheduled or on-demand run picks up the new data automatically.
No git commit, no push, no rebuild.

For local development, keep using `input/holdings.csv` and
`input/targets.csv` (which are gitignored).

---

## Troubleshooting

### "Drive folder is missing: holdings.csv, targets.csv"
- Re-check the folder ID in the `DRIVE_FOLDER_ID` secret. It's the
  segment after `/folders/` in the URL.
- Verify the service account email (from the JSON `"client_email"`
  field) is listed as **Viewer** on the folder.
- Confirm both filenames are spelled exactly as `holdings.csv` and
  `targets.csv`. Case matters.

### "Authentication unsuccessful" / SMTP login fails
- Double-check `SMTP_PASS` is the 16-char **App Password**, not your
  regular Google account password.
- Confirm 2-Step Verification is on (App Passwords require it).

### Email never arrives
- Check the GitHub Actions log for `"Sent newsletter to ..."`. If it
  says that, the issue is on Gmail's side — check Spam.
- If a scheduled slot didn't fire at all, open Apps Script →
  **Executions** and check the recent `tick()` runs. A slot is skipped
  (logged `too-late`) if the trigger didn't run within 90 minutes of the
  slot time; trigger manually with "Run workflow" if you need that send.

### A scheduled send was missed
- Apps Script triggers are reliable but not instantaneous; a slot fires
  on the first `tick()` at or after its time, within a 90-minute window.
- Check Apps Script → **Executions** for `checkSchedule` log lines like
  `Dispatched slot "midday" ...` or `Slot "..." skipped: ...`.
- To force a slot to re-send today, delete its `sent:YYYY-MM-DD:slot`
  entry under **Project Settings → Script Properties**, or just use the
  "Run workflow" button.

### Apps Script doesn't dispatch
- In Apps Script: **Executions** (left sidebar). Look for failed
  `tick()` runs. The error column tells you what went wrong.
- Common: `GH_TOKEN` expired or wrongly scoped. Recreate the PAT
  (step 6a) and update the Script Property.
- The label `tarzan-update-handled` is added to threads we've
  processed. To re-test the same reply, remove the label from the
  thread in Gmail.

### "Run workflow" button is missing
The `workflow_dispatch` trigger requires the workflow file to be on the
**default branch**. Push `.github/workflows/newsletter.yml` to `main`
(or `mainline`) first.

### Pipeline fails on `enrich_holdings`
yfinance rate-limits. Wait a few minutes, re-run.

---

## Manual local test (no Drive, no email send)

For pure local development with the same CSVs you use day-to-day:

```bash
# from repo root
SMTP_USER=fake@example.com SMTP_PASS=fake \
RECIPIENT_EMAIL=fake@example.com \
DRY_RUN=1 \
HOLDINGS_PATH=input/holdings.csv \
TARGETS_PATH=input/targets.csv \
python scripts/send_newsletter.py
```

The script prints `DRY_RUN=1 — skipping SMTP send.` and writes the
HTML to `output/newsletter_<timestamp>.html`. Open that file in a
browser to inspect.
