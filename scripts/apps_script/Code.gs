/**
 * Tarzan — Gmail-side scheduler + "Update" reply listener.
 *
 * This Google Apps Script is the SINGLE SOURCE OF SCHEDULING for the
 * Tarzan newsletter. It owns the *when*; GitHub Actions is only the
 * *runner*. A single time-driven trigger calls tick() every few
 * minutes, and tick() does two independent jobs:
 *
 *   1. checkSchedule() — fires the daily market slots (morning / midday
 *      / close) at their Europe/Rome local time, AT MOST ONCE PER DAY
 *      PER SLOT. This replaces GitHub Actions' built-in cron, which was
 *      best-effort: it queued runs under load and released them in a
 *      burst, causing several newsletters to arrive back-to-back.
 *
 *   2. processInbox() — the original on-demand path: when you reply
 *      "Update" to a newsletter, it dispatches an extra send.
 *
 * Both jobs trigger the same GitHub workflow via a repository_dispatch
 * event (event_type: send_now). The slot name travels in
 * client_payload.label and ends up in the email subject.
 *
 * Why this fixes the duplicate/burst problem
 * ------------------------------------------
 * - Google's time-driven triggers are punctual and run in the
 *   Europe/Rome timezone, so DST is handled natively — no more
 *   winter/summer double-cron hack.
 * - Each slot is guarded by an idempotency marker keyed on
 *   (date, slot) stored in Script Properties. Polling jitter, retries,
 *   or two overlapping trigger runs can never send the same slot twice.
 * - A slot only fires inside a bounded window after its target time
 *   (MAX_LAG_MINUTES). If the script was down past the window, the slot
 *   is skipped rather than sent stale — we prefer "fewer than 3 today"
 *   over "3 at once at the wrong time".
 *
 * Setup
 * -----
 * 1. Open https://script.google.com and create a new project.
 * 2. Paste this file as Code.gs.
 * 3. In Project Settings, set the following Script Properties:
 *      GH_OWNER       Your GitHub username or org (e.g. "simonsa")
 *      GH_REPO        Repository name (e.g. "Tarzan-personal")
 *      GH_TOKEN       Fine-grained PAT with "Actions: read & write"
 *                     scope on this repo (no other scopes needed).
 *      LABEL_NAME     Optional Gmail label applied to processed
 *                     "Update" threads. Default "tarzan-update-handled".
 *      SUBJECT_MATCH  Optional. Default "Tarzan Portfolio Digest" — only
 *                     threads whose subject contains this string are
 *                     eligible for the "Update" reply path.
 *      WORD_MATCH     Optional. Default "update" — body must contain
 *                     this token (case-insensitive, word boundary).
 * 4. Run `installTrigger()` once. Approve the OAuth consent for
 *    Gmail + UrlFetch scopes. This installs the every-5-minute tick().
 * 5. (Optional) Run `checkSchedule()` or `processInbox()` manually to
 *    test each path on demand.
 *
 * IMPORTANT: remove the `schedule:` block from
 * .github/workflows/newsletter.yml (already done) — scheduling must
 * live in exactly one place, here.
 */

// ---------------------------------------------------------------------------
// Scheduling configuration
// ---------------------------------------------------------------------------

// The timezone all slot times are expressed in. zoneinfo/Java handle
// DST automatically, so a slot fires at the same wall-clock time
// year-round.
const SCHEDULE_TZ = 'Europe/Rome';

// Daily market slots. Times are Europe/Rome local, aligned to Borsa
// Italiana hours (open ~09:00, close 17:30):
//   morning  09:05  — just after the open, sent every day
//   midday   13:00  — mid-session, weekdays only
//   close    17:35  — just after the close, weekdays only
// Edit these to taste; the rest of the logic adapts automatically.
const SLOTS = [
  { name: 'morning', hour: 9,  minute: 5,  weekdaysOnly: false },
  { name: 'midday',  hour: 13, minute: 0,  weekdaysOnly: true },
  { name: 'close',   hour: 17, minute: 35, weekdaysOnly: true },
];

// A slot may fire only within this many minutes after its target time.
// Past the window it is skipped (avoids stale, bursty sends if the
// trigger was delayed or the script was paused). With a 5-minute poll,
// 90 minutes leaves ample room to catch every slot punctually.
const MAX_LAG_MINUTES = 90;

// Script Property key prefix for per-(date, slot) idempotency markers.
const SENT_MARKER_PREFIX = 'sent:';

// Markers older than this many days are pruned on each run so Script
// Properties don't grow unbounded.
const MARKER_RETENTION_DAYS = 3;

// ---------------------------------------------------------------------------
// On-demand "Update" reply configuration
// ---------------------------------------------------------------------------

const DEFAULT_LABEL = 'tarzan-update-handled';
const DEFAULT_SUBJECT_MATCH = 'Tarzan Portfolio Digest';
const DEFAULT_WORD_MATCH = 'update';
// How far back we scan for new "Update" replies on each run. 1 day
// is plenty given the 5-minute polling cadence and is a safety net
// in case a trigger run fails for a few hours.
const SEARCH_WINDOW_DAYS = 1;

// ---------------------------------------------------------------------------
// Trigger entry point
// ---------------------------------------------------------------------------

/**
 * Single time-driven entry point. Runs the scheduler first, then the
 * on-demand inbox scan. Each is independent and failure-isolated so a
 * problem in one path never blocks the other.
 */
function tick() {
  try {
    checkSchedule();
  } catch (err) {
    Logger.log('checkSchedule() failed: %s', err && err.stack ? err.stack : err);
  }
  try {
    processInbox();
  } catch (err) {
    Logger.log('processInbox() failed: %s', err && err.stack ? err.stack : err);
  }
}

/**
 * Install a single 5-minute time-driven trigger for tick(). Idempotent:
 * removes any existing trigger pointing at tick() (or the legacy
 * processInbox handler) first, so re-running never stacks triggers.
 */
function installTrigger() {
  const triggers = ScriptApp.getProjectTriggers();
  for (const t of triggers) {
    const fn = t.getHandlerFunction();
    if (fn === 'tick' || fn === 'processInbox') {
      ScriptApp.deleteTrigger(t);
    }
  }
  ScriptApp.newTrigger('tick')
    .timeBased()
    .everyMinutes(5)
    .create();
  Logger.log('Trigger installed: tick() runs every 5 minutes.');
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/**
 * Fire any market slot that is currently due and not yet sent today.
 *
 * "Due" means: the current Europe/Rome time is at or after the slot's
 * target time, but within MAX_LAG_MINUTES of it, weekend rules permit
 * it, and no idempotency marker exists for (today, slot). A LockService
 * lock serializes concurrent trigger runs so the check-then-send is
 * atomic.
 *
 * @return {number} how many slots were dispatched this run.
 */
function checkSchedule() {
  const props = PropertiesService.getScriptProperties();
  const owner = _required_(props, 'GH_OWNER');
  const repo = _required_(props, 'GH_REPO');
  const token = _required_(props, 'GH_TOKEN');

  const now = new Date();
  const today = _localDateStr_(now);            // yyyy-MM-dd in Rome
  const dow = parseInt(_localFormat_(now, 'u'), 10);  // 1=Mon … 7=Sun
  const isWeekend = dow >= 6;
  const nowMin = _localMinuteOfDay_(now);

  const lock = LockService.getScriptLock();
  // Wait briefly for any concurrent run; if we can't get the lock,
  // the other run is handling this tick — bail out cleanly.
  if (!lock.tryLock(20000)) {
    Logger.log('checkSchedule: could not acquire lock, another run is active.');
    return 0;
  }

  let dispatched = 0;
  try {
    _pruneMarkers_(props, now);

    for (const slot of SLOTS) {
      const decision = _slotDecision_(slot, nowMin, isWeekend, props, today);
      if (decision.fire) {
        _dispatch_(owner, repo, token, slot.name, 'scheduled:' + slot.name + ' ' + today);
        _markSent_(props, today, slot.name, now);
        dispatched += 1;
        Logger.log('Dispatched slot "%s" for %s (now=%s).', slot.name, today, _localFormat_(now, 'HH:mm'));
      } else if (decision.reason !== 'not-due') {
        Logger.log('Slot "%s" skipped: %s.', slot.name, decision.reason);
      }
    }
  } finally {
    lock.releaseLock();
  }

  return dispatched;
}

/**
 * Decide whether a single slot should fire right now. Pure-ish: the
 * only side-effect-free inputs are passed in; it reads (but never
 * writes) the sent-markers to check idempotency.
 *
 * @return {{fire: boolean, reason: string}}
 *   reason is one of: "fire", "not-due", "too-late", "weekend",
 *   "already-sent".
 */
function _slotDecision_(slot, nowMin, isWeekend, props, today) {
  const slotMin = slot.hour * 60 + slot.minute;

  if (nowMin < slotMin) {
    return { fire: false, reason: 'not-due' };
  }
  if (nowMin >= slotMin + MAX_LAG_MINUTES) {
    return { fire: false, reason: 'too-late (lag ' + (nowMin - slotMin) + 'm > ' + MAX_LAG_MINUTES + 'm)' };
  }
  if (slot.weekdaysOnly && isWeekend) {
    return { fire: false, reason: 'weekend' };
  }
  if (_alreadySent_(props, today, slot.name)) {
    return { fire: false, reason: 'already-sent' };
  }
  return { fire: true, reason: 'fire' };
}

// ---------------------------------------------------------------------------
// Idempotency markers (per date + slot), stored in Script Properties
// ---------------------------------------------------------------------------

function _markerKey_(dateStr, slotName) {
  return SENT_MARKER_PREFIX + dateStr + ':' + slotName;
}

function _alreadySent_(props, dateStr, slotName) {
  return props.getProperty(_markerKey_(dateStr, slotName)) !== null;
}

function _markSent_(props, dateStr, slotName, now) {
  props.setProperty(_markerKey_(dateStr, slotName), now.toISOString());
}

/**
 * Delete sent-markers older than MARKER_RETENTION_DAYS so Script
 * Properties stay small. Markers are keyed "sent:yyyy-MM-dd:slot".
 */
function _pruneMarkers_(props, now) {
  const cutoff = new Date(now.getTime() - MARKER_RETENTION_DAYS * 24 * 60 * 60 * 1000);
  const cutoffStr = _localDateStr_(cutoff);
  const all = props.getProperties();
  for (const key in all) {
    if (key.indexOf(SENT_MARKER_PREFIX) !== 0) continue;
    const parts = key.split(':');          // ["sent", "yyyy-MM-dd", slot]
    if (parts.length < 3) continue;
    if (parts[1] < cutoffStr) {            // lexicographic works for ISO dates
      props.deleteProperty(key);
    }
  }
}

// ---------------------------------------------------------------------------
// Europe/Rome time helpers (DST handled by the platform formatter)
// ---------------------------------------------------------------------------

function _localFormat_(date, pattern) {
  return Utilities.formatDate(date, SCHEDULE_TZ, pattern);
}

function _localDateStr_(date) {
  return _localFormat_(date, 'yyyy-MM-dd');
}

function _localMinuteOfDay_(date) {
  const h = parseInt(_localFormat_(date, 'HH'), 10);
  const m = parseInt(_localFormat_(date, 'mm'), 10);
  return h * 60 + m;
}

// ---------------------------------------------------------------------------
// On-demand "Update" reply listener
// ---------------------------------------------------------------------------

/**
 * Scans the inbox for unread "Update" replies on Tarzan threads and
 * dispatches one GitHub event per matching thread. Unchanged behavior
 * from the original listener; now invoked from tick() alongside the
 * scheduler.
 */
function processInbox() {
  const props = PropertiesService.getScriptProperties();
  const owner = _required_(props, 'GH_OWNER');
  const repo = _required_(props, 'GH_REPO');
  const token = _required_(props, 'GH_TOKEN');
  const subjectMatch = props.getProperty('SUBJECT_MATCH') || DEFAULT_SUBJECT_MATCH;
  const wordMatch = (props.getProperty('WORD_MATCH') || DEFAULT_WORD_MATCH).toLowerCase();
  const labelName = props.getProperty('LABEL_NAME') || DEFAULT_LABEL;

  const label = _ensureLabel_(labelName);

  // Gmail search query: replies (in:inbox), subject contains the marker,
  // newer than the window, and NOT yet labelled as handled.
  const query =
    'in:inbox' +
    ' subject:"' + subjectMatch + '"' +
    ' newer_than:' + SEARCH_WINDOW_DAYS + 'd' +
    ' -label:' + labelName.replace(/\s+/g, '-');

  const threads = GmailApp.search(query, 0, 50);
  Logger.log('Found %s candidate threads', threads.length);

  let dispatched = 0;
  for (const thread of threads) {
    if (_threadHasUpdateRequest_(thread, wordMatch)) {
      _dispatch_(owner, repo, token, 'on-demand', _summarize_(thread));
      thread.addLabel(label);
      dispatched += 1;
    }
  }
  Logger.log('Dispatched %s newsletter request(s)', dispatched);
  return dispatched;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _required_(props, key) {
  const value = props.getProperty(key);
  if (!value) {
    throw new Error(
      'Missing Script Property "' + key + '". ' +
      'Set it in Project Settings > Script Properties.'
    );
  }
  return value;
}

function _ensureLabel_(name) {
  let label = GmailApp.getUserLabelByName(name);
  if (!label) {
    label = GmailApp.createLabel(name);
  }
  return label;
}

/**
 * Returns true if any message in the thread (other than the original
 * Tarzan newsletter) contains the trigger word in its plain-text body.
 * This filters out the original outbound newsletters (which contain
 * neither "Update" body nor are addressed to Tarzan from the user).
 */
function _threadHasUpdateRequest_(thread, wordMatch) {
  const messages = thread.getMessages();
  // The original newsletter is the first message in the thread; replies
  // are all messages after it. We only consider messages from the
  // current user (i.e. replies they sent themselves).
  if (messages.length < 2) return false;
  const myAddress = Session.getActiveUser().getEmail().toLowerCase();
  const tokenRegex = new RegExp('\\b' + wordMatch + '\\b', 'i');
  for (let i = 1; i < messages.length; i += 1) {
    const m = messages[i];
    const sender = (m.getFrom() || '').toLowerCase();
    if (sender.indexOf(myAddress) === -1) continue;
    const body = m.getPlainBody() || '';
    if (tokenRegex.test(body)) {
      return true;
    }
  }
  return false;
}

function _summarize_(thread) {
  const subject = thread.getFirstMessageSubject() || '(no subject)';
  return subject.slice(0, 80);
}

/**
 * POST a repository_dispatch event to GitHub. The workflow file at
 * .github/workflows/newsletter.yml listens for `event_type: send_now`
 * and runs the pipeline immediately.
 *
 * @param {string} dispatchLabel  value placed in client_payload.label,
 *   surfaced in the email subject ("morning"/"midday"/"close" for
 *   scheduled slots, "on-demand" for an "Update" reply).
 * @param {string} summary  short human context for the logs.
 */
function _dispatch_(owner, repo, token, dispatchLabel, summary) {
  const url =
    'https://api.github.com/repos/' + owner + '/' + repo + '/dispatches';
  const payload = {
    event_type: 'send_now',
    client_payload: {
      label: dispatchLabel,
      origin: 'gmail-apps-script',
      thread_subject: summary,
      timestamp: new Date().toISOString(),
    },
  };
  const response = UrlFetchApp.fetch(url, {
    method: 'post',
    contentType: 'application/json',
    headers: {
      Accept: 'application/vnd.github+json',
      Authorization: 'Bearer ' + token,
      'X-GitHub-Api-Version': '2022-11-28',
    },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true,
  });
  const code = response.getResponseCode();
  if (code !== 204) {
    Logger.log(
      'GitHub dispatch failed: HTTP %s\nBody: %s',
      code, response.getContentText().slice(0, 500)
    );
    throw new Error('GitHub dispatch failed with HTTP ' + code);
  }
  Logger.log('GitHub dispatch OK (label=%s): %s', dispatchLabel, summary);
}
