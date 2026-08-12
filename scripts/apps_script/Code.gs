/**
 * Tarzan — Gmail-side scheduler + "Update" reply listener.
 *
 * This Google Apps Script is the SINGLE SOURCE OF SCHEDULING for the
 * Tarzan newsletter. It owns the *when*; GitHub Actions is only the
 * *runner*. A single time-driven trigger calls tick() every few
 * minutes, and tick() does two independent jobs:
 *
 *   1. checkSchedule() — fires the market-hours slots at their
 *      Europe/Rome local time, AT MOST ONCE PER DAY PER SLOT. Every day
 *      opens with an 08:00 pre-open briefing and closes with a 23:00
 *      recap. In between, weekdays get one digest every 90 minutes
 *      across Borsa Italiana continuous trading (09:05 up to the 17:30
 *      close) plus a post-close wrap-up at 17:35; on weekends a single
 *      midday digest (13:05) goes out. This replaces GitHub Actions' cron,
 *      which was best-effort: it queued runs under load and released
 *      them in a burst, causing several newsletters to arrive
 *      back-to-back.
 *
 *   2. processInbox() — the original on-demand path: when you reply
 *      "Update" to a newsletter, it dispatches an extra send.
 *
 * Both jobs trigger the same GitHub workflow via a repository_dispatch
 * event (event_type: send_now). The slot label travels in
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
 *   is skipped rather than sent stale — we prefer a missed hourly
 *   digest over a stale one at the wrong time.
 *
 * Setup
 * -----
 * 1. Open https://script.google.com and create a new project.
 * 2. Paste this file as Code.gs.
 * 3. In Project Settings, set the following Script Properties:
 *      GH_OWNER       Your GitHub username or org (e.g. "simonsa")
 *      GH_REPO        Repository name (e.g. "Tarzan-personal")
 *      GH_TOKEN       Fine-grained PAT on this repo with BOTH
 *                     "Contents: Read and write" AND "Actions: Read and
 *                     write". Contents R/W is required for
 *                     repository_dispatch — "Actions" alone returns
 *                     HTTP 403.
 *      LABEL_NAME     Optional Gmail label applied to processed
 *                     "Update" threads. Default "tarzan-update-handled".
 *      SUBJECT_MATCH  Optional. Default "Portfolio Digest" — only
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

// Daily market slots. Times are Europe/Rome local. Every day gets an
// 08:00 pre-open briefing and a 23:00 recap. On weekdays we also send
// one digest every 90 minutes across Borsa Italiana continuous trading
// (from just after the 09:00 open to the 17:30 close) plus a wrap-up
// just after the close; on weekends markets are closed, so a single
// midday digest goes out.
//
// Times are deliberately "off the hour" (the cadence starts at :05) for
// a tidy inbox rhythm.
//
// Each slot carries:
//   name   unique id — used for the per-(date, slot) idempotency marker,
//          so every send fires at most once per day.
//   label  shown in the email subject (need NOT be unique).
//   hour/minute  Europe/Rome local time of the slot.
//   days   'weekday' (Mon–Fri), 'weekend' (Sat–Sun), or 'all'.
const SLOT_INTERVAL_MINUTES = 90;   // 1.5h cadence between weekday sends
const SLOT_START_MINUTE = 9 * 60 + 5;   // first slot: 09:05
const MARKET_CLOSE_MINUTE = 17 * 60 + 30;   // 17:30 Borsa Italiana close

function _buildSlots_() {
  const slots = [];
  // Weekday cadence: 09:05, 10:35, 12:05, … up to (and including) the
  // last step at or before the 17:30 close.
  for (let m = SLOT_START_MINUTE; m <= MARKET_CLOSE_MINUTE; m += SLOT_INTERVAL_MINUTES) {
    const h = Math.floor(m / 60);
    const min = m % 60;
    const hh = (h < 10 ? '0' : '') + h;
    const mm = (min < 10 ? '0' : '') + min;
    slots.push({ name: 'wd-' + hh + mm, label: hh + ':' + mm, hour: h, minute: min, days: 'weekday' });
  }
  // Post-close wrap-up just after the 17:30 close.
  slots.push({ name: 'close', label: 'close', hour: 17, minute: 35, days: 'weekday' });
  // Weekend: a single midday digest.
  slots.push({ name: 'weekend', label: 'weekend', hour: 13, minute: 5, days: 'weekend' });
  // Pre-open briefing and end-of-day recap, every day of the week.
  slots.push({ name: 'preopen', label: '08:00', hour: 8, minute: 0, days: 'all' });
  slots.push({ name: 'night', label: '23:00', hour: 23, minute: 0, days: 'all' });
  return slots;
}
const SLOTS = _buildSlots_();

/**
 * Run this from the editor after editing _buildSlots_(). It asserts the
 * one invariant a new slot can silently break: two slots on the same day
 * closer together than MAX_LAG_MINUTES would both come due inside a
 * single tick and send two emails back-to-back — the exact burst this
 * scheduler exists to prevent.
 */
function validateSlots() {
  const names = {};
  for (const a of SLOTS) {
    if (names[a.name]) throw new Error('duplicate slot name: ' + a.name);
    names[a.name] = true;
    for (const b of SLOTS) {
      if (a === b) continue;
      const sameDay = a.days === b.days || a.days === 'all' || b.days === 'all';
      const gap = Math.abs((a.hour * 60 + a.minute) - (b.hour * 60 + b.minute));
      if (sameDay && gap < MAX_LAG_MINUTES) {
        throw new Error('slots "' + a.name + '" and "' + b.name + '" are ' + gap +
                        'm apart, under MAX_LAG_MINUTES (' + MAX_LAG_MINUTES + ')');
      }
    }
  }
  Logger.log('%s slots OK.', SLOTS.length);
}

// A slot may fire only within this many minutes after its target time.
// Past the window it is skipped (avoids stale, bursty sends if the
// trigger was delayed or paused). It MUST stay smaller than the gap
// between any two consecutive slots so two never fire in the same tick;
// the tightest gap is the last cadence slot (16:35) to the post-close
// wrap-up (17:35) = 60 min. 25 minutes still comfortably catches every
// slot given the 5-minute polling cadence.
const MAX_LAG_MINUTES = 25;

// Script Property key prefix for per-(date, slot) idempotency markers.
const SENT_MARKER_PREFIX = 'sent:';

// Durable delivery claims use a separate namespace and outlive the explicit
// reconciliation window. Values contain only hashed intent/control metadata.
const DELIVERY_CLAIM_PREFIX = 'delivery_claim:';
const DELIVERY_CLAIM_RETENTION_DAYS = 45;
const DELIVERY_RECONCILIATION_WINDOW_DAYS = 30;
const DELIVERY_STATE_SCHEMA_VERSION = '1.0';

// Markers older than this many days are pruned on each run so Script
// Properties don't grow unbounded.
const MARKER_RETENTION_DAYS = 3;

// ---------------------------------------------------------------------------
// On-demand "Update" reply configuration
// ---------------------------------------------------------------------------

const DEFAULT_LABEL = 'tarzan-update-handled';
// Must match the emailed subject's prefix (SUBJECT_PREFIX in the workflow =
// "Portfolio Digest", so subjects read "Portfolio Digest - HH:MM - uP&L …").
// Gmail's subject: search only scans the Subject header, so a stale value here
// matches nothing and silently breaks "Update" replies.
const DEFAULT_SUBJECT_MATCH = 'Portfolio Digest';
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
        const eventId = 'scheduled:' + today + ':' + slot.name;
        _dispatch_(owner, repo, token, slot.label, 'scheduled:' + slot.name + ' ' + today, eventId);
        _markSent_(props, today, slot.name, now);
        dispatched += 1;
        Logger.log('Dispatched slot "%s" (label=%s) for %s (now=%s).',
                   slot.name, slot.label, today, _localFormat_(now, 'HH:mm'));
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
 *   reason is one of: "fire", "not-due", "too-late", "wrong-day",
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
  // Day-of-week eligibility: 'weekday' (Mon–Fri), 'weekend' (Sat–Sun),
  // or 'all'. Markets are closed on weekends, so hourly weekday slots
  // don't run then and the weekend slot doesn't run on weekdays.
  const days = slot.days || 'all';
  if (days === 'weekday' && isWeekend) {
    return { fire: false, reason: 'wrong-day (weekday-only)' };
  }
  if (days === 'weekend' && !isWeekend) {
    return { fire: false, reason: 'wrong-day (weekend-only)' };
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

  // Gmail search query: subject contains the marker, newer than the window,
  // and NOT yet labelled as handled. Deliberately NOT scoped to in:inbox — the
  // digest is self-sent (sender == recipient), so Gmail files the thread under
  // Sent / All Mail, not the Inbox, and the "Update" reply lives there too.
  // Gmail search excludes Spam and Trash by default, and _matchingUpdateMessageId_
  // still requires a genuine user reply carrying the trigger word, so dropping
  // in:inbox cannot broaden what actually dispatches.
  const query =
    'subject:"' + subjectMatch + '"' +
    ' newer_than:' + SEARCH_WINDOW_DAYS + 'd' +
    ' -label:' + labelName.replace(/\s+/g, '-');

  const threads = GmailApp.search(query, 0, 50);
  Logger.log('Found %s candidate threads', threads.length);

  let dispatched = 0;
  for (const thread of threads) {
    const matchingMessageId = _matchingUpdateMessageId_(thread, wordMatch);
    if (matchingMessageId) {
      const eventId = 'update:' + thread.getId() + ':' + matchingMessageId;
      _dispatch_(owner, repo, token, 'on-demand', _summarize_(thread), eventId);
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
 * Returns the newest matching message ID sent by the active user, or null when
 * no such reply contains the trigger word. The original Tarzan newsletter is
 * excluded from the scan.
 */
function _matchingUpdateMessageId_(thread, wordMatch) {
  const messages = thread.getMessages();
  if (messages.length < 2) return null;
  const myAddress = Session.getActiveUser().getEmail().toLowerCase();
  const tokenRegex = new RegExp('\\b' + wordMatch + '\\b', 'i');
  for (let i = messages.length - 1; i >= 1; i -= 1) {
    const m = messages[i];
    const sender = (m.getFrom() || '').toLowerCase();
    if (sender.indexOf(myAddress) === -1) continue;
    const body = m.getPlainBody() || '';
    if (tokenRegex.test(body)) {
      return m.getId();
    }
  }
  return null;
}

function _threadHasUpdateRequest_(thread, wordMatch) {
  return _matchingUpdateMessageId_(thread, wordMatch) !== null;
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
function _dispatch_(owner, repo, token, dispatchLabel, summary, eventId) {
  const url =
    'https://api.github.com/repos/' + owner + '/' + repo + '/dispatches';
  const payload = {
    event_type: 'send_now',
    client_payload: {
      label: dispatchLabel,
      origin: 'gmail-apps-script',
      event_id: eventId,
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


// ---------------------------------------------------------------------------
// Durable delivery-claim web endpoint
// ---------------------------------------------------------------------------

/**
 * Authenticated transactional claim endpoint used by the final workflow step.
 * Deploy this script as a web app and set CLAIM_SERVICE_TOKEN in Script
 * Properties. Requests and stored records contain no recipient or portfolio
 * data; only logical hashes, purpose, state, and control timestamps persist.
 */
function doPost(e) {
  try {
    const request = JSON.parse((e && e.postData && e.postData.contents) || '{}');
    const props = PropertiesService.getScriptProperties();
    const expectedToken = _required_(props, 'CLAIM_SERVICE_TOKEN');
    if (!_secureEqual_(String(request.auth_token || ''), expectedToken)) {
      return _claimResponse_({ ok: false, error_code: 'AUTHENTICATION_FAILED' });
    }
    if (request.state_schema_version !== DELIVERY_STATE_SCHEMA_VERSION) {
      return _claimResponse_({ ok: false, error_code: 'STATE_SCHEMA_MISMATCH' });
    }

    const lock = LockService.getScriptLock();
    if (!lock.tryLock(20000)) {
      return _claimResponse_({ ok: false, error_code: 'LOCK_UNAVAILABLE' });
    }
    try {
      _pruneDeliveryClaims_(props, new Date());
      if (request.action === 'claim') {
        return _claimResponse_(_createDeliveryClaim_(props, request));
      }
      if (request.action === 'transition') {
        return _claimResponse_(_transitionDeliveryClaim_(props, request));
      }
      return _claimResponse_({ ok: false, error_code: 'UNKNOWN_ACTION' });
    } finally {
      lock.releaseLock();
    }
  } catch (err) {
    // Do not reflect request bodies, credentials, or raw exception chains.
    return _claimResponse_({ ok: false, error_code: 'MALFORMED_OR_INTERNAL_ERROR' });
  }
}

function _claimResponse_(value) {
  value.retention_days = DELIVERY_CLAIM_RETENTION_DAYS;
  value.reconciliation_window_days = DELIVERY_RECONCILIATION_WINDOW_DAYS;
  return ContentService
    .createTextOutput(JSON.stringify(value))
    .setMimeType(ContentService.MimeType.JSON);
}

function _secureEqual_(left, right) {
  const l = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, left);
  const r = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, right);
  if (l.length !== r.length) return false;
  let different = 0;
  for (let i = 0; i < l.length; i += 1) different |= (l[i] ^ r[i]);
  return different === 0;
}

function _deliveryClaimKey_(logicalId) {
  if (!/^[0-9a-f]{64}$/.test(String(logicalId || ''))) {
    throw new Error('invalid logical id');
  }
  return DELIVERY_CLAIM_PREFIX + logicalId;
}

function _createDeliveryClaim_(props, request) {
  if (!/^[0-9a-f]{64}$/.test(String(request.intent_digest || ''))) {
    return { ok: false, error_code: 'INVALID_INTENT_DIGEST' };
  }
  if (request.purpose !== 'NORMAL_NEWSLETTER' &&
      request.purpose !== 'CRITICAL_FAILURE_NOTIFICATION') {
    return { ok: false, error_code: 'INVALID_PURPOSE' };
  }
  const key = _deliveryClaimKey_(request.logical_id);
  const existingRaw = props.getProperty(key);
  if (existingRaw !== null) {
    const existing = JSON.parse(existingRaw);
    const conflict = existing.intent_digest !== request.intent_digest;
    return {
      ok: true,
      created: false,
      duplicate: !conflict,
      conflict: conflict,
      state: existing.state,
    };
  }
  const now = new Date().toISOString();
  const record = {
    schema_version: DELIVERY_STATE_SCHEMA_VERSION,
    intent_digest: request.intent_digest,
    purpose: request.purpose,
    state: 'CLAIMED',
    created_at: now,
    updated_at: now,
  };
  props.setProperty(key, JSON.stringify(record));
  return {
    ok: true,
    created: true,
    duplicate: false,
    conflict: false,
    state: record.state,
  };
}

function _transitionDeliveryClaim_(props, request) {
  const key = _deliveryClaimKey_(request.logical_id);
  const raw = props.getProperty(key);
  if (raw === null) return { ok: false, error_code: 'CLAIM_NOT_FOUND' };
  const record = JSON.parse(raw);
  const expected = Array.isArray(request.expected) ? request.expected : [];
  const target = String(request.target || '');
  if (expected.indexOf(record.state) === -1) {
    if (record.state === target) {
      return { ok: true, state: record.state };
    }
    return { ok: false, error_code: 'CONDITIONAL_TRANSITION_FAILED' };
  }
  const allowed = {
    CLAIMED: ['SMTP_INVOCATION_STARTED', 'DEFINITE_PRE_SEND_FAILURE'],
    SMTP_INVOCATION_STARTED: ['ACKNOWLEDGED_SUCCESS', 'UNCERTAIN'],
    ACKNOWLEDGED_SUCCESS: [],
    DEFINITE_PRE_SEND_FAILURE: [],
    UNCERTAIN: [],
  };
  if (!allowed[record.state] || allowed[record.state].indexOf(target) === -1) {
    return { ok: false, error_code: 'INVALID_TRANSITION' };
  }
  record.state = target;
  record.updated_at = new Date().toISOString();
  props.setProperty(key, JSON.stringify(record));
  return { ok: true, state: record.state };
}

function _pruneDeliveryClaims_(props, now) {
  const cutoff = now.getTime() - DELIVERY_CLAIM_RETENTION_DAYS * 24 * 60 * 60 * 1000;
  const values = props.getProperties();
  for (const key in values) {
    if (key.indexOf(DELIVERY_CLAIM_PREFIX) !== 0) continue;
    try {
      const record = JSON.parse(values[key]);
      if (new Date(record.created_at).getTime() < cutoff) props.deleteProperty(key);
    } catch (err) {
      // Corrupt control records are removed while holding the script lock.
      props.deleteProperty(key);
    }
  }
}
