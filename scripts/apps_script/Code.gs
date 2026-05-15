/**
 * Tarzan — Gmail "Update" reply listener.
 *
 * This Google Apps Script watches the Gmail inbox for replies that
 * contain the word "Update" (case-insensitive) on a subject originally
 * sent by the Tarzan newsletter, and triggers a repository_dispatch
 * event on GitHub so a fresh, on-demand newsletter is generated and
 * sent.
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
 *                     threads. Default "tarzan-update-handled".
 *      SUBJECT_MATCH  Optional. Default "Tarzan Portfolio Digest" — only threads
 *                     whose subject contains this string are eligible.
 *      WORD_MATCH     Optional. Default "update" — body must contain
 *                     this token (case-insensitive, word boundary).
 * 4. Run `installTrigger()` once. Approve the OAuth consent for
 *    Gmail + UrlFetch scopes.
 * 5. To test without waiting for cron, run `processInbox()`
 *    manually and reply "Update" to a previous newsletter first.
 *
 * Why this design
 * ---------------
 * - We poll Gmail every 5 min via a time-driven trigger. Apps Script
 *   does NOT support push notifications without paid Workspace.
 * - We use Gmail labels (instead of the read/unread state) to track
 *   which threads we've already handled. This avoids race conditions
 *   if you read the email yourself.
 * - We dispatch via the GitHub REST API "repository_dispatch" event.
 *   The workflow listens for `event_type: send_now` and runs the
 *   pipeline immediately.
 */

const DEFAULT_LABEL = 'tarzan-update-handled';
const DEFAULT_SUBJECT_MATCH = 'Tarzan Portfolio Digest';
const DEFAULT_WORD_MATCH = 'update';
// How far back we scan for new "Update" replies on each run. 1 day
// is plenty given the 5-minute polling cadence and is a safety net
// in case a trigger run fails for a few hours.
const SEARCH_WINDOW_DAYS = 1;

/**
 * Entry point: scans the inbox for unread "Update" replies on Tarzan
 * threads and dispatches one GitHub event per matching thread.
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
      _dispatchSendNow_(owner, repo, token, _summarize_(thread));
      thread.addLabel(label);
      dispatched += 1;
    }
  }
  Logger.log('Dispatched %s newsletter request(s)', dispatched);
  return dispatched;
}

/**
 * Install a 5-minute time-driven trigger for processInbox. Idempotent:
 * removes any existing trigger pointing at processInbox first.
 */
function installTrigger() {
  const triggers = ScriptApp.getProjectTriggers();
  for (const t of triggers) {
    if (t.getHandlerFunction() === 'processInbox') {
      ScriptApp.deleteTrigger(t);
    }
  }
  ScriptApp.newTrigger('processInbox')
    .timeBased()
    .everyMinutes(5)
    .create();
  Logger.log('Trigger installed: processInbox runs every 5 minutes.');
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
 * and triggers an on-demand run.
 */
function _dispatchSendNow_(owner, repo, token, label) {
  const url =
    'https://api.github.com/repos/' + owner + '/' + repo + '/dispatches';
  const payload = {
    event_type: 'send_now',
    client_payload: {
      label: 'on-demand',
      origin: 'gmail-apps-script',
      thread_subject: label,
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
  Logger.log('GitHub dispatch OK for thread: %s', label);
}
