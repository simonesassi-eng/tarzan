/* Executes the real Update listener; no mailbox, OAuth token or dispatch is live. */
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const vm = require('node:vm');

let now = Date.parse('2026-10-21T16:00:00Z');
class ClockDate extends Date {
  constructor(...args) { super(...(args.length ? args : [now])); }
  static now() { return now; }
}
const owner = 'owner@example.invalid';
let effective = owner;
let profileOwner = owner;
let profileFailure = false;
let nativeFailure = false;
let lockAvailable = true;
let threads = [];
let releases = 0;
const values = new Map();
const native = new Map();
const dispatches = [];
const searches = [];
const props = {
  getProperty: key => values.has(key) ? values.get(key) : null,
  setProperty: (key, value) => { values.set(key, value); },
  getProperties: () => Object.fromEntries(values),
  deleteProperty: key => { values.delete(key); },
};
const gmail = {
  Users: {
    getProfile: () => {
      if (profileFailure) throw new Error('OAuth denied');
      return {emailAddress: profileOwner};
    },
    Messages: {
      get: (mailbox, id, options) => {
        assert.equal(mailbox, 'me');
        assert.equal(options.format, 'metadata');
        assert.ok(options.fields.includes('labelIds'));
        if (nativeFailure) throw new Error('native verification unavailable');
        return native.get(id);
      },
    },
  },
};
const context = vm.createContext({
  Date: ClockDate,
  Session: {getEffectiveUser: () => ({getEmail: () => effective})},
  Gmail: gmail,
  GmailApp: {
    getUserLabelByName: name => ({name}),
    createLabel: name => ({name}),
    search: (query, start, count) => {
      searches.push({query, start, count});
      const found = query.startsWith('label:') ? threads.filter(t => t.legacy) : threads;
      return found.slice(start, start + count);
    },
  },
  PropertiesService: {getScriptProperties: () => props},
  LockService: {
    getScriptLock: () => ({
      tryLock: () => lockAvailable,
      releaseLock: () => { releases++; },
    }),
  },
  UrlFetchApp: {
    fetch: (url, options) => {
      assert.ok(url.endsWith('/dispatches'));
      dispatches.push(JSON.parse(options.payload));
      return {getResponseCode: () => 204};
    },
  },
  Utilities: {
    DigestAlgorithm: {SHA_256: 'sha256'},
    computeDigest: (algorithm, value) => [...crypto.createHash(algorithm).update(value).digest()],
  },
  Logger: {log: () => {}},
});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);

function reset() {
  values.clear();
  values.set('GH_OWNER', 'synthetic-owner');
  values.set('GH_REPO', 'synthetic-repo');
  values.set('GH_TOKEN', 'synthetic-credential');
  values.set('update_protocol_version', '2');
  native.clear();
  dispatches.length = 0;
  searches.length = 0;
  threads = [];
  effective = owner;
  profileOwner = owner;
  profileFailure = nativeFailure = false;
  lockAvailable = true;
  context.Gmail = gmail;
}
function message(id, {from = owner, body = 'Update', sent = true, age = 1000,
                       nativeFrom = from, headers = []} = {}) {
  native.set(id, {
    id, labelIds: sent ? ['SENT'] : ['INBOX'],
    internalDate: String(now - age),
    payload: {headers: [{name: 'From', value: nativeFrom}, ...headers]},
  });
  return {getId: () => id, getFrom: () => from, getPlainBody: () => body};
}
function thread(replies, options = {}) {
  const original = {getId: () => 'original', getFrom: () => owner,
                    getPlainBody: () => 'Original digest: reply Update for a fresh issue'};
  return {
    legacy: !!options.legacy,
    getId: () => options.id || 'thread-id',
    getMessages: () => [original, ...replies],
    getFirstMessageSubject: () => 'PRIVATE Portfolio Digest - €123,456.78', // synthetic privacy trap
    addLabel: () => {
      if (options.labelFailure) throw new Error('informational label failure');
    },
  };
}
function denied(options) {
  reset();
  threads = [thread([message('reply', options)])];
  assert.equal(context.processInbox(), 0);
  assert.equal(dispatches.length, 0);
}

denied({from: '"owner@example.invalid" <outsider@example.invalid>'});
denied({from: 'Owner <owner@example.invalid>', sent: false}); // forged From, received natively
denied({from: 'owner@example.invalid.evil.invalid'});
denied({body: 'I wrote no command.\n> Update'});
denied({body: 'Thanks.\nOn Wed, Owner wrote:\nUpdate'});
denied({body: 'Grazie.\nIl 21 ottobre Owner ha scritto:\nUpdate'});
denied({body: '---------- Messaggio inoltrato ----------\nUpdate'});
denied({body: 'updated'});
denied({age: 46 * 60000});
denied({age: -3 * 60000});
denied({headers: [{name: 'Auto-Submitted', value: 'auto-generated'}]});
denied({headers: [{name: 'X-Tarzan-Message-Type', value: 'newsletter'}]});
denied({nativeFrom: 'outsider@example.invalid'});
denied({headers: [{name: 'From', value: owner}]}); // duplicate native header fails closed

for (const fault of ['empty-owner', 'profile-mismatch', 'profile-denied', 'missing-service', 'native-denied']) {
  reset();
  threads = [thread([message('reply')])];
  if (fault === 'empty-owner') effective = '';
  if (fault === 'profile-mismatch') profileOwner = 'outsider@example.invalid';
  if (fault === 'profile-denied') profileFailure = true;
  if (fault === 'missing-service') context.Gmail = undefined;
  if (fault === 'native-denied') nativeFailure = true;
  assert.equal(context.processInbox(), 0, fault);
  assert.equal(dispatches.length, 0, fault);
}

reset();
threads = [thread([message('first', {from: 'Owner <owner@example.invalid>', body: 'Please UPDATE.'})])];
assert.equal(context.processInbox(), 1);
assert.equal(context.processInbox(), 0);
threads[0] = thread([message('first'), message('second')]);
assert.equal(context.processInbox(), 1); // later reply in the same labelled conversation
assert.equal(context.processInbox(), 0);
assert.equal(dispatches.length, 2);
assert.notEqual(dispatches[0].client_payload.event_id, dispatches[1].client_payload.event_id);
assert.match(dispatches[0].client_payload.event_id, /^update:[a-f0-9]{64}$/);
assert.equal(dispatches[0].client_payload.timestamp, new Date(now - 1000).toISOString());
const encoded = JSON.stringify(dispatches);
for (const secret of ['PRIVATE', '123,456', 'thread_subject', owner, 'thread-id', 'first', 'second']) {
  assert.ok(!encoded.includes(secret), secret);
}
assert.ok(searches.every(s => !s.query.includes('-label:')));

reset();
values.set('WORD_MATCH', 'up.date+');
threads = [thread([message('literal', {body: 'upXdate'})])];
assert.equal(context.processInbox(), 0);
threads = [thread([message('literal', {body: 'up.date+'})])];
assert.equal(context.processInbox(), 1); // regex metacharacters are literal

reset();
const unrelated = Array.from({length: 50}, (_, i) =>
  thread([message('other-' + i, {body: 'no command'})], {id: 'thread-' + i}));
threads = [...unrelated, thread([message('page-two')])];
assert.equal(context.processInbox(), 1);
assert.ok(searches.some(s => s.start === 50));

reset();
values.delete('update_protocol_version');
threads = [thread([message('legacy')], {legacy: true})];
assert.equal(context.processInbox(), 0); // old handled threads are migrated, not resent
threads = [thread([message('legacy'), message('after-install')], {legacy: true})];
assert.equal(context.processInbox(), 1);
assert.equal(context.processInbox(), 0);

reset();
threads = [thread([message('label-failure')], {labelFailure: true})];
assert.throws(() => context.processInbox(), /label failure/);
assert.equal(dispatches.length, 1);
assert.equal(context.processInbox(), 0); // marker committed before informational label

reset();
lockAvailable = false;
threads = [thread([message('locked')])];
assert.equal(context.processInbox(), 0);
assert.equal(dispatches.length, 0);

reset();
for (let i = 0; i < 500; i++) values.set('update_message:capacity-' + i, new Date(now).toISOString());
values.set('delivery_claim:untouched', 'retained');
threads = [thread([message('capacity')])];
assert.equal(context.processInbox(), 0);
assert.equal(dispatches.length, 0);
assert.equal(values.get('delivery_claim:untouched'), 'retained');
assert.equal([...values.keys()].filter(k => k.startsWith('update_message:')).length, 500);
assert.ok(releases > 0);
console.log('Apps Script native owner/SENT, quoted-text, per-message, privacy and pagination checks passed');
