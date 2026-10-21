/* Executes the production Apps Script endpoint with isolated platform doubles. */
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const vm = require('node:vm');

const values = new Map([['CLAIM_SERVICE_TOKEN', 'test-only-credential']]);
const writes = [];
const props = {
  getProperty: key => values.has(key) ? values.get(key) : null,
  setProperty: (key, value) => { writes.push(key); values.set(key, value); },
  getProperties: () => Object.fromEntries(values),
  deleteProperty: key => { values.delete(key); },
};
let lockAvailable = true;
let releases = 0;
const context = vm.createContext({
  PropertiesService: { getScriptProperties: () => props },
  LockService: {
    getScriptLock: () => ({
      tryLock: () => lockAvailable,
      releaseLock: () => { releases++; },
    }),
  },
  ContentService: {
    MimeType: { JSON: 'application/json' },
    createTextOutput: body => ({ setMimeType: () => JSON.parse(body) }),
  },
  Utilities: {
    DigestAlgorithm: { SHA_256: 'sha256' },
    computeDigest: (algorithm, value) => [...crypto.createHash(algorithm).update(value).digest()],
  },
  Logger: { log: () => {} },
});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);

const logicalId = 'a'.repeat(64);
const base = {
  auth_token: 'test-only-credential',
  state_schema_version: '1.0',
  logical_id: logicalId,
};
const post = request => context.doPost({ postData: { contents: JSON.stringify({ ...base, ...request }) } });
const create = {
  action: 'claim',
  intent_digest: 'b'.repeat(64),
  purpose: 'NORMAL_NEWSLETTER',
};
const transition = {
  action: 'transition',
  expected: ['CLAIMED'],
  target: 'SMTP_INVOCATION_STARTED',
  operation_id: 'c'.repeat(32),
};

assert.equal(post(create).created, true);
assert.equal(post(create).duplicate, true);
assert.equal(post(create).state, 'CLAIMED');
const granted = post(transition);
assert.equal(granted.ok, true);
assert.equal(granted.operation_id, transition.operation_id);
const writesBeforeRetry = writes.length;
assert.equal(post(transition).ok, true); // response lost: only the same operation may retry
assert.equal(writes.length, writesBeforeRetry);
assert.equal(
  post({ ...transition, operation_id: 'd'.repeat(32) }).error_code,
  'CONDITIONAL_TRANSITION_FAILED',
);
assert.equal(post({ ...transition, operation_id: undefined }).error_code, 'INVALID_TRANSITION_OPERATION_ID');
assert.equal(post({ ...transition, operation_id: 'invalid' }).error_code, 'INVALID_TRANSITION_OPERATION_ID');

const ack = {
  action: 'transition', expected: ['SMTP_INVOCATION_STARTED'],
  target: 'ACKNOWLEDGED_SUCCESS', operation_id: 'e'.repeat(32),
};
assert.equal(post(ack).ok, true);
assert.equal(post(ack).ok, true);
assert.equal(post(transition).error_code, 'CONDITIONAL_TRANSITION_FAILED');
assert.equal(post(create).state, 'ACKNOWLEDGED_SUCCESS');

const secondId = '1'.repeat(64);
assert.equal(post({ ...create, logical_id: secondId }).created, true);
assert.equal(
  post({ ...transition, logical_id: secondId, target: 'ACKNOWLEDGED_SUCCESS' }).error_code,
  'INVALID_TRANSITION',
);
assert.equal(post({ ...create, intent_digest: '2'.repeat(64) }).conflict, true);

const beforeRejections = writes.length;
assert.equal(post({ ...create, auth_token: '' }).error_code, 'AUTHENTICATION_FAILED');
assert.equal(post({ ...create, auth_token: 'anything-else' }).error_code, 'AUTHENTICATION_FAILED');
assert.equal(post({ ...create, state_schema_version: 'obsolete' }).error_code, 'STATE_SCHEMA_MISMATCH');
lockAvailable = false;
assert.equal(post({ ...create, logical_id: '3'.repeat(64) }).error_code, 'LOCK_UNAVAILABLE');
lockAvailable = true;
assert.equal(writes.length, beforeRejections);
assert.ok(releases > 0);

// Corruption is not permission to delete a potentially already-sent claim.
const corruptId = '4'.repeat(64);
values.set('delivery_claim:' + corruptId, '{corrupt-json');
assert.equal(
  post({ ...create, logical_id: corruptId }).error_code,
  'MALFORMED_OR_INTERNAL_ERROR',
);
assert.equal(values.get('delivery_claim:' + corruptId), '{corrupt-json');

// Well-formed expired claims are still pruned under the lock.
const expiredId = '5'.repeat(64);
values.set('delivery_claim:' + expiredId, JSON.stringify({
  created_at: '2000-01-01T00:00:00Z', state: 'ACKNOWLEDGED_SUCCESS',
}));
post({ ...create, logical_id: secondId });
assert.equal(values.has('delivery_claim:' + expiredId), false);
console.log('Apps Script claim, CAS, retry, authentication and corruption checks passed');
