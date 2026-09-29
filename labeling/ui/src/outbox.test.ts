import { describe, expect, it } from 'vitest';
import { HttpError, dequeue, enqueue, readOutbox, retryable, withRetry, writeOutbox } from './outbox';

class MemoryStorage {
  private m = new Map<string, string>();
  getItem(k: string) { return this.m.get(k) ?? null; }
  setItem(k: string, v: string) { this.m.set(k, v); }
  removeItem(k: string) { this.m.delete(k); }
  get length() { return this.m.size; }
  clear() { this.m.clear(); }
  key() { return null; }
}

describe('outbox', () => {
  it('keeps one pending entry per item and round-trips through storage', () => {
    const s = new MemoryStorage() as unknown as Storage;
    let q = enqueue([], { item_id: 'a', label: 1, seconds: 2 });
    q = enqueue(q, { item_id: 'a', label: 0, seconds: 3 });
    q = enqueue(q, { item_id: 'b', label: 'skip', seconds: 1 });
    writeOutbox('ana', q, s);
    expect(readOutbox('ana', s)).toEqual([{ item_id: 'a', label: 0, seconds: 3 }, { item_id: 'b', label: 'skip', seconds: 1 }]);
    writeOutbox('ana', dequeue(q, 'a').filter((x) => x.item_id !== 'b'), s);
    expect(readOutbox('ana', s)).toEqual([]);
  });

  it('survives corrupt or missing storage', () => {
    const s = new MemoryStorage() as unknown as Storage;
    s.setItem('cliniq-outbox:ana', '{not json');
    expect(readOutbox('ana', s)).toEqual([]);
    expect(readOutbox('ana', null)).toEqual([]);
  });
});

describe('retries', () => {
  it('retries network and server errors, not bad input or conflicts', () => {
    expect(retryable(new TypeError('Failed to fetch'))).toBe(true);
    expect(retryable(new HttpError('boom', 503))).toBe(true);
    expect(retryable(new HttpError('conflict', 409))).toBe(false);
    expect(retryable(new HttpError('bad', 422))).toBe(false);
  });

  it('gives up after the backoff schedule', async () => {
    let calls = 0;
    const waits: number[] = [];
    const fail = async () => { calls++; throw new TypeError('offline'); };
    await expect(withRetry(fail, [1, 2], async (ms) => { waits.push(ms); })).rejects.toThrow('offline');
    expect(calls).toBe(3);
    expect(waits).toEqual([1, 2]);
  });

  it('returns as soon as an attempt succeeds and never retries a conflict', async () => {
    let calls = 0;
    const flaky = async () => { calls++; if (calls < 2) throw new TypeError('reset'); return 'ok'; };
    expect(await withRetry(flaky, [1, 1], async () => {})).toBe('ok');
    let conflictCalls = 0;
    const conflict = async () => { conflictCalls++; throw new HttpError('conflict', 409); };
    await expect(withRetry(conflict, [1, 1], async () => {})).rejects.toBeInstanceOf(HttpError);
    expect(conflictCalls).toBe(1);
  });
});
