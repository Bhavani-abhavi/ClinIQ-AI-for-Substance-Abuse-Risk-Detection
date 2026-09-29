// Pending label submissions survive failed requests and page reloads (unit-tested in outbox.test.ts).

export interface Pending {
  item_id: string;
  label: 0 | 1 | 'skip';
  seconds: number;
}

export class HttpError extends Error {
  constructor(message: string, readonly status: number, readonly detail?: unknown) {
    super(message);
  }
}

/** Network failures and server errors are worth retrying; 4xx answers (bad input, conflicts) are not. */
export function retryable(err: unknown): boolean {
  return !(err instanceof HttpError) || err.status >= 500;
}

export const BACKOFF_MS = [250, 750, 2000];

export async function withRetry<T>(fn: () => Promise<T>, delays: number[] = BACKOFF_MS,
                                   sleep = (ms: number) => new Promise((r) => setTimeout(r, ms)),
                                   onRetry?: (attempt: number) => void): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (!retryable(err) || attempt >= delays.length) throw err;
      onRetry?.(attempt + 1);
      await sleep(delays[attempt]);
    }
  }
}

const key = (annotator: string) => `cliniq-outbox:${annotator}`;

export function readOutbox(annotator: string, storage: Storage | null = safeStorage()): Pending[] {
  try {
    return JSON.parse(storage?.getItem(key(annotator)) ?? '[]') as Pending[];
  } catch {
    return [];
  }
}

export function writeOutbox(annotator: string, items: Pending[], storage: Storage | null = safeStorage()): void {
  try {
    if (items.length) storage?.setItem(key(annotator), JSON.stringify(items));
    else storage?.removeItem(key(annotator));
  } catch {
    /* storage unavailable: pending saves live only in memory for this page */
  }
}

/** One entry per item: a newer choice for the same item replaces the queued one. */
export function enqueue(items: Pending[], p: Pending): Pending[] {
  return [...items.filter((x) => x.item_id !== p.item_id), p];
}

export function dequeue(items: Pending[], itemId: string): Pending[] {
  return items.filter((x) => x.item_id !== itemId);
}

function safeStorage(): Storage | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage;
  } catch {
    return null;
  }
}
