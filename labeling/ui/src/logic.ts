// Pure helpers for the labeling UI (unit-tested in logic.test.ts).

export type Action = 'relevant' | 'not_relevant' | 'skip';

export interface Segment {
  text: string;
  mark: boolean;
}

/** Split text into plain and highlighted segments. Spans are [start, end) and may arrive unsorted or overlapping. */
export function segments(text: string, spans: number[][]): Segment[] {
  const sorted = [...spans].filter(([a, b]) => b > a).sort((x, y) => x[0] - y[0]);
  const out: Segment[] = [];
  let at = 0;
  for (const [a, b] of sorted) {
    const start = Math.max(a, at);
    if (start >= b) continue;
    if (start > at) out.push({ text: text.slice(at, start), mark: false });
    out.push({ text: text.slice(start, b), mark: true });
    at = b;
  }
  if (at < text.length) out.push({ text: text.slice(at), mark: false });
  return out;
}

/** Keyboard shortcuts: 1 = relevant, 0 = not relevant, s = skip. Ignored while typing in a field. */
export function keyToAction(key: string, targetIsInput: boolean): Action | null {
  if (targetIsInput) return null;
  if (key === '1') return 'relevant';
  if (key === '0') return 'not_relevant';
  if (key === 's' || key === 'S') return 'skip';
  return null;
}

export function actionToLabel(action: Action): 0 | 1 | 'skip' {
  return action === 'relevant' ? 1 : action === 'not_relevant' ? 0 : 'skip';
}

export function describeSuggestion(s: { label: number; confidence: number } | null): string {
  if (!s) return 'No model yet: it trains after the first labels.';
  const name = s.label === 1 ? 'SUD-relevant' : 'Not relevant';
  return `Model suggests: ${name} (${Math.round(s.confidence * 100)}% confident)`;
}

export function formatAccuracy(value: number | null): string {
  return value === null ? '—' : `${Math.round(value * 100)}%`;
}
