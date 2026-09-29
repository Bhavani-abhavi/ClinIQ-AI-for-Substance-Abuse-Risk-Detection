export interface Task {
  item_id: string;
  text: string;
  drug: string | null;
  suggestion: { label: number; confidence: number } | null;
  highlights: number[][];
}

export interface AnnotatorStats {
  annotator: string;
  labels: number;
  gold_seen: number;
  gold_accuracy: number | null;
  median_seconds: number | null;
  fast_labels: number;
  flags: string[];
}

export interface Stats {
  items: number;
  resolved: number;
  label_counts: Record<string, number>;
  annotators: AnnotatorStats[];
  agreement: { double_labeled: number; cohen_kappa: number | null };
  conflicts: number;
  model_runs: { labels: number; positives: number; test_ap: number | null }[];
}

export interface Conflict {
  item_id: string;
  text: string;
  votes: { annotator: string; label: number }[];
}

async function send<T>(path: string, init?: RequestInit): Promise<T | null> {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...init });
  if (res.status === 204) return null;
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  task: (annotator: string) => send<Task>(`/api/task?annotator=${encodeURIComponent(annotator)}`),
  label: (annotator: string, item_id: string, label: 0 | 1 | 'skip', seconds: number) =>
    send<{ saved?: boolean }>('/api/labels', { method: 'POST', body: JSON.stringify({ annotator, item_id, label, seconds }) }),
  stats: () => send<Stats>('/api/stats'),
  conflicts: () => send<Conflict[]>('/api/conflicts'),
  adjudicate: (item_id: string, label: 0 | 1, by: string) =>
    send<{ saved: boolean }>('/api/adjudicate', { method: 'POST', body: JSON.stringify({ item_id, label, by }) }),
};
