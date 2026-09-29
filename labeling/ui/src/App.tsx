import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type Conflict, type Stats, type Task } from './api';
import { HttpError, type Pending, dequeue, enqueue, readOutbox, withRetry, writeOutbox } from './outbox';
import { type Action, actionToLabel, describeSuggestion, formatAccuracy, keyToAction, segments } from './logic';

const NAME_KEY = 'cliniq-annotator';
// Ignore labels this soon after a new review appears: a double press or key repeat would otherwise label an
// item nobody has seen (4 of 60 pilot labels arrived 50-80 ms after the previous one).
export const MIN_DWELL_MS = 300;

function readName(): string {
  try {
    return localStorage.getItem(NAME_KEY) ?? '';
  } catch {
    return '';
  }
}

export function App() {
  const [name, setName] = useState(readName);
  const [draft, setDraft] = useState(name);
  const [tab, setTab] = useState<'label' | 'quality'>('label');
  const [pilot, setPilot] = useState(false);

  useEffect(() => {
    api.mode().then((m) => setPilot(m?.mode === 'pilot')).catch(() => setPilot(false));
  }, []);

  const start = (e: React.FormEvent) => {
    e.preventDefault();
    const clean = draft.trim();
    if (!clean) return;
    try {
      localStorage.setItem(NAME_KEY, clean);
    } catch {
      /* storage unavailable: keep the name for this session only */
    }
    setName(clean);
  };

  if (!name) {
    return (
      <main className="shell">
        <h1>ClinIQ labeling workbench</h1>
        <form className="card" onSubmit={start}>
          <label htmlFor="annotator">Your name</label>
          <input id="annotator" value={draft} onChange={(e) => setDraft(e.target.value)} autoFocus />
          <button type="submit">Start labeling</button>
        </form>
      </main>
    );
  }

  return (
    <main className="shell">
      <header className="top">
        <h1>{pilot ? 'ClinIQ labeling pilot' : 'ClinIQ labeling workbench'}</h1>
        {!pilot && (
          <nav aria-label="Views">
            <button aria-pressed={tab === 'label'} onClick={() => setTab('label')}>Label</button>
            <button aria-pressed={tab === 'quality'} onClick={() => setTab('quality')}>Quality</button>
          </nav>
        )}
        <span className="who">
          Annotator: <strong data-testid="annotator">{name}</strong>{' '}
          <button className="link" onClick={() => { setName(''); setDraft(''); }}>switch</button>
        </span>
      </header>
      {tab === 'label' || pilot ? <LabelView annotator={name} pilot={pilot} /> : <QualityView reviewer={name} />}
    </main>
  );
}

type SaveState = { kind: 'idle' } | { kind: 'saving' } | { kind: 'retrying'; attempt: number } | { kind: 'failed'; message: string };

function LabelView({ annotator, pilot }: { annotator: string; pilot: boolean }) {
  const [task, setTask] = useState<Task | null | undefined>(undefined);
  const [count, setCount] = useState(0);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [save, setSave] = useState<SaveState>({ kind: 'idle' });
  const shownAt = useRef(performance.now());
  const busy = save.kind === 'saving' || save.kind === 'retrying';

  const load = useCallback(async () => {
    try {
      setTask(await api.task(annotator));
      shownAt.current = performance.now();
    } catch (e) {
      setError((e as Error).message);
    }
  }, [annotator]);

  /** Send one pending label; the outbox entry is removed only once the server has it (or refused it for good). */
  const deliver = useCallback(async (p: Pending) => {
    try {
      await withRetry(() => api.label(annotator, p.item_id, p.label, p.seconds), undefined, undefined,
                      (attempt) => setSave({ kind: 'retrying', attempt }));
      writeOutbox(annotator, dequeue(readOutbox(annotator), p.item_id));
      return 'saved' as const;
    } catch (e) {
      if (e instanceof HttpError && e.status === 409) {       // labeled differently before: keep the stored label
        writeOutbox(annotator, dequeue(readOutbox(annotator), p.item_id));
        setNotice(`Already labeled earlier (${e.message}); your earlier label was kept.`);
        return 'conflict' as const;
      }
      throw e;
    }
  }, [annotator]);

  // On open: resend anything a previous page could not save, then load the next task.
  useEffect(() => {
    void (async () => {
      const pending = readOutbox(annotator);
      if (pending.length) {
        setSave({ kind: 'saving' });
        try {
          for (const p of pending) await deliver(p);
          setNotice(`Saved ${pending.length} label${pending.length > 1 ? 's' : ''} left over from before.`);
          setSave({ kind: 'idle' });
        } catch (e) {
          setSave({ kind: 'failed', message: (e as Error).message });
        }
      }
      await load();
    })();
  }, [annotator, deliver, load]);

  const submit = useCallback(async (p: Pending) => {
    setSave({ kind: 'saving' });
    setError('');
    try {
      const outcome = await deliver(p);
      if (outcome === 'saved' && p.label !== 'skip') setCount((c) => c + 1);
      setSave({ kind: 'idle' });
      await load();
    } catch (e) {
      setSave({ kind: 'failed', message: (e as Error).message });
    }
  }, [deliver, load]);

  const act = useCallback(async (action: Action) => {
    if (!task || busy || save.kind === 'failed') return;   // resolve the unsaved label first
    if (performance.now() - shownAt.current < MIN_DWELL_MS) return;
    const p: Pending = { item_id: task.item_id, label: actionToLabel(action),
                         seconds: Math.round((performance.now() - shownAt.current) / 100) / 10 };
    writeOutbox(annotator, enqueue(readOutbox(annotator), p));   // survives a reload before the server answers
    await submit(p);
  }, [annotator, busy, save.kind, submit, task]);

  const retry = useCallback(async () => {
    const pending = readOutbox(annotator);
    if (!pending.length) return setSave({ kind: 'idle' });
    for (const p of pending) await submit(p);
  }, [annotator, submit]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat) return;                                   // a held key must not label several items
      const t = e.target as HTMLElement;
      const action = keyToAction(e.key, t.tagName === 'INPUT' || t.tagName === 'TEXTAREA');
      if (action === 'skip' && pilot) return;                 // every pilot item needs an answer
      if (action) {
        e.preventDefault();
        void act(action);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [act, pilot]);

  if (task === undefined) return <p aria-live="polite">Loading…</p>;
  return (
    <section className="card" aria-labelledby="task-heading">
      <p className="meta">
        <span data-testid="session-count">Labeled this session: {count}</span>
        {task?.progress && <span data-testid="pilot-progress">Item {task.progress.done + 1} of {task.progress.total}</span>}
        <span data-testid="save-state" aria-live="polite">
          {save.kind === 'saving' && 'Saving…'}
          {save.kind === 'retrying' && `Connection problem, retrying (${save.attempt})…`}
        </span>
        {error && <span role="alert" className="error">{error}</span>}
      </p>
      {save.kind === 'failed' && (
        <p role="alert" className="error" data-testid="save-failed">
          Not saved: {save.message}. Your label is kept on this device.{' '}
          <button onClick={() => void retry()}>Retry now</button>
        </p>
      )}
      {notice && <p className="meta" data-testid="notice">{notice}</p>}
      {task === null ? (
        <p data-testid="empty">The queue is empty. Thank you.</p>
      ) : (
        <>
          <h2 id="task-heading">Is this review about substance use?</h2>
          {task.drug && <p className="drug">Drug: {task.drug}</p>}
          <blockquote data-testid="review" data-item={task.item_id}>
            {segments(task.text, task.highlights).map((s, i) => (s.mark ? <mark key={i}>{s.text}</mark> : <span key={i}>{s.text}</span>))}
          </blockquote>
          <p className="suggestion" data-testid="suggestion">
            {task.condition === 'manual' ? 'Label this one on your own: no model suggestion for this item.' : describeSuggestion(task.suggestion)}
          </p>
          <div className="actions">
            <button onClick={() => act('relevant')} disabled={busy || save.kind === 'failed'}>SUD-relevant <kbd>1</kbd></button>
            <button onClick={() => act('not_relevant')} disabled={busy || save.kind === 'failed'}>Not relevant <kbd>0</kbd></button>
            {!pilot && <button className="secondary" onClick={() => act('skip')} disabled={busy || save.kind === 'failed'}>Skip <kbd>S</kbd></button>}
          </div>
          {task.condition !== 'manual' && <p className="hint">Highlights mark substance-use words to help reading; they are not the answer.</p>}
        </>
      )}
    </section>
  );
}

function QualityView({ reviewer }: { reviewer: string }) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [conflicts, setConflicts] = useState<Conflict[]>([]);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    try {
      const [s, c] = await Promise.all([api.stats(), api.conflicts()]);
      setStats(s);
      setConflicts(c ?? []);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const decide = async (itemId: string, label: 0 | 1) => {
    try {
      await api.adjudicate(itemId, label, reviewer);
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  if (!stats) return <p aria-live="polite">{error || 'Loading…'}</p>;
  const kappa = stats.agreement.cohen_kappa;
  return (
    <section className="quality">
      {error && <p role="alert" className="error">{error}</p>}
      <div className="tiles">
        <div className="tile"><strong>{stats.resolved}</strong><span>resolved of {stats.items}</span></div>
        <div className="tile"><strong data-testid="kappa">{kappa === null ? '—' : kappa.toFixed(2)}</strong><span>Cohen's kappa ({stats.agreement.double_labeled} double-labeled)</span></div>
        <div className="tile"><strong data-testid="conflict-count">{stats.conflicts}</strong><span>open conflicts</span></div>
      </div>

      <h2>Annotators</h2>
      <div className="table-wrap"><table>
        <thead><tr><th>Annotator</th><th>Labels</th><th>Gold accuracy</th><th>Median sec</th><th>Flags</th></tr></thead>
        <tbody>
          {stats.annotators.map((a) => (
            <tr key={a.annotator} data-testid={`row-${a.annotator}`}>
              <td>{a.annotator}</td>
              <td>{a.labels}</td>
              <td>{formatAccuracy(a.gold_accuracy)} <small>({a.gold_seen} gold)</small></td>
              <td>{a.median_seconds ?? '—'}</td>
              <td>{a.flags.join(', ') || 'none'}</td>
            </tr>
          ))}
        </tbody>
      </table></div>

      <h2>Conflicts</h2>
      {conflicts.length === 0 ? <p data-testid="no-conflicts">No open conflicts.</p> : conflicts.map((c) => (
        <article className="card conflict" key={c.item_id} data-testid="conflict">
          <p>{c.text}</p>
          <p className="meta">{c.votes.map((v) => `${v.annotator}: ${v.label === 1 ? 'relevant' : 'not relevant'}`).join(' · ')}</p>
          <div className="actions">
            <button onClick={() => decide(c.item_id, 1)}>Resolve as SUD-relevant</button>
            <button onClick={() => decide(c.item_id, 0)}>Resolve as not relevant</button>
          </div>
        </article>
      ))}

      <h2>Model after each retrain</h2>
      <div className="table-wrap"><table>
        <thead><tr><th>Labels</th><th>Positives</th><th>Test AP</th></tr></thead>
        <tbody>
          {stats.model_runs.map((r, i) => (
            <tr key={i}><td>{r.labels}</td><td>{r.positives}</td><td>{r.test_ap ?? '—'}</td></tr>
          ))}
        </tbody>
      </table></div>
      <p><a href="/api/export" download="labels.jsonl">Export resolved labels (JSONL)</a></p>
    </section>
  );
}
