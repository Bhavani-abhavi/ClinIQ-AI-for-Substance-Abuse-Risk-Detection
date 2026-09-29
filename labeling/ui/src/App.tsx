import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type Conflict, type Stats, type Task } from './api';
import { type Action, actionToLabel, describeSuggestion, formatAccuracy, keyToAction, segments } from './logic';

const NAME_KEY = 'cliniq-annotator';

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
        <h1>ClinIQ labeling workbench</h1>
        <nav aria-label="Views">
          <button aria-pressed={tab === 'label'} onClick={() => setTab('label')}>Label</button>
          <button aria-pressed={tab === 'quality'} onClick={() => setTab('quality')}>Quality</button>
        </nav>
        <span className="who">
          Annotator: <strong data-testid="annotator">{name}</strong>{' '}
          <button className="link" onClick={() => { setName(''); setDraft(''); }}>switch</button>
        </span>
      </header>
      {tab === 'label' ? <LabelView annotator={name} /> : <QualityView reviewer={name} />}
    </main>
  );
}

function LabelView({ annotator }: { annotator: string }) {
  const [task, setTask] = useState<Task | null | undefined>(undefined);
  const [count, setCount] = useState(0);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const shownAt = useRef(performance.now());

  const load = useCallback(async () => {
    try {
      setTask(await api.task(annotator));
      shownAt.current = performance.now();
    } catch (e) {
      setError((e as Error).message);
    }
  }, [annotator]);

  useEffect(() => { void load(); }, [load]);

  const act = useCallback(async (action: Action) => {
    if (!task || busy) return;
    setBusy(true);
    setError('');
    try {
      const seconds = Math.round((performance.now() - shownAt.current) / 100) / 10;
      await api.label(annotator, task.item_id, actionToLabel(action), seconds);
      if (action !== 'skip') setCount((c) => c + 1);
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [annotator, busy, load, task]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      const action = keyToAction(e.key, t.tagName === 'INPUT' || t.tagName === 'TEXTAREA');
      if (action) {
        e.preventDefault();
        void act(action);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [act]);

  if (task === undefined) return <p aria-live="polite">Loading…</p>;
  return (
    <section className="card" aria-labelledby="task-heading">
      <p className="meta">
        <span data-testid="session-count">Labeled this session: {count}</span>
        {error && <span role="alert" className="error">{error}</span>}
      </p>
      {task === null ? (
        <p data-testid="empty">The queue is empty. Thank you.</p>
      ) : (
        <>
          <h2 id="task-heading">Is this review about substance use?</h2>
          {task.drug && <p className="drug">Drug: {task.drug}</p>}
          <blockquote data-testid="review" data-item={task.item_id}>
            {segments(task.text, task.highlights).map((s, i) => (s.mark ? <mark key={i}>{s.text}</mark> : <span key={i}>{s.text}</span>))}
          </blockquote>
          <p className="suggestion" data-testid="suggestion">{describeSuggestion(task.suggestion)}</p>
          <div className="actions">
            <button onClick={() => act('relevant')} disabled={busy}>SUD-relevant <kbd>1</kbd></button>
            <button onClick={() => act('not_relevant')} disabled={busy}>Not relevant <kbd>0</kbd></button>
            <button className="secondary" onClick={() => act('skip')} disabled={busy}>Skip <kbd>S</kbd></button>
          </div>
          <p className="hint">Highlights mark substance-use words to help reading; they are not the answer.</p>
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
      <table>
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
      </table>

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
      <table>
        <thead><tr><th>Labels</th><th>Positives</th><th>Test AP</th></tr></thead>
        <tbody>
          {stats.model_runs.map((r, i) => (
            <tr key={i}><td>{r.labels}</td><td>{r.positives}</td><td>{r.test_ap ?? '—'}</td></tr>
          ))}
        </tbody>
      </table>
      <p><a href="/api/export" download="labels.jsonl">Export resolved labels (JSONL)</a></p>
    </section>
  );
}
