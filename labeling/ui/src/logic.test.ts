import { describe, expect, it } from 'vitest';
import { actionToLabel, describeSuggestion, formatAccuracy, keyToAction, segments } from './logic';

describe('segments', () => {
  it('marks spans and keeps the rest', () => {
    expect(segments('on Suboxone after detox', [[3, 11], [18, 23]])).toEqual([
      { text: 'on ', mark: false },
      { text: 'Suboxone', mark: true },
      { text: ' after ', mark: false },
      { text: 'detox', mark: true },
    ]);
  });

  it('handles unsorted, overlapping and empty spans', () => {
    const text = 'withdrawal cravings';
    const out = segments(text, [[11, 19], [0, 10], [5, 12], [3, 3]]);
    expect(out.map((s) => s.text).join('')).toBe(text);
    expect(out.filter((s) => s.mark).map((s) => s.text)).toEqual(['withdrawal', ' c', 'ravings']);
  });

  it('returns the whole text when nothing is highlighted', () => {
    expect(segments('plain review', [])).toEqual([{ text: 'plain review', mark: false }]);
  });
});

describe('shortcuts', () => {
  it('maps keys to actions unless the user is typing', () => {
    expect(keyToAction('1', false)).toBe('relevant');
    expect(keyToAction('0', false)).toBe('not_relevant');
    expect(keyToAction('S', false)).toBe('skip');
    expect(keyToAction('1', true)).toBeNull();
    expect(keyToAction('x', false)).toBeNull();
  });

  it('turns actions into API labels', () => {
    expect([actionToLabel('relevant'), actionToLabel('not_relevant'), actionToLabel('skip')]).toEqual([1, 0, 'skip']);
  });
});

describe('text', () => {
  it('describes suggestions and accuracy', () => {
    expect(describeSuggestion({ label: 1, confidence: 0.824 })).toBe('Model suggests: SUD-relevant (82% confident)');
    expect(describeSuggestion(null)).toContain('No model yet');
    expect(formatAccuracy(0.667)).toBe('67%');
    expect(formatAccuracy(null)).toBe('—');
  });
});
