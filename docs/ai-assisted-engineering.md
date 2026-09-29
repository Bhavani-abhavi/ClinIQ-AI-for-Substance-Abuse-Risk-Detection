# AI-assisted engineering: how changes were checked

The labeling workbench, the pilot, the bias test and the surgical box annotation were built with an AI coding
assistant (Claude Code). It drafted code, tests and write-ups; I set the goals and constraints and decided what
shipped. The rule: nothing is claimed until a test, a data check or a review supports it, and when a check
disagrees with a draft, the draft changes, not the check.

Eight real cases from this project. Each shows the change, what caught the problem, and the check that now
guards it.

## 1. The proxy label had a substring bug that only a human reviewer exposed

- **Change:** the project's proxy label marks a review relevant if its condition or drug name contains a
  substance-use keyword, matched as a substring. Every benchmark (rules, DistilBERT, active learning, bias) was
  built on it, and all their tests passed.
- **Caught by:** the assisted-vs-manual labeling pilot. I labeled 60 reviews and disagreed with the proxy on
  about 40%. Reading the disagreements showed a pattern: "meth" matched inside methylphenidate,
  sulfamethoxazole, methylprednisolone, dextromethorphan and indomethacin.
- **Scale:** across the dataset, 803 of 3,316 proxy-positive reviews (24%) were positive only because of that
  substring.
- **Fix:** "meth" must be a whole word. Other keywords stay substrings, so methamphetamine, methadone and
  dextroamphetamine still count. The rules baseline got the same fix so comparisons stay fair, and every
  benchmark was rerun on the corrected labels.
- **Check:** `test_meth_matches_only_as_a_whole_word`.
- **Lesson:** tests confirmed the code did what it said; only a person reading the data showed that what it
  said was wrong.

## 2. A double press labeled a review nobody had seen

- **Change:** the labeling UI took a keyboard label as soon as the next review appeared.
- **Caught by:** the same pilot. 4 of 60 labels took 0.0 to 0.1 s and arrived 50 to 80 ms after the previous
  one, so a double press or key repeat had labeled the next review unseen.
- **Fix:** key repeat is ignored, and no label or save counts in the first 300 ms after an item appears. The
  box tool's `Enter` got the same guard. The pilot report excludes labels under 0.3 s and lists them.
- **Check:** Playwright tests for a double press and for a held key (synthetic repeat events), on three
  browsers.

## 3. Unlocked database reads corrupted results under load

- **Change:** the workbench kept one SQLite connection for all server threads. Saves took a lock; the quality
  stats, conflict list and export read without it.
- **Caught by:** the Playwright suite running 17 tests in parallel. One server crashed with a malformed row
  while the UI requested stats and conflicts in parallel with saves. An earlier one-off Firefox failure was
  probably the same bug.
- **Fix:** every database path holds the lock (`@_locked` on the store's methods), in the workbench, the pilot
  and the surgical annotation session.
- **Check:** `test_concurrent_requests_share_one_connection_safely` runs annotators and readers on threads at
  once. With the lock removed it failed 5 of 5 runs; with it, 0 of 5.

## 4. A metric that measured the threshold, not the learning

- **Change:** the first active-learning simulation reported F1 at a 0.5 threshold. Random sampling scored F1
  0.0 at 500 labels.
- **Caught by:** checking whether 0.0 was plausible before using it. The predicted probabilities showed small
  models put almost every test review below 0.5, because the pool is 6% positive and the test set 50%. With a
  tuned threshold the same 500 labels gave F1 of about 0.69 to 0.71.
- **Fix:** average precision (threshold-free) became the headline, with the reason written into the code and
  README.
- **Check:** `outputs/active_learning.json` reports AP, AUROC and positives found. The simulation uses 10
  matched seeds after an earlier 5-seed run disagreed with itself by 0.04 AP.

## 5. A bias result that was partly a truncation effect

- **Change:** the counterfactual bias test showed identity statements flipping up to 9.0% of DistilBERT
  predictions.
- **Caught by:** asking what else a prefix changes. The model reads 128 tokens, 266 of the 600 reviews are
  longer, and a prefix pushes their endings out of view.
- **Fix:** neutral control prefixes ("As a person, ...") were added. They flip 2.2 to 2.8% on their own, so
  the identity effect is the excess: up to 6.2 points, not 9.0. Counterfactual augmentation cut the worst case
  to 3.0% at similar F1 (0.873 to 0.871). That still misses the 2% bar set before running, and it's reported
  as a fail. Rerun on the corrected labels (case 1): the worst identity flips 3.7% against 2.0% for the control,
  augmentation cuts it to 2.2% (F1 0.914 to 0.911), and it still fails.
- **Check:** `outputs/bert_bias.json` and `outputs/bert_bias_cda.json` both include the controls;
  `tests/test_bert_bias.py`.

## 6. A write-up claim the data didn't support

- **Change:** a README draft said identity flips went "almost all toward 'not relevant'".
- **Caught by:** counting before committing. 196 of 318 flips (62%) went that way; only "gay man" was lopsided
  (51 of 54).
- **Fix:** the README gives both numbers.
- **Related:** an independent review of the resumes flagged "unchanged F1" as too exact (0.873 vs 0.871) and
  asked that the 7.3x label saving (7.7x after the label fix) be called a proxy-label simulation. Both were changed everywhere.

## 7. Fast mouse drags lost their last segment

- **Change:** the box editor updated a box on each pointer-move event and stopped on pointer-up.
- **Caught by:** drawing boxes on real JIGSAWS frames in a browser. A quick drag ended short of where the mouse
  was released, because the release point was never applied.
- **Fix:** pointer-up applies the final position.
- **Check:** the Playwright flow draws, moves and resizes boxes on Chromium, Firefox and WebKit. Assertions
  compare against what the server stored, because the browsers round mouse positions a pixel differently.

## 8. A test expectation that was wrong, not the code

- **Change:** a test expected 6 of 7 pre-labels accepted after the annotator moved one box 45 px.
- **Caught by:** the test failing with 5. Tracing it showed the model refits after every save, so the edited
  frame legitimately shifted the next prediction. That is the intended behaviour.
- **Fix:** the edit moved to the last frame. The test now checks acceptance without depending on later refits.

## What this workflow needs from a person

- Keep asking whether a number is plausible before it goes anywhere.
- Prefer a failing test to a confident explanation.
- Treat an independent review's wording fixes as seriously as bugs.
- Most of the cases above were caught by checks written first, or by looking at the data, not by rereading
  code.
