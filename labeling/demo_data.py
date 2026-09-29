"""Synthetic drug reviews for the workbench demo and end-to-end tests (written for this repo, not real data)."""
from __future__ import annotations

SUD = [
    "I have been on {drug} for {n} months and my cravings for pain pills are finally under control at night.",
    "After years of heroin use, {drug} let me get through withdrawal without getting sick and stay in my program.",
    "My counselor started me on {drug} during detox and I have now been sober for {n} months with my family.",
    "The first week off oxycodone was brutal but {drug} took the edge off the withdrawal symptoms for me.",
    "I relapsed twice before trying {drug}; this time the cravings are quieter and I go to meetings every week.",
    "{drug} helps me stop drinking; the urge to have a beer after work is mostly gone after {n} weeks.",
    "I was addicted to Xanax and tapering with {drug} under my doctor made the anxiety rebound bearable for me.",
    "Coming off methadone with {drug} was slow but I have been clean for {n} months and back at my old job.",
]
PLAIN = [
    "{drug} cleared up my acne within {n} weeks, although my skin felt dry for the first few days of using it.",
    "I take {drug} for high blood pressure and my readings dropped from 150 to 128 within {n} weeks, no side effects.",
    "My migraines went from four a month to one after starting {drug}, but I gained some weight over {n} months.",
    "{drug} worked well for my seasonal allergies; the sneezing stopped and I only felt a little drowsy at first.",
    "The doctor prescribed {drug} for my thyroid and my energy came back after about {n} weeks on the right dose.",
    "I used {drug} for a sinus infection and it cleared in {n} days, though it upset my stomach every evening.",
    "{drug} helped my sleep for the first {n} weeks but then stopped working and I felt groggy every morning.",
    "My cholesterol dropped after {n} months on {drug}; the muscle aches were mild and faded after a while.",
]
SUD_DRUGS = ['Suboxone', 'Naltrexone', 'Buprenorphine', 'Vivitrol', 'Clonidine']
PLAIN_DRUGS = ['Doxycycline', 'Lisinopril', 'Topiramate', 'Cetirizine', 'Levothyroxine', 'Amoxicillin']


def demo_rows(per_template: int = 5) -> tuple[list[dict], set[str]]:
    """Return (rows, gold_ids). Every eighth item of each class is gold."""
    rows = []
    for label, templates, drugs in ((1, SUD, SUD_DRUGS), (0, PLAIN, PLAIN_DRUGS)):
        k = 0
        for t, template in enumerate(templates):
            for j in range(per_template):
                drug = drugs[(t + j) % len(drugs)]
                rows.append({'id': f'{"s" if label else "p"}{k:03d}', 'drug': drug, 'label': label,
                             'text': template.format(drug=drug, n=2 + (3 * j + t) % 11)})
                k += 1
    rows.sort(key=lambda r: r['id'][1:] + r['id'][0])       # interleave classes
    gold = {r['id'] for r in rows if int(r['id'][1:]) % 8 == 0}
    return rows, gold
