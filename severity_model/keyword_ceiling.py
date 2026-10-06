"""
Diagnostic, not a model. The severity labels come from PHMSA's structured flags
(FATALITY_IND, INJURY_IND, ...), and a narrative may never mention what the flag
records. This estimates how much of each class is even visible in the text:
what share of critical narratives contain a death word, severe ones an injury
word, and how often minor narratives contain them too (after a rough negation
strip). The word lists are my guesses, so treat the numbers as approximate.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
data = json.load(open(ROOT / "data/processed/severity_colab_export.json", encoding="utf-8"))
LABELS = data["labels"]
rows = data["train"] + data["dev"] + data["test"]      # descriptive only, nothing is tuned on it

DEATH = re.compile(r"\b(fatal\w*|death\w*|died|dies|dead|killed|deceased|passed away|"
                   r"lost (?:his|her|their) li(?:fe|ves)|succumbed)\b", re.I)
INJURY = re.compile(r"\b(injur\w*|hospital\w*|ambulance|paramedic\w*|treated for)\b", re.I)
FIRE = re.compile(r"\b(ignit\w*|explo\w*|fireball|flash fire|burn\w*|fire)\b", re.I)
# drop "no ...", "not ...", "without ..." up to the next comma/period before matching
NEGATION = re.compile(r"\b(no|not|without|zero|neither|nor|never)\b[^.;,]{0,40}", re.I)


def has(pattern, text):
    return bool(pattern.search(NEGATION.sub(" ", text)))


print(f"{'class':<10}{'n':>6}   {'death word':>10}{'injury word':>13}{'fire word':>11}")
for k, label in enumerate(LABELS):
    texts = [r["text"] for r in rows if r["label"] == k]
    d = sum(has(DEATH, t) for t in texts) / len(texts)
    i = sum(has(INJURY, t) for t in texts) / len(texts)
    f = sum(has(FIRE, t) for t in texts) / len(texts)
    print(f"{label:<10}{len(texts):>6}   {d:>10.0%}{i:>13.0%}{f:>11.0%}")

crit = [r["text"] for r in rows if r["label"] == 3 and not has(DEATH, r["text"])]
print(f"\n{len(crit)} critical narratives contain NO death word. First 8, first 250 characters:")
for t in crit[:8]:
    print(f"\n- {t[:250]}")