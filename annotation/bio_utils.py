"""
Shared char-span -> BIO-tag conversion, used by both baselines so scoring
is apples-to-apples.
"""

import spacy

_tokenizer = spacy.blank("en")

def resolve_overlaps(entities, log_drops=False, context=""):
    """
    Applies the schema's own tie-break rule (Phase 0, §1): when spans
    overlap, the narrower/more specific entity wins; the broader span is
    dropped entirely rather than partially trimmed. Greedy: sorts by span
    length ascending so narrow spans claim their text first, then keeps
    any wider span only if it doesn't collide with an already-kept one.
    """
    sorted_ents = sorted(entities, key=lambda e: (e["end"] - e["start"], e["start"]))
    kept = []
    dropped = []
    for e in sorted_ents:
        if any(e["start"] < k["end"] and e["end"] > k["start"] for k in kept):
            dropped.append(e)
            continue
        kept.append(e)

    if log_drops and dropped:
        for e in dropped:
            print(f"  [dropped overlap{(' - ' + context) if context else ''}] "
                  f"({e['label']}) '{e['text']}'")

    return sorted(kept, key=lambda e: e["start"])


def spans_to_bio(text, entities):
    """Returns (tokens, bio_tags) for one sentence."""
    entities = resolve_overlaps(entities)
    doc = _tokenizer(text)
    tags = ["O"] * len(doc)

    for ent in entities:
        start, end, label = ent["start"], ent["end"], ent["label"]
        first = True
        for i, tok in enumerate(doc):
            if tok.idx >= end:
                break
            if tok.idx + len(tok.text) <= start:
                continue
            # token overlaps the entity span
            tags[i] = f"B-{label}" if first else f"I-{label}"
            first = False

    return [t.text for t in doc], tags