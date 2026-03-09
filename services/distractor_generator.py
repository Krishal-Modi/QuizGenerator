"""Distractor Generator – produce plausible, conceptually-related wrong answers.

Contributor note
───────────────
Krish Thakkar contributed to the embedding-based distractor selection and the
semantic de-duplication pass that keeps options distinct (so learners see
plausible *alternatives*, not near-duplicates).

Design goals
────────────
1. Distractors come from the SAME topic cluster (never random noise).
2. Each distractor is semantically distinct from the correct answer AND from
   the other distractors (checked via embeddings when available).
3. Multiple strategies are layered: knowledge-graph neighbours → embedding
   similarity → context-derived phrases → rule-based fallbacks.  Each layer
   only fires when the previous one didn't produce enough candidates.
4. "Obviously wrong" options (e.g. "None of the above") are only used as a
   last resort.
"""

from __future__ import annotations
import random
import re
from typing import Dict, List, Optional

# Similarity ceiling – any candidate above this is treated as a near-duplicate.
# We keep this fairly strict so the final options don't feel like the same
# answer rephrased.
_SIM_THRESHOLD = 0.82


class DistractorGenerator:
    """Generate exactly 3 high-quality distractors for a given MCQ."""

    def __init__(self):
        self._sentence_model = None
        self._model_attempted = False

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def generate(
        self,
        correct_answer: str,
        concept: str,
        all_concepts: List[Dict],
        knowledge_graph: Dict,
        context: str,
        count: int = 3,
    ) -> List[str]:
        """
        Return exactly *count* unique distractors.

        Strategies (layered – each fills remaining slots):
        1. Knowledge-graph neighbour definitions
        2. Embedding-nearest concept descriptions
        3. Context-derived alternative phrases
        4. Rule-based variations
        """
        if not correct_answer or not correct_answer.strip():
            return self._last_resort_distractors(concept, count)

        candidates: List[str] = []

        # Strategy 1 – Knowledge-graph neighbours
        # Human note: we start here because it tends to produce "on-topic" wrong
        # answers cheaply, before we spend time on embeddings.
        candidates += self._from_knowledge_graph(
            concept, all_concepts, knowledge_graph, correct_answer
        )

        # Strategy 2 – Embedding-similar concepts
        # Human note: embeddings help us find "close enough to be tempting"
        # concepts without copying the correct answer.
        if len(candidates) < count * 2:
            candidates += self._from_embeddings(
                concept, all_concepts, correct_answer
            )

        # Strategy 3 – Context-derived phrases
        if len(candidates) < count * 2:
            candidates += self._from_context(context, concept, correct_answer)

        # De-duplicate with semantic check
        # This is the quality gate: avoid options that are too close to each
        # other or to the correct answer (Krish's work focused here).
        unique = self._filter_unique(candidates, correct_answer, count)

        # Strategy 4 – Rule-based fallback for any remaining slots
        if len(unique) < count:
            fallbacks = self._rule_based_fallback(
                correct_answer, concept, count - len(unique)
            )
            unique += fallbacks

        return unique[:count]

    # ──────────────────────────────────────────────────────────
    # Strategy 1 – Knowledge Graph neighbours
    # ──────────────────────────────────────────────────────────

    def _from_knowledge_graph(
        self,
        concept: str,
        all_concepts: List[Dict],
        knowledge_graph: Dict,
        correct_answer: str,
    ) -> List[str]:
        candidates: List[str] = []
        nodes = knowledge_graph.get("nodes", [])
        edges = knowledge_graph.get("edges", [])

        # Find the concept's node id
        concept_node_id = None
        for n in nodes:
            if n["label"].lower() == concept.lower():
                concept_node_id = n["id"]
                break

        if not concept_node_id:
            return candidates

        # Collect 1-hop neighbour labels
        neighbour_ids: set = set()
        for e in edges:
            if e["from"] == concept_node_id:
                neighbour_ids.add(e["to"])
            elif e["to"] == concept_node_id:
                neighbour_ids.add(e["from"])

        neighbour_labels = []
        for n in nodes:
            if n["id"] in neighbour_ids and n["label"].lower() != concept.lower():
                neighbour_labels.append(n["label"])

        # For each neighbour, try to find a short descriptive phrase from concepts list
        concept_lookup = {c["name"].lower(): c for c in all_concepts}
        for label in neighbour_labels:
            desc = concept_lookup.get(label.lower(), {}).get("context", "")
            if desc and len(desc) > 10:
                # Extract a sentence-length chunk
                sentence = self._first_sentence(desc)
                if sentence and sentence.lower() != correct_answer.lower():
                    candidates.append(sentence)
            else:
                # Use the label itself as distractor text
                if label.lower() != correct_answer.lower():
                    candidates.append(label)

        random.shuffle(candidates)
        return candidates[:6]

    # ──────────────────────────────────────────────────────────
    # Strategy 2 – Embedding similarity
    # ──────────────────────────────────────────────────────────

    def _from_embeddings(
        self,
        concept: str,
        all_concepts: List[Dict],
        correct_answer: str,
    ) -> List[str]:
        self._load_sentence_model()
        if self._sentence_model is None:
            return []

        try:
            import numpy as np

            other_concepts = [
                c for c in all_concepts if c["name"].lower() != concept.lower()
            ]
            if not other_concepts:
                return []

            names = [c["name"] for c in other_concepts]
            embeddings = self._sentence_model.encode([concept] + names)
            concept_emb = embeddings[0]

            scored = []
            for i, name in enumerate(names):
                sim = float(
                    np.dot(concept_emb, embeddings[i + 1])
                    / (np.linalg.norm(concept_emb) * np.linalg.norm(embeddings[i + 1]) + 1e-9)
                )
                scored.append((name, sim))

            # Pick moderately similar (0.25-0.80) — close enough to be plausible,
            # far enough to be wrong
            scored.sort(key=lambda x: x[1], reverse=True)
            candidates = [
                name
                for name, sim in scored
                if 0.20 < sim < 0.85
                and name.lower() != correct_answer.lower()
            ]
            return candidates[:6]
        except Exception:
            return []

    # ──────────────────────────────────────────────────────────
    # Strategy 3 – Context-derived phrases
    # ──────────────────────────────────────────────────────────

    def _from_context(
        self,
        context: str,
        concept: str,
        correct_answer: str,
    ) -> List[str]:
        """Extract plausible-sounding phrases from surrounding text."""
        if not context:
            return []

        sentences = [
            s.strip()
            for s in re.split(r"[.!?]", context)
            if len(s.strip()) > 15
        ]
        # Exclude sentences that are too similar to the correct answer
        candidates = []
        for s in sentences:
            s_clean = s.strip()
            if (
                s_clean.lower() != correct_answer.lower()
                and concept.lower() not in s_clean.lower()
                and len(s_clean) < 250
            ):
                candidates.append(s_clean)

        random.shuffle(candidates)
        return candidates[:6]

    # ──────────────────────────────────────────────────────────
    # Strategy 4 – Rule-based fallback
    # ──────────────────────────────────────────────────────────

    def _rule_based_fallback(
        self,
        correct_answer: str,
        concept: str,
        count: int,
    ) -> List[str]:
        results: List[str] = []
        words = correct_answer.split()

        if len(words) >= 4:
            # Create modified versions of the correct answer
            modifiers = [
                ("involves", "does not involve"),
                ("is", "is not"),
                ("increases", "decreases"),
                ("enables", "prevents"),
                ("before", "after"),
                ("related to", "unrelated to"),
            ]
            for pos_word, neg_word in modifiers:
                if pos_word in correct_answer.lower() and len(results) < count:
                    variant = re.sub(
                        re.escape(pos_word), neg_word, correct_answer, count=1, flags=re.IGNORECASE
                    )
                    results.append(variant)

        # Generic plausible-but-wrong options derived from concept name
        generic = [
            f"A process unrelated to {concept}",
            f"The opposite characteristic of {concept}",
            f"A common misconception about {concept}",
        ]
        for g in generic:
            if len(results) < count:
                results.append(g)

        return results[:count]

    # ──────────────────────────────────────────────────────────
    # Semantic deduplication
    # ──────────────────────────────────────────────────────────

    def _filter_unique(
        self,
        candidates: List[str],
        correct_answer: str,
        count: int,
    ) -> List[str]:
        """Keep only semantically distinct candidates.

        We want the learner to *think*, not to play "spot the duplicate".
        """
        if not candidates:
            return []

        self._load_sentence_model()

        if self._sentence_model is None:
            # Fall back to simple string dedup
            seen_lower: set = {correct_answer.lower()}
            unique: List[str] = []
            for c in candidates:
                if c.lower() not in seen_lower:
                    seen_lower.add(c.lower())
                    unique.append(c)
                if len(unique) >= count:
                    break
            return unique

        try:
            import numpy as np

            all_texts = [correct_answer] + candidates
            embeddings = self._sentence_model.encode(all_texts)
            correct_emb = embeddings[0]

            accepted: List[str] = []
            accepted_embs = []

            for i, cand in enumerate(candidates):
                cand_emb = embeddings[i + 1]

                # Check vs correct answer (don't leak the right option)
                sim_correct = float(
                    np.dot(correct_emb, cand_emb)
                    / (np.linalg.norm(correct_emb) * np.linalg.norm(cand_emb) + 1e-9)
                )
                if sim_correct > _SIM_THRESHOLD:
                    continue

                # Check vs already accepted (avoid two distractors that are the
                # same idea phrased differently)
                too_similar = False
                for acc_emb in accepted_embs:
                    sim = float(
                        np.dot(cand_emb, acc_emb)
                        / (np.linalg.norm(cand_emb) * np.linalg.norm(acc_emb) + 1e-9)
                    )
                    if sim > _SIM_THRESHOLD:
                        too_similar = True
                        break

                if not too_similar:
                    accepted.append(cand)
                    accepted_embs.append(cand_emb)

                if len(accepted) >= count:
                    break

            return accepted
        except Exception:
            # Fallback
            seen_lower = {correct_answer.lower()}
            unique = []
            for c in candidates:
                if c.lower() not in seen_lower:
                    seen_lower.add(c.lower())
                    unique.append(c)
                if len(unique) >= count:
                    break
            return unique

    # ──────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────

    def _load_sentence_model(self):
        if self._sentence_model is not None or self._model_attempted:
            return
        self._model_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self._sentence_model = None

    @staticmethod
    def _first_sentence(text: str) -> str:
        """Return the first sentence (up to 200 chars)."""
        for sep in (".", "!", "?"):
            idx = text.find(sep)
            if 10 < idx < 200:
                return text[: idx + 1].strip()
        return text[:200].strip()

    @staticmethod
    def _last_resort_distractors(concept: str, count: int) -> List[str]:
        pool = [
            f"A process unrelated to {concept}",
            f"The opposite characteristic of {concept}",
            f"A common misconception about {concept}",
            f"An unrelated concept from a different domain",
        ]
        return pool[:count]
