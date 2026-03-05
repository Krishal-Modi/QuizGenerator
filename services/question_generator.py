"""
Question Generator Service – HIGH-QUALITY MCQ Generation Engine
================================================================
Generates Multiple-Choice Questions with:
• Bloom's Taxonomy-based difficulty (Easy / Medium / Hard)
• Exactly 4 options (1 correct + 3 plausible distractors from same topic cluster)
• Concept-aligned questions linked to the Knowledge Graph
• Duplicate question filtering via embedding similarity
• Fast generation (<2 s) using cached embeddings & pre-extracted concepts

Output per question
───────────────────
{
  "question": "<clear conceptual question>",
  "options":  ["A. …", "B. …", "C. …", "D. …"],
  "correct_answer": "<A/B/C/D>",
  "concept": "<concept name>",
  "difficulty": "<Easy/Medium/Hard>",
  "explanation": "<short explanation referencing lecture concept>"
}
"""

from __future__ import annotations
import random
import re
import os
from typing import Dict, List, Optional

from services.difficulty_controller import DifficultyController
from services.distractor_generator import DistractorGenerator
from services.concept_mapper import ConceptMapper


_OPTION_LETTERS = ["A", "B", "C", "D"]


class QuestionGenerator:
    """
    Production quiz-generation engine.

    Public entry point: ``generate_questions(...)``
    Returns a list of MCQ dicts following the project's canonical schema.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "QUESTION_GEN_MODEL", "google/flan-t5-base"
        )
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._initialized = False

        # Sub-modules
        self._difficulty = DifficultyController()
        self._distractors = DistractorGenerator()
        self._mapper = ConceptMapper()

    # ──────────────────────────────────────────────────────────
    # Lazy model loading
    # ──────────────────────────────────────────────────────────

    def _load_model(self):
        if self._initialized:
            return
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            import torch

            print(f"[QuestionGen] Loading model: {self.model_name}")
            self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._initialized = True
            print(f"[QuestionGen] Model loaded on {self._device}")
        except Exception as e:
            print(f"[QuestionGen] Model unavailable ({e}). Using rule-based generation.")
            self._initialized = False

    # ──────────────────────────────────────────────────────────
    # PUBLIC  →  generate_questions
    # ──────────────────────────────────────────────────────────

    def generate_questions(
        self,
        text: str,
        concepts: List[Dict],
        knowledge_graph: Dict,
        num_questions: int = 10,
        question_types: List[str] = None,  # kept for backward compat; MCQ only
        difficulty: str = "mixed",
    ) -> List[Dict]:
        """
        Generate *num_questions* MCQs from lecture text + concepts.

        Args:
            text:            Full document text.
            concepts:        Pre-extracted concept dicts (name, score, …).
            knowledge_graph: KG dict with 'nodes' and 'edges'.
            num_questions:   Target count.
            question_types:  Accepted for backward compatibility, ignored (MCQ only).
            difficulty:      'easy' | 'medium' | 'hard' | 'mixed'

        Returns:
            List[Dict] – each dict follows the canonical MCQ schema.
        """
        if not concepts:
            raise ValueError("No concepts provided for question generation.")
        if not text:
            raise ValueError("No text provided for question generation.")

        print(
            f"[QuestionGen] Generating {num_questions} MCQs from "
            f"{len(text)} chars, {len(concepts)} concepts, difficulty={difficulty}"
        )

        # ── 1. Pre-compute caches (makes per-question work fast) ──
        self._mapper.build_concept_index(concepts, text)
        self._load_model()

        # ── 2. Select concepts (over-sample to tolerate skipped ones) ──
        selected = DifficultyController.select_concepts_for_difficulty(
            concepts, difficulty, num_questions * 3
        )

        # ── 3. Generate MCQs ──
        questions: List[Dict] = []
        used_stems: List[str] = []  # for duplicate detection

        for concept in selected:
            if len(questions) >= num_questions:
                break

            q = self._generate_single_mcq(
                concept, text, concepts, knowledge_graph, difficulty, used_stems
            )
            if q is not None:
                questions.append(q)
                used_stems.append(q["question"])

        # ── 4. Shuffle + assign IDs ──
        random.shuffle(questions)
        for i, q in enumerate(questions):
            q["id"] = f"q_{i}"

        # Also produce internal 'text' and 'type' fields for backward compat
        # with quiz_service / take.html which read q['text'] and q['type']
        for q in questions:
            q["text"] = q["question"]
            q["type"] = "mcq"

        print(f"[QuestionGen] Generated {len(questions)} MCQs")
        return questions

    # ──────────────────────────────────────────────────────────
    # Single MCQ pipeline
    # ──────────────────────────────────────────────────────────

    def _generate_single_mcq(
        self,
        concept: Dict,
        text: str,
        all_concepts: List[Dict],
        knowledge_graph: Dict,
        target_difficulty: str,
        existing_questions: List[str],
    ) -> Optional[Dict]:
        concept_name: str = concept["name"]
        diff_label = DifficultyController.assign_difficulty_label(concept, target_difficulty)
        context = self._mapper.get_context(text, concept_name)

        if not context or len(context.strip()) < 30:
            return None

        # Related concept for medium/hard stems
        related_names = self._mapper.get_related_concepts(
            concept_name, all_concepts, knowledge_graph, top_k=3
        )
        related = related_names[0] if related_names else None

        # ── Question text ──
        question_text = self._make_question_text(
            diff_label, concept_name, related, context
        )
        if not question_text:
            return None

        # ── Duplicate check ──
        if self._mapper.is_duplicate_question(question_text, existing_questions):
            return None

        # ── Correct answer ──
        correct_answer = self._extract_correct_answer(
            text, context, concept_name, question_text
        )
        if not correct_answer or len(correct_answer.strip()) < 3:
            return None

        # ── Distractors (exactly 3) ──
        distractors = self._distractors.generate(
            correct_answer=correct_answer,
            concept=concept_name,
            all_concepts=all_concepts,
            knowledge_graph=knowledge_graph,
            context=context,
            count=3,
        )
        if len(distractors) < 3:
            return None  # quality gate

        # ── Assemble 4 options, shuffle, record letter ──
        raw_options = [correct_answer] + distractors[:3]
        random.shuffle(raw_options)
        correct_index = raw_options.index(correct_answer)
        correct_letter = _OPTION_LETTERS[correct_index]

        labelled_options = [
            f"{_OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(raw_options)
        ]

        # ── Explanation ──
        explanation = self._build_explanation(context, concept_name, correct_answer)

        return {
            "question": question_text,
            "options": labelled_options,
            "correct_answer": correct_letter,
            "concept": concept_name,
            "difficulty": diff_label.capitalize(),
            "explanation": explanation,
        }

    # ──────────────────────────────────────────────────────────
    # Question text generation
    # ──────────────────────────────────────────────────────────

    def _make_question_text(
        self,
        difficulty: str,
        concept: str,
        related: Optional[str],
        context: str,
    ) -> Optional[str]:
        """Try model generation first; fall back to Bloom's templates."""
        # Try transformer
        if self._initialized:
            prompt = DifficultyController.get_model_prompt(
                difficulty, concept, context, related
            )
            q = self._generate_with_model(prompt)
            if q and len(q) > 15:
                return q

        # Rule-based stem from difficulty controller
        return DifficultyController.get_question_stem(
            difficulty, concept, related, context
        )

    def _generate_with_model(self, prompt: str) -> Optional[str]:
        try:
            import torch

            inputs = self._tokenizer(
                prompt, return_tensors="pt", max_length=512, truncation=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7,
                )
            question = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if not question.endswith("?"):
                question += "?"
            return question
        except Exception as e:
            print(f"[QuestionGen] Model generation failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # Answer extraction
    # ──────────────────────────────────────────────────────────

    def _extract_correct_answer(
        self,
        full_text: str,
        context: str,
        concept: str,
        question: str,
    ) -> str:
        """
        Extract a concise correct answer from the source material.
        Uses question-type heuristics → sentence similarity fallback.
        """
        q_lower = question.lower()
        sentences = [
            s.strip()
            for s in re.split(r"[.!?]", context)
            if len(s.strip()) > 15
        ]

        # ── Heuristic: definition questions ──
        if any(kw in q_lower for kw in ("what is", "define", "definition", "describes", "refers to")):
            for s in sentences:
                if concept.lower() in s.lower():
                    m = re.search(
                        rf"{re.escape(concept)}\s+(?:is|refers to|means|defined as)\s+(.+)",
                        s,
                        re.IGNORECASE,
                    )
                    if m:
                        return m.group(1).strip()[:200]

        # ── Heuristic: purpose questions ──
        if any(kw in q_lower for kw in ("why", "purpose", "used for")):
            for s in sentences:
                if concept.lower() in s.lower() and any(
                    w in s.lower() for w in ("used for", "purpose", "helps", "enables", "allows")
                ):
                    return s[:200]

        # ── Heuristic: process / how questions ──
        if any(kw in q_lower for kw in ("how", "process", "method", "work")):
            for s in sentences:
                if concept.lower() in s.lower() and any(
                    w in s.lower() for w in ("process", "step", "method", "by")
                ):
                    return s[:200]

        # ── Sentence-similarity fallback ──
        self._mapper._load_sentence_model()
        if self._mapper._sentence_model is not None:
            try:
                import numpy as np

                relevant = [s for s in sentences if len(s) > 20]
                if relevant:
                    embs = self._mapper._sentence_model.encode([question] + relevant)
                    q_emb = embs[0]
                    best, best_sim = relevant[0], -1.0
                    for i, s in enumerate(relevant):
                        sim = float(
                            np.dot(q_emb, embs[i + 1])
                            / (np.linalg.norm(q_emb) * np.linalg.norm(embs[i + 1]) + 1e-9)
                        )
                        if sim > best_sim:
                            best_sim = sim
                            best = s
                    return best[:250]
            except Exception:
                pass

        # ── Last resort: most informative sentence containing concept ──
        relevant = [s for s in sentences if concept.lower() in s.lower()]
        if relevant:
            return max(relevant, key=len)[:250]

        return f"{concept}: " + context[:150].strip()

    # ──────────────────────────────────────────────────────────
    # Explanation builder
    # ──────────────────────────────────────────────────────────

    def _build_explanation(self, context: str, concept: str, answer: str) -> str:
        sentences = re.split(r"[.!?]", context)
        relevant = [
            s.strip()
            for s in sentences
            if concept.lower() in s.lower() and len(s.strip()) > 20
        ]
        if relevant:
            key_info = ". ".join(relevant[:2]).strip()
            if not key_info.endswith("."):
                key_info += "."
        else:
            key_info = context[:300].strip()
            if not key_info.endswith("."):
                key_info += "."

        display = answer[:180] + "..." if len(answer) > 180 else answer
        return (
            f"Correct: {display}\n\n"
            f"{key_info}\n\n"
            f"Concept: {concept}"
        )

