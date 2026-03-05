"""
Question Generator – High-Quality MCQ Engine
==============================================
Multi-provider architecture for educational MCQ generation.

Provider priority (auto-detected from environment variables):
  1. OpenAI GPT     — set OPENAI_API_KEY
  2. Google Gemini  — set GEMINI_API_KEY
  3. Local FLAN-T5  — always available (offline fallback)

API providers generate COMPLETE MCQs (question + 4 options + answer +
explanation) in a single prompt — producing ChatGPT/Gemini-level quality.

The local FLAN-T5 model uses a multi-step approach:
  Step 1 → generate question text (instruction-tuned prompt)
  Step 2 → extract answer via QA-style prompt
  Step 3 → generate distractors via sub-module
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Optional

from services.difficulty_controller import DifficultyController
from services.distractor_generator import DistractorGenerator
from services.concept_mapper import ConceptMapper

_LETTERS = ["A", "B", "C", "D"]


class QuestionGenerator:
    """
    Production MCQ engine with automatic provider selection.

    Public entry point: ``generate_questions(...)``
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "QUESTION_GEN_MODEL", "google/flan-t5-base"
        )
        self._provider: Optional[str] = None  # openai | gemini | local
        self._openai_client = None
        self._gemini_model = None
        self._t5_model = None
        self._t5_tokenizer = None
        self._device = "cpu"

        # Sub-modules
        self._difficulty = DifficultyController()
        self._distractors = DistractorGenerator()
        self._mapper = ConceptMapper()

    # ================================================================
    # Provider detection & lazy initialization
    # ================================================================

    def _init_provider(self):
        """Detect and initialize the best available LLM provider."""
        if self._provider is not None:
            return

        # 1. OpenAI
        key = os.getenv("OPENAI_API_KEY", "")
        if key and key != "your-openai-api-key":
            try:
                import openai

                self._openai_client = openai.OpenAI(api_key=key)
                self._provider = "openai"
                print("[QuestionGen] Provider: OpenAI GPT")
                return
            except ImportError:
                print("[QuestionGen] `openai` package not installed, skipping.")
            except Exception as exc:
                print(f"[QuestionGen] OpenAI init error: {exc}")

        # 2. Google Gemini
        key = os.getenv("GEMINI_API_KEY", "")
        if key and key != "your-gemini-api-key":
            try:
                import google.generativeai as genai

                genai.configure(api_key=key)
                model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                self._gemini_model = genai.GenerativeModel(model_name)
                self._provider = "gemini"
                print(f"[QuestionGen] Provider: Google Gemini ({model_name})")
                return
            except ImportError:
                print(
                    "[QuestionGen] `google-generativeai` package not installed, skipping."
                )
            except Exception as exc:
                print(f"[QuestionGen] Gemini init error: {exc}")

        # 3. Local FLAN-T5 (always available)
        self._load_local_model()
        self._provider = "local"

    def _load_local_model(self):
        if self._t5_model is not None:
            return
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            import torch

            print(f"[QuestionGen] Loading local model: {self.model_name}")
            self._t5_tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self._t5_model = T5ForConditionalGeneration.from_pretrained(
                self.model_name
            )
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._t5_model.to(self._device)
            print(f"[QuestionGen] Local model ready on {self._device}")
        except Exception as exc:
            print(f"[QuestionGen] Local model unavailable: {exc}")
            self._t5_model = None

    # ================================================================
    # Document understanding helpers
    # ================================================================

    def _build_document_summary(self, text: str, concepts: List[Dict]) -> str:
        """Build a concise overview of the full document for holistic question generation.

        This ensures the model understands the WHOLE document — not just a small
        snippet — so questions are logically grounded in the complete material.
        """
        concept_names = [c["name"] for c in concepts[:20]]

        # Extract the most informative sentences from the document
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 25]

        # Pick sentences that mention key concepts (up to ~1200 chars)
        key_sentences = []
        used_concepts = set()
        for s in sentences:
            s_lower = s.lower()
            for cn in concept_names:
                if cn.lower() in s_lower and cn not in used_concepts:
                    key_sentences.append(s.strip())
                    used_concepts.add(cn)
                    break
            if len(". ".join(key_sentences)) > 1200:
                break

        # If we have few concept-sentences, add the first few document sentences
        if len(key_sentences) < 5:
            for s in sentences[:10]:
                if s not in key_sentences:
                    key_sentences.append(s)
                if len(key_sentences) >= 8:
                    break

        overview = ". ".join(key_sentences)
        if not overview.endswith("."):
            overview += "."

        topic_list = ", ".join(concept_names[:15])
        return (
            f"DOCUMENT OVERVIEW: This material covers the following key topics: "
            f"{topic_list}.\n\n"
            f"KEY CONTENT: {overview[:1500]}"
        )

    # ================================================================
    # PUBLIC → generate_questions
    # ================================================================

    def generate_questions(
        self,
        text: str,
        concepts: List[Dict],
        knowledge_graph: Dict,
        num_questions: int = 10,
        question_types: List[str] = None,  # backward compat, ignored
        difficulty: str = "mixed",
    ) -> List[Dict]:
        """
        Generate *num_questions* MCQs from the full document text + concepts.

        The generator first builds a document-level summary of all key topics
        so that every question is grounded in overall document understanding
        rather than isolated snippets.

        Returns list of MCQ dicts with keys:
            question, options, correct_answer, concept, difficulty, explanation
        Plus backward-compat keys: text, type, id
        """
        if not concepts:
            raise ValueError("No concepts provided for question generation.")
        if not text:
            raise ValueError("No text provided for question generation.")

        print(
            f"[QuestionGen] Generating {num_questions} MCQs "
            f"({len(concepts)} concepts, difficulty={difficulty})"
        )

        self._init_provider()
        self._mapper.build_concept_index(concepts, text)

        # Build a condensed document overview for holistic understanding
        self._doc_summary = self._build_document_summary(text, concepts)

        selected = DifficultyController.select_concepts_for_difficulty(
            concepts, difficulty, num_questions * 3
        )

        questions: List[Dict] = []
        used: List[str] = []

        for concept in selected:
            if len(questions) >= num_questions:
                break
            q = self._generate_single_mcq(
                concept, text, concepts, knowledge_graph, difficulty, used
            )
            if q is not None:
                questions.append(q)
                used.append(q["question"])

        # Finalize
        random.shuffle(questions)
        for i, q in enumerate(questions):
            q["id"] = f"q_{i}"
            q["text"] = q["question"]  # backward compat
            q["type"] = "mcq"  # backward compat

        print(f"[QuestionGen] Generated {len(questions)} MCQs via {self._provider}")
        return questions

    # ================================================================
    # Single MCQ pipeline
    # ================================================================

    def _generate_single_mcq(
        self, concept, text, all_concepts, kg, target_difficulty, existing
    ) -> Optional[Dict]:
        name = concept["name"]
        diff = DifficultyController.assign_difficulty_label(concept, target_difficulty)
        ctx = self._mapper.get_context(text, name)
        if not ctx or len(ctx.strip()) < 30:
            return None

        related_names = self._mapper.get_related_concepts(
            name, all_concepts, kg, top_k=3
        )
        related = related_names[0] if related_names else None

        # ── Attempt 1: Provider-specific generation ──
        mcq = None
        if self._provider in ("openai", "gemini"):
            mcq = self._api_generate_mcq(diff, name, related, ctx)
        else:
            mcq = self._local_generate_mcq(
                diff, name, related, ctx, text, all_concepts, kg
            )

        # ── Attempt 2: Rule-based fallback ──
        if mcq is None:
            mcq = self._rule_based_mcq(
                diff, name, related, ctx, text, all_concepts, kg
            )

        if mcq is None:
            return None

        # Duplicate check
        if self._mapper.is_duplicate_question(mcq["question"], existing):
            return None

        mcq["concept"] = name
        mcq["difficulty"] = diff.capitalize()
        return mcq

    # ================================================================
    # API Generation (OpenAI / Gemini) — single-prompt complete MCQ
    # ================================================================

    def _api_generate_mcq(
        self, difficulty, concept, related, context
    ) -> Optional[Dict]:
        prompt = self._build_api_prompt(difficulty, concept, related, context)
        raw = self._call_api(prompt)
        parsed = self._parse_mcq_response(raw)
        if parsed and len(parsed.get("options", [])) == 4:
            return parsed
        return None

    def _build_api_prompt(self, difficulty, concept, related, context):
        bloom = DifficultyController.get_bloom_info(difficulty)
        taxonomy = bloom["taxonomy"]
        verbs = ", ".join(random.sample(bloom["verbs"], min(3, len(bloom["verbs"]))))

        if difficulty.lower() == "easy":
            diff_instruction = (
                f"Create a RECALL / UNDERSTANDING level question.\n"
                f"The student should need to {verbs}.\n"
                f"Focus on definitions, facts, or key terms.\n"
                f"The question should test whether the student truly understands "
                f"the concept — NOT just recognize a keyword."
            )
        elif difficulty.lower() == "medium":
            rel = f" Relate it to '{related}'." if related else ""
            diff_instruction = (
                f"Create an APPLICATION / ANALYSIS level question.\n"
                f"The student should need to {verbs}.{rel}\n"
                f"Focus on relationships, cause-and-effect, or comparisons.\n"
                f"The student should need to REASON about the concept — "
                f"not just recall a definition. Present a concrete situation "
                f"where the student must APPLY their knowledge."
            )
        else:
            rel = f" You may contrast with '{related}'." if related else ""
            diff_instruction = (
                f"Create an EVALUATION / SYNTHESIS level question.\n"
                f"The student should need to {verbs}.{rel}\n"
                f"Use a realistic scenario requiring critical thinking.\n"
                f"The question MUST present a specific scenario, case study, "
                f"or hypothetical situation. The student should EVALUATE "
                f"trade-offs or PREDICT outcomes based on deep understanding."
            )

        # Include the document-level summary so the model understands
        # the broader context — not just a small snippet.
        doc_ctx = getattr(self, "_doc_summary", "")
        doc_block = f"\n{doc_ctx}\n\n" if doc_ctx else ""

        prompt = (
            "You are an expert educational assessment designer who creates "
            "exam-quality questions that test deep understanding.\n\n"
            "Generate ONE high-quality multiple-choice question "
            "from the lecture content below.\n\n"
            f"{doc_block}"
            f"SPECIFIC SECTION ON '{concept}':\n{context[:2000]}\n\n"
            f"TOPIC: {concept}\n"
            f"DIFFICULTY: {difficulty.upper()} (Bloom's Taxonomy: {taxonomy})\n\n"
            f"INSTRUCTIONS:\n{diff_instruction}\n\n"
            "QUALITY RULES (MANDATORY):\n"
            "- The question must require THINKING, not just pattern matching\n"
            "- Write a clear, unambiguous question ending with ?\n"
            "- Provide EXACTLY 4 options labelled A), B), C), D)\n"
            "- Only ONE option must be unambiguously correct\n"
            "- WRONG options must reflect common MISCONCEPTIONS or plausible "
            "errors students actually make (e.g. confusing similar concepts, "
            "partial truths, reversed cause-effect, over-generalizations)\n"
            "- All options must be similar in length, specificity, and tone\n"
            "- Do NOT include 'All of the above' or 'None of the above'\n"
            "- Do NOT make the correct answer obviously different from others\n"
            "- The explanation must justify WHY the answer is correct AND "
            "briefly explain why each wrong option fails\n\n"
            "FORMAT (follow exactly):\n"
            "QUESTION: <question text?>\n"
            "A) <option>\n"
            "B) <option>\n"
            "C) <option>\n"
            "D) <option>\n"
            "ANSWER: <A, B, C, or D>\n"
            "EXPLANATION: <explanation>"
        )
        return prompt

    def _call_api(self, prompt: str) -> Optional[str]:
        if self._provider == "openai":
            return self._call_openai(prompt)
        if self._provider == "gemini":
            return self._call_gemini(prompt)
        return None

    def _call_openai(self, prompt: str) -> Optional[str]:
        try:
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            resp = self._openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a university-level exam question designer. "
                            "You write questions that test genuine understanding "
                            "and logical reasoning — never trivia or pattern matching. "
                            "Your wrong options always target common student "
                            "misconceptions. Follow the requested format exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.75,
                max_tokens=600,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[QuestionGen] OpenAI error: {exc}")
            return None

    def _call_gemini(self, prompt: str) -> Optional[str]:
        try:
            resp = self._gemini_model.generate_content(
                prompt,
                generation_config={"temperature": 0.7, "max_output_tokens": 500},
            )
            return resp.text.strip()
        except Exception as exc:
            print(f"[QuestionGen] Gemini error: {exc}")
            return None

    # ================================================================
    # Local (FLAN-T5) Generation — multi-step approach
    # ================================================================

    def _local_generate_mcq(
        self, difficulty, concept, related, context, text, all_concepts, kg
    ) -> Optional[Dict]:
        """
        Multi-step local generation:
          1. Try complete MCQ prompt (sometimes works with larger models)
          2. Generate question → answer → distractors separately
        """
        # Step 1: Try complete MCQ generation
        mcq = self._local_try_complete(difficulty, concept, related, context)
        if mcq:
            return mcq

        # Step 2: Multi-step generation
        mcq = self._local_try_multistep(
            difficulty, concept, related, context, all_concepts, kg
        )
        return mcq

    def _local_try_complete(
        self, difficulty, concept, related, context
    ) -> Optional[Dict]:
        """Attempt complete MCQ generation with FLAN-T5."""
        bloom = DifficultyController.get_bloom_info(difficulty)
        verb = random.choice(bloom["verbs"])
        # Include a brief document overview for broader understanding
        doc_hint = ""
        raw_summary = getattr(self, "_doc_summary", "")
        if raw_summary:
            # Extract just the topic list (first line) — keep it short for T5
            topic_line = raw_summary.split("\n")[0][:200]
            doc_hint = f"{topic_line}\n\n"
        prompt = (
            f"{doc_hint}"
            f"You are an exam question writer. Create a {difficulty} difficulty "
            f"multiple choice question about \"{concept}\" that requires the "
            f"student to {verb}. Provide 4 options (A, B, C, D) where only one "
            f"is correct and the wrong options represent common misconceptions. "
            f"Include the correct answer letter and a brief explanation.\n\n"
            f"Context: {context[:500]}\n\n"
            f"QUESTION:"
        )
        raw = self._t5_generate(prompt, max_length=300)
        if not raw:
            return None
        # Prepend QUESTION: since the prompt ended with it
        return self._parse_mcq_response("QUESTION: " + raw)

    def _local_try_multistep(
        self, difficulty, concept, related, context, all_concepts, kg
    ) -> Optional[Dict]:
        """Generate question, answer, and distractors in separate model calls."""
        bloom = DifficultyController.get_bloom_info(difficulty)
        verb = random.choice(bloom["verbs"])

        # ── Step A: Generate question text ──
        if difficulty.lower() == "easy":
            q_prompt = (
                f"Based on the following passage, generate a clear educational "
                f"question that tests whether a student can {verb} the concept "
                f'of "{concept}". The question should require understanding, '
                f"not just keyword recognition.\n\n"
                f"Passage: {context[:500]}\n\n"
                f"Question:"
            )
        elif difficulty.lower() == "medium":
            rel_part = f' and relate it to "{related}"' if related else ""
            q_prompt = (
                f"Based on the following passage, generate an analytical question "
                f"that requires a student to {verb} the concept of "
                f'"{concept}"{rel_part}. The question should test reasoning '
                f"about cause-effect or comparison, not just recall.\n\n"
                f"Passage: {context[:500]}\n\n"
                f"Question:"
            )
        else:
            q_prompt = (
                f"Based on the following passage, generate a challenging question "
                f"that presents a realistic scenario and requires a student to "
                f'{verb} and think critically about "{concept}". The student '
                f"must evaluate or predict an outcome.\n\n"
                f"Passage: {context[:500]}\n\n"
                f"Question:"
            )

        question = self._t5_generate(q_prompt, max_length=128)
        if not question or len(question) < 10:
            return None
        question = question.strip()
        if not question.endswith("?"):
            question += "?"

        # ── Step B: Extract correct answer via QA prompt ──
        a_prompt = (
            f"Answer the question based on the context.\n\n"
            f"Context: {context[:500]}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        answer = self._t5_generate(a_prompt, max_length=100)
        if not answer or len(answer.strip()) < 3:
            answer = self._extract_answer_from_context(context, concept, question)
        if not answer or len(answer.strip()) < 3:
            return None
        answer = answer.strip()

        # Truncate overly long answers
        if len(answer) > 200:
            # Cut at sentence boundary
            cut = answer[:200].rfind(".")
            answer = answer[: cut + 1] if cut > 50 else answer[:200]

        # ── Step C: Generate distractors ──
        distractors = self._generate_local_distractors(
            question, answer, concept, context, all_concepts, kg
        )
        if len(distractors) < 3:
            return None

        return self._assemble_mcq(question, answer, distractors, context, concept)

    def _generate_local_distractors(
        self, question, answer, concept, context, all_concepts, kg
    ) -> List[str]:
        """
        Try to generate distractors with the model first,
        then fall back to the distractor sub-module.
        """
        model_distractors: List[str] = []

        # Try model-based distractors
        if self._t5_model is not None:
            d_prompt = (
                f"A student is answering: {question}\n"
                f"The correct answer is: {answer}\n\n"
                f"Generate 3 wrong answers that represent common student "
                f"misconceptions. Each wrong answer should be plausible "
                f"and from the same domain but clearly incorrect.\n\n"
                f"Context: {context[:300]}\n\n"
                f"Wrong answers:"
            )
            raw = self._t5_generate(d_prompt, max_length=150)
            if raw:
                # Try to extract individual items
                parts = re.split(r"[,\n;]|\d+[.)]\s*", raw)
                for p in parts:
                    p = p.strip().strip("-").strip()
                    if (
                        p
                        and len(p) > 3
                        and p.lower() != answer.lower()
                        and p.lower() != concept.lower()
                    ):
                        model_distractors.append(p)

        if len(model_distractors) >= 3:
            return model_distractors[:3]

        # Fall back to distractor sub-module
        sub_distractors = self._distractors.generate(
            correct_answer=answer,
            concept=concept,
            all_concepts=all_concepts,
            knowledge_graph=kg,
            context=context,
            count=3 - len(model_distractors),
        )
        return (model_distractors + sub_distractors)[:3]

    def _t5_generate(self, prompt: str, max_length: int = 200) -> Optional[str]:
        """Low-level FLAN-T5 generation."""
        if self._t5_model is None:
            self._load_local_model()
        if self._t5_model is None:
            return None
        try:
            import torch

            inputs = self._t5_tokenizer(
                prompt, return_tensors="pt", max_length=512, truncation=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._t5_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            return (
                self._t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            )
        except Exception as exc:
            print(f"[QuestionGen] T5 generation error: {exc}")
            return None

    # ================================================================
    # Rule-based fallback (last resort)
    # ================================================================

    def _rule_based_mcq(
        self, difficulty, concept, related, context, text, all_concepts, kg
    ) -> Optional[Dict]:
        """Generate MCQ using Bloom's templates + distractor sub-module."""
        question = DifficultyController.get_question_stem(
            difficulty, concept, related, context
        )
        if not question:
            return None

        answer = self._extract_answer_from_context(context, concept, question)
        if not answer or len(answer.strip()) < 3:
            return None

        distractors = self._distractors.generate(
            correct_answer=answer,
            concept=concept,
            all_concepts=all_concepts,
            knowledge_graph=kg,
            context=context,
            count=3,
        )
        if len(distractors) < 3:
            return None

        return self._assemble_mcq(question, answer, distractors, context, concept)

    # ================================================================
    # Shared helpers
    # ================================================================

    @staticmethod
    def _strip_option_label(text: str) -> str:
        """Remove leading letter labels like 'A. ', 'A) ', 'a. ' from option text."""
        return re.sub(r'^[A-Da-d][.):]\s*', '', text.strip())

    def _assemble_mcq(
        self,
        question: str,
        correct: str,
        distractors: List[str],
        context: str,
        concept: str,
    ) -> Dict:
        """Shuffle options, assign letters, build final MCQ dict.

        Options are stored as PLAIN TEXT without letter prefixes.
        The UI templates add their own A/B/C/D labels.
        """
        # Strip any existing letter prefixes from all option strings
        correct_clean = self._strip_option_label(correct)
        dist_clean = [self._strip_option_label(d) for d in distractors[:3]]

        raw = [correct_clean] + dist_clean
        random.shuffle(raw)
        idx = raw.index(correct_clean)
        letter = _LETTERS[idx]
        explanation = self._build_explanation(context, concept, correct_clean)
        return {
            "question": question,
            "options": raw,          # plain text, NO "A. " prefix
            "correct_answer": letter, # "A", "B", "C", or "D"
            "explanation": explanation,
        }

    def _extract_answer_from_context(
        self, context: str, concept: str, question: str
    ) -> Optional[str]:
        """Extract a correct answer from context using heuristics + similarity."""
        q_lower = question.lower()
        sentences = [
            s.strip() for s in re.split(r"[.!?]", context) if len(s.strip()) > 15
        ]

        # ── Definition questions ──
        if any(
            kw in q_lower
            for kw in ("what is", "define", "definition", "describes", "refers to")
        ):
            for s in sentences:
                if concept.lower() in s.lower():
                    m = re.search(
                        rf"{re.escape(concept)}\s+(?:is|refers to|means|defined as)\s+(.+)",
                        s,
                        re.IGNORECASE,
                    )
                    if m:
                        return m.group(1).strip()[:200]

        # ── Purpose questions ──
        if any(kw in q_lower for kw in ("why", "purpose", "used for")):
            for s in sentences:
                if concept.lower() in s.lower() and any(
                    w in s.lower()
                    for w in ("used for", "purpose", "helps", "enables", "allows")
                ):
                    return s[:200]

        # ── Process / how questions ──
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
                    embs = self._mapper._sentence_model.encode(
                        [question] + relevant
                    )
                    q_emb = embs[0]
                    best, best_sim = relevant[0], -1.0
                    for i, s in enumerate(relevant):
                        sim = float(
                            np.dot(q_emb, embs[i + 1])
                            / (
                                np.linalg.norm(q_emb)
                                * np.linalg.norm(embs[i + 1])
                                + 1e-9
                            )
                        )
                        if sim > best_sim:
                            best_sim = sim
                            best = s
                    return best[:250]
            except Exception:
                pass

        # ── Last resort ──
        relevant = [s for s in sentences if concept.lower() in s.lower()]
        if relevant:
            return max(relevant, key=len)[:250]
        return context[:150].strip() if context else None

    def _build_explanation(
        self, context: str, concept: str, answer: str
    ) -> str:
        """Build a pedagogically useful explanation for the correct answer."""
        sentences = re.split(r"[.!?]", context)
        relevant = [
            s.strip()
            for s in sentences
            if concept.lower() in s.lower() and len(s.strip()) > 20
        ]
        if relevant:
            info = ". ".join(relevant[:2]).strip()
            if not info.endswith("."):
                info += "."
        else:
            info = context[:300].strip()
            if not info.endswith("."):
                info += "."

        display = (answer[:180] + "...") if len(answer) > 180 else answer
        return (
            f"The correct answer is: {display}\n\n"
            f"Key reasoning: {info}\n\n"
            f"Concept: {concept}"
        )

    # ================================================================
    # Response parsers (multi-strategy)
    # ================================================================

    def _parse_mcq_response(self, raw: Optional[str]) -> Optional[Dict]:
        """Try multiple parsing strategies to extract structured MCQ."""
        if not raw or len(raw) < 20:
            return None

        result = self._parse_structured(raw)
        if result:
            return result

        result = self._parse_json(raw)
        if result:
            return result

        result = self._parse_loose(raw)
        return result

    def _parse_structured(self, raw: str) -> Optional[Dict]:
        """Parse QUESTION: / A) B) C) D) / ANSWER: / EXPLANATION: format."""
        # Question
        q_m = re.search(
            r"QUESTION:\s*(.+?)(?=\n\s*[A-D]\))",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        if not q_m:
            q_m = re.search(
                r"^(.+?\?)\s*\n\s*[A-D]\)", raw, re.DOTALL | re.MULTILINE
            )
        if not q_m:
            return None
        question = q_m.group(1).strip()

        # Options
        opts: Dict[str, str] = {}
        for letter in _LETTERS:
            pat = (
                rf"{letter}\)\s*(.+?)"
                rf"(?=\n\s*[A-D]\)|\n\s*(?:ANSWER|CORRECT|EXPLANATION):|$)"
            )
            m = re.search(pat, raw, re.DOTALL | re.IGNORECASE)
            if m:
                opts[letter] = m.group(1).strip()
        if len(opts) != 4:
            return None

        # Answer letter
        a_m = re.search(r"(?:ANSWER|CORRECT):\s*([A-Da-d])", raw, re.IGNORECASE)
        if not a_m:
            return None
        letter = a_m.group(1).upper()
        if letter not in opts:
            return None

        # Explanation (optional)
        e_m = re.search(r"EXPLANATION:\s*(.+)", raw, re.DOTALL | re.IGNORECASE)
        explanation = ""
        if e_m:
            explanation = e_m.group(1).strip().split("\n")[0]

        # Store options as plain text — templates add their own A/B/C/D labels
        clean_opts = [self._strip_option_label(opts[l]) for l in _LETTERS]
        return {
            "question": question,
            "options": clean_opts,
            "correct_answer": letter,
            "explanation": explanation
            or f"The correct answer is {letter}. {opts.get(letter, '')}",
        }

    def _parse_json(self, raw: str) -> Optional[Dict]:
        """Parse JSON-formatted response."""
        try:
            m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group())

            q = data.get("question", "")
            options = data.get("options", [])
            correct = data.get("correct_answer", data.get("answer", ""))
            explanation = data.get("explanation", "")

            if not q or len(options) != 4 or not correct:
                return None

            # Store as plain text — strip any letter prefixes
            clean_opts = [self._strip_option_label(str(opt)) for opt in options]

            c = correct.strip().upper()
            if len(c) > 1:
                c = c[0]
            if c not in _LETTERS:
                return None

            return {
                "question": q,
                "options": clean_opts,
                "correct_answer": c,
                "explanation": explanation,
            }
        except (json.JSONDecodeError, KeyError):
            return None

    def _parse_loose(self, raw: str) -> Optional[Dict]:
        """Last-resort loose parsing for varied formats."""
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        if len(lines) < 5:
            return None

        # Find question
        question = None
        opt_start = 0
        for i, line in enumerate(lines):
            clean = re.sub(
                r"^(?:QUESTION|Q):\s*", "", line, flags=re.IGNORECASE
            ).strip()
            if clean.endswith("?"):
                question = clean
                opt_start = i + 1
                break
        if not question:
            question = lines[0].strip()
            if not question.endswith("?"):
                question += "?"
            opt_start = 1

        # Find options
        opts: Dict[str, str] = {}
        for line in lines[opt_start:]:
            m = re.match(r"^([A-Da-d])[.)]\s*(.+)", line)
            if m:
                opts[m.group(1).upper()] = m.group(2).strip()
        if len(opts) < 4:
            return None

        # Find answer
        letter = None
        for line in lines:
            m = re.search(
                r"(?:answer|correct)[:\s]+([A-Da-d])", line, re.IGNORECASE
            )
            if m:
                letter = m.group(1).upper()
                break
        if not letter or letter not in opts:
            return None

        # Explanation
        explanation = ""
        for line in lines:
            m = re.search(
                r"(?:explanation|reason)[:\s]+(.+)", line, re.IGNORECASE
            )
            if m:
                explanation = m.group(1).strip()
                break

        clean_opts = [self._strip_option_label(opts[l]) for l in _LETTERS if l in opts]
        return {
            "question": question,
            "options": clean_opts[:4],
            "correct_answer": letter,
            "explanation": explanation or f"The correct answer is {letter}.",
        }

