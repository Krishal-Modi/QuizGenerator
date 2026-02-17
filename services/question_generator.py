"""
Question Generator Service - Generate MCQ, True/False, and Short Answer questions
using Transformer-based models (T5, FLAN-T5)
"""
import re
import random
from typing import List, Dict, Optional, Tuple
import os


class QuestionGenerator:
    """
    AI-powered question generation using transformer models.
    
    Generates three types of questions:
    1. Multiple Choice Questions (MCQ)
    2. True/False Questions
    3. Short Answer Questions
    
    Uses FLAN-T5 for question generation with concept-aware prompting.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv('QUESTION_GEN_MODEL', 'google/flan-t5-base')
        self._model = None
        self._tokenizer = None
        self._sentence_model = None
        self._initialized = False
    
    def _load_model(self):
        """Lazy load the transformer model"""
        if self._initialized:
            return
        
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            import torch
            
            print(f"Loading model: {self.model_name}")
            self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            # Move to GPU if available
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model.to(self._device)
            
            self._initialized = True
            print(f"Model loaded on {self._device}")
            
        except ImportError:
            print("transformers or torch not installed. Using rule-based generation.")
            self._initialized = False
        except Exception as e:
            print(f"Failed to load model: {e}. Using rule-based generation.")
            self._initialized = False
    
    def _load_sentence_model(self):
        """Load sentence transformer for similarity-based distractor generation"""
        if self._sentence_model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            self._sentence_model = None
    
    def _chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 300) -> List[str]:
        """
        Split long text into overlapping chunks for better context extraction.
        This ensures we can generate questions from the ENTIRE document.
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence end near the chunk boundary
                for sep in ['. ', '\n\n', '\n', '. ']:
                    boundary = text.rfind(sep, start + chunk_size - 200, end + 200)
                    if boundary > start:
                        end = boundary + len(sep)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        
        return chunks

    def _get_best_context_for_concept(self, text: str, concept_name: str, context_size: int = 1500) -> str:
        """
        Find the best, most relevant context for a given concept from the full document.
        Uses multiple search strategies for comprehensive coverage.
        """
        contexts = []
        
        # Strategy 1: Direct search for concept mentions
        pattern = re.compile(re.escape(concept_name), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        
        for match in matches[:3]:  # Get up to 3 occurrences
            start = max(0, match.start() - context_size // 2)
            end = min(len(text), match.end() + context_size // 2)
            context = text[start:end].strip()
            if context:
                contexts.append(context)
        
        # Strategy 2: Search with word variations
        words = concept_name.split()
        if len(words) > 1:
            for word in words:
                if len(word) > 3:
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    match = pattern.search(text)
                    if match:
                        start = max(0, match.start() - context_size // 3)
                        end = min(len(text), match.end() + context_size // 3)
                        ctx = text[start:end].strip()
                        if ctx and ctx not in contexts:
                            contexts.append(ctx)
                            break
        
        if contexts:
            # Return the longest context (most informative)
            return max(contexts, key=len)
        
        # Fallback: use chunks and find best matching chunk
        chunks = self._chunk_text(text, chunk_size=1500)
        best_chunk = ''
        best_score = 0
        
        for chunk in chunks:
            score = chunk.lower().count(concept_name.lower())
            # Also check individual words
            for word in concept_name.split():
                if len(word) > 3:
                    score += chunk.lower().count(word.lower()) * 0.5
            if score > best_score:
                best_score = score
                best_chunk = chunk
        
        return best_chunk if best_chunk else (chunks[0] if chunks else text[:context_size])

    def generate_questions(self, text: str, concepts: List[Dict],
                          knowledge_graph: Dict, num_questions: int = 10,
                          question_types: List[str] = None,
                          difficulty: str = 'mixed') -> List[Dict]:
        """
        Generate questions from the ENTIRE document text based on extracted concepts.
        Reads all pages: chunks the full text for comprehensive question generation.
        
        Args:
            text: Source document text (from ALL pages)
            concepts: List of extracted concept dictionaries
            knowledge_graph: The concept knowledge graph
            num_questions: Target number of questions
            question_types: List of ['mcq', 'true_false', 'short_answer']
            difficulty: 'easy', 'medium', 'hard', or 'mixed'
            
        Returns:
            List of question dictionaries
        """
        # Validate inputs
        if not concepts:
            raise ValueError("No concepts provided for question generation.")
        
        if question_types is None or len(question_types) == 0:
            question_types = ['mcq', 'true_false', 'short_answer']
        
        if not text:
            raise ValueError("No text provided for question generation.")
        
        print(f"[QuestionGen] Generating {num_questions} questions from {len(text)} chars of text")
        print(f"[QuestionGen] Using {len(concepts)} concepts, types: {question_types}")
        
        questions = []
        
        # Distribute questions across types
        if len(question_types) == 0:
            raise ValueError("Question types list is empty.")
        
        per_type = num_questions // len(question_types)
        remainder = num_questions % len(question_types)
        
        type_counts = {t: per_type for t in question_types}
        for i, t in enumerate(question_types):
            if i < remainder:
                type_counts[t] += 1
        
        # Generate questions for each type
        for q_type, count in type_counts.items():
            if count == 0:
                continue
            
            if q_type == 'mcq':
                type_questions = self._generate_mcq_questions(
                    text, concepts, knowledge_graph, count, difficulty
                )
            elif q_type == 'true_false':
                type_questions = self._generate_true_false_questions(
                    text, concepts, count, difficulty
                )
            elif q_type == 'short_answer':
                type_questions = self._generate_short_answer_questions(
                    text, concepts, count, difficulty
                )
            else:
                continue
            
            questions.extend(type_questions)
        
        # Shuffle questions
        random.shuffle(questions)
        
        # Add unique IDs
        for i, q in enumerate(questions):
            q['id'] = f'q_{i}'
        
        return questions
    
    def _generate_mcq_questions(self, text: str, concepts: List[Dict],
                                knowledge_graph: Dict, count: int,
                                difficulty: str) -> List[Dict]:
        """Generate Multiple Choice Questions from the ENTIRE document"""
        questions = []
        self._load_model()
        
        # Select more concepts than needed to ensure we get enough valid questions
        selected_concepts = self._select_concepts_for_difficulty(concepts, difficulty, count * 2)
        
        for concept in selected_concepts:
            if len(questions) >= count:
                break
                
            concept_name = concept['name']
            # Use enhanced context extraction that searches the full document
            context = concept.get('context', self._get_best_context_for_concept(text, concept_name))
            
            if not context or len(context.strip()) < 30:
                continue
            
            # Generate question using model or rules
            if self._initialized:
                question_text = self._generate_question_with_model(context, concept_name, 'mcq')
            else:
                question_text = self._generate_question_rule_based(context, concept_name, 'mcq')
            
            if not question_text:
                continue
            
            # Generate answer and distractors
            correct_answer = self._extract_answer(text, concept_name, question_text)
            distractors = self._generate_distractors(
                concept_name, concepts, knowledge_graph, correct_answer
            )
            
            # Ensure we have exactly 3 distractors
            while len(distractors) < 3:
                distractors.append(f"None of the above ({len(distractors) + 1})")
            
            # Create options list
            options = [correct_answer] + distractors[:3]
            random.shuffle(options)
            
            # Validate: Ensure only one correct answer in MCQ
            correct_count = sum(1 for opt in options if opt.lower() == correct_answer.lower())
            if correct_count != 1:
                # Regenerate if duplicate found
                distractors = self._generate_distractors(
                    concept_name, concepts, knowledge_graph, correct_answer
                )
                while len(distractors) < 3:
                    distractors.append(f"None of the above ({len(distractors) + 1})")
                options = [correct_answer] + distractors[:3]
                random.shuffle(options)
            
            # Generate a detailed explanation for learning
            explanation = self._generate_detailed_explanation(context, concept_name, correct_answer)
            
            questions.append({
                'text': question_text,
                'type': 'mcq',
                'options': options,
                'correct_answer': correct_answer,
                'concept': concept_name,
                'difficulty': self._determine_difficulty(concept, difficulty),
                'explanation': explanation
            })
        
        return questions[:count]
    
    def _generate_true_false_questions(self, text: str, concepts: List[Dict],
                                       count: int, difficulty: str) -> List[Dict]:
        """Generate True/False Questions from full document"""
        questions = []
        self._load_model()
        
        selected_concepts = self._select_concepts_for_difficulty(concepts, difficulty, count * 2)
        
        for concept in selected_concepts:
            if len(questions) >= count:
                break
            
            concept_name = concept['name']
            context = concept.get('context', self._get_best_context_for_concept(text, concept_name))
            
            if not context:
                continue
            
            # Decide if generating true or false statement
            is_true = random.choice([True, False])
            
            if self._initialized:
                statement = self._generate_statement_with_model(context, concept_name, is_true)
            else:
                statement = self._generate_statement_rule_based(context, concept_name, is_true)
            
            if not statement:
                continue
            
            questions.append({
                'text': statement,
                'type': 'true_false',
                'correct_answer': 'True' if is_true else 'False',
                'concept': concept_name,
                'difficulty': self._determine_difficulty(concept, difficulty),
                'explanation': f"This statement is {'true' if is_true else 'false'} based on: {context[:200]}..."
            })
        
        return questions[:count]
    
    def _generate_short_answer_questions(self, text: str, concepts: List[Dict],
                                         count: int, difficulty: str) -> List[Dict]:
        """Generate Short Answer Questions from full document"""
        questions = []
        self._load_model()
        
        selected_concepts = self._select_concepts_for_difficulty(concepts, difficulty, count * 2)
        
        for concept in selected_concepts:
            if len(questions) >= count:
                break
                
            concept_name = concept['name']
            context = concept.get('context', self._get_best_context_for_concept(text, concept_name))
            
            if not context:
                continue
            
            if self._initialized:
                question_text = self._generate_question_with_model(context, concept_name, 'short_answer')
                answer = self._generate_answer_with_model(context, question_text)
            else:
                question_text = self._generate_question_rule_based(context, concept_name, 'short_answer')
                answer = self._extract_answer(text, concept_name, question_text)
            
            if not question_text or not answer:
                continue
            
            questions.append({
                'text': question_text,
                'type': 'short_answer',
                'correct_answer': answer,
                'concept': concept_name,
                'difficulty': self._determine_difficulty(concept, difficulty),
                'explanation': f"Key information: {context[:300]}...",
                'keywords': self._extract_answer_keywords(answer)
            })
        
        return questions[:count]
    
    def _generate_question_with_model(self, context: str, concept: str, 
                                      q_type: str) -> Optional[str]:
        """Generate question using transformer model"""
        try:
            import torch
            
            # Create prompt based on question type
            if q_type == 'mcq':
                prompt = f"Generate a multiple choice question about {concept} based on this context: {context}"
            elif q_type == 'short_answer':
                prompt = f"Generate a short answer question about {concept} based on this context: {context}"
            else:
                prompt = f"Generate a question about {concept} from: {context}"
            
            inputs = self._tokenizer(prompt, return_tensors="pt", 
                                    max_length=512, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7
                )
            
            question = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up question
            question = question.strip()
            if not question.endswith('?'):
                question += '?'
            
            return question
            
        except Exception as e:
            print(f"Model generation failed: {e}")
            return self._generate_question_rule_based(context, concept, q_type)
    
    def _generate_question_rule_based(self, context: str, concept: str,
                                      q_type: str) -> Optional[str]:
        """Generate question using rule-based templates that create meaningful questions"""
        templates = {
            'mcq': [
                f"Which of the following best describes {concept}?",
                f"What is {concept}?",
                f"Which statement about {concept} is correct?",
                f"The term {concept} refers to:",
                f"What is the main characteristic of {concept}?",
                f"How is {concept} defined in this context?",
                f"Which definition best fits the concept of {concept}?",
                f"What role does {concept} play in the displayed information?",
                f"Which of the following is an example of {concept}?",
                f"What are the key attributes of {concept}?",
                f"According to the passage, {concept} is primarily associated with:",
                f"What distinguishes {concept} from similar concepts?",
            ],
            'short_answer': [
                f"Define {concept} in your own words.",
                f"Explain what {concept} means and its significance.",
                f"What is the purpose and importance of {concept}?",
                f"Describe {concept} and how it works.",
                f"How does {concept} function in this context?",
                f"What are the main characteristics of {concept}?",
                f"Explain the relationship between {concept} and the main topic.",
                f"Why is {concept} important to understanding this topic?",
                f"Summarize the key points about {concept}.",
                f"What are the practical applications of {concept}?",
            ],
            'true_false': [
                f"{concept} is a fundamental concept in this field.",
                f"The definition of {concept} is clearly explained in the passage.",
            ]
        }
        
        template_list = templates.get(q_type, templates['mcq'])
        return random.choice(template_list)
    
    def _generate_statement_with_model(self, context: str, concept: str,
                                       is_true: bool) -> Optional[str]:
        """Generate true/false statement using model"""
        try:
            import torch
            
            if is_true:
                prompt = f"Generate a true factual statement about {concept} based on: {context}"
            else:
                prompt = f"Generate a false statement about {concept} that contradicts: {context}"
            
            inputs = self._tokenizer(prompt, return_tensors="pt",
                                    max_length=512, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=3,
                    temperature=0.8
                )
            
            statement = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return statement.strip()
            
        except Exception as e:
            return self._generate_statement_rule_based(context, concept, is_true)
    
    def _generate_statement_rule_based(self, context: str, concept: str,
                                       is_true: bool) -> Optional[str]:
        """Generate statement using rules - improved for better True/False questions"""
        # Extract a sentence containing the concept
        sentences = re.split(r'[.!?]', context)
        relevant_sentences = [s for s in sentences if concept.lower() in s.lower()]
        
        if not relevant_sentences:
            # Create statement from concept if no relevant sentence found
            if is_true:
                return f"{concept} is an important concept mentioned in the material."
            else:
                return f"{concept} has no relevance to the main topic."
        
        sentence = relevant_sentences[0].strip()
        
        if is_true:
            # Ensure it's a proper statement
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            return sentence
        else:
            # Create false statement by intelligently negating or modifying
            # Make it sound plausible but incorrect
            false_statements = [
                f"{concept} is the opposite of what the passage states.",
                f"According to the material, {concept} is something completely different.",
                f"The passage suggests that {concept} plays no role in this context.",
                f"{concept} is primarily associated with concepts not mentioned in the text.",
                f"The definition of {concept} provided contradicts the main thesis.",
            ]
            
            # Sometimes use sentence modification
            if random.random() > 0.5 and len(sentence) > 20:
                # Simple negation
                negations = [' is not ', ' never ', ' rarely ', ' does not ', ' cannot ']
                # Find a good insertion point
                words = sentence.split()
                if len(words) > 3:
                    insert_pos = random.randint(2, min(4, len(words) - 1))
                    words.insert(insert_pos, random.choice(negations).strip())
                    return ' '.join(words)
            
            return random.choice(false_statements)
    
    def _generate_answer_with_model(self, context: str, question: str) -> Optional[str]:
        """Generate answer using model"""
        try:
            import torch
            
            prompt = f"Answer this question based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:"
            
            inputs = self._tokenizer(prompt, return_tensors="pt",
                                    max_length=512, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=3
                )
            
            answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer.strip()
            
        except Exception as e:
            return None
    
    def _extract_answer(self, text: str, concept: str, question: str) -> str:
        """Extract answer from text"""
        # Find sentences containing the concept
        sentences = re.split(r'[.!?]', text)
        relevant = [s.strip() for s in sentences 
                   if concept.lower() in s.lower() and len(s) > 20]
        
        if relevant:
            # Return the most informative sentence
            return max(relevant, key=len)[:200]
        
        return concept
    
    def _generate_distractors(self, concept: str, all_concepts: List[Dict],
                              knowledge_graph: Dict, correct_answer: str) -> List[str]:
        """Generate distractor options for MCQ"""
        distractors = []
        
        self._load_sentence_model()
        
        # Method 1: Use related concepts from knowledge graph
        edges = knowledge_graph.get('edges', [])
        nodes = knowledge_graph.get('nodes', [])
        
        # Find node ID for current concept
        concept_node = None
        for node in nodes:
            if node['label'].lower() == concept.lower():
                concept_node = node['id']
                break
        
        if concept_node:
            # Get neighboring concepts
            neighbors = set()
            for edge in edges:
                if edge['from'] == concept_node:
                    neighbors.add(edge['to'])
                elif edge['to'] == concept_node:
                    neighbors.add(edge['from'])
            
            for node in nodes:
                if node['id'] in neighbors and node['label'].lower() != concept.lower():
                    distractors.append(node['label'])
        
        # Method 2: Use similar concepts based on embeddings
        if self._sentence_model and len(distractors) < 3:
            try:
                concept_names = [c['name'] for c in all_concepts 
                               if c['name'].lower() != concept.lower()]
                
                if concept_names:
                    embeddings = self._sentence_model.encode([concept] + concept_names)
                    concept_emb = embeddings[0]
                    
                    import numpy as np
                    similarities = []
                    for i, name in enumerate(concept_names):
                        sim = np.dot(concept_emb, embeddings[i + 1]) / (
                            np.linalg.norm(concept_emb) * np.linalg.norm(embeddings[i + 1])
                        )
                        similarities.append((name, sim))
                    
                    # Sort by similarity (use moderately similar for good distractors)
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    for name, sim in similarities[1:6]:  # Skip most similar
                        if name not in distractors:
                            distractors.append(name)
                            
            except Exception as e:
                print(f"Embedding-based distractor generation failed: {e}")
        
        # Method 3: Fallback - use random other concepts
        if len(distractors) < 3:
            random_concepts = [c['name'] for c in all_concepts 
                              if c['name'].lower() != concept.lower() 
                              and c['name'] not in distractors]
            random.shuffle(random_concepts)
            distractors.extend(random_concepts[:3 - len(distractors)])
        
        return distractors[:3]
    
    def _select_concepts_for_difficulty(self, concepts: List[Dict],
                                        difficulty: str, count: int) -> List[Dict]:
        """Select concepts matching the target difficulty"""
        if difficulty == 'mixed':
            selected = random.sample(concepts, min(count, len(concepts)))
        else:
            # Filter by score (higher score = easier concept)
            if difficulty == 'easy':
                filtered = [c for c in concepts if c.get('score', 0.5) > 0.6]
            elif difficulty == 'medium':
                filtered = [c for c in concepts if 0.3 <= c.get('score', 0.5) <= 0.6]
            else:  # hard
                filtered = [c for c in concepts if c.get('score', 0.5) < 0.3]
            
            if len(filtered) < count:
                filtered = concepts  # Fallback to all concepts
            
            selected = random.sample(filtered, min(count, len(filtered)))
        
        return selected
    
    def _determine_difficulty(self, concept: Dict, target: str) -> str:
        """Determine the difficulty level of a question"""
        if target != 'mixed':
            return target
        
        score = concept.get('score', 0.5)
        if score > 0.6:
            return 'easy'
        elif score > 0.3:
            return 'medium'
        else:
            return 'hard'
    
    def _generate_detailed_explanation(self, context: str, concept: str, answer: str) -> str:
        """
        Generate a detailed, educational explanation for the correct answer.
        This helps students learn from both correct and incorrect answers.
        """
        # Clean and truncate for display
        display_answer = answer[:200] + "..." if len(answer) > 200 else answer
        
        # Find the most relevant sentence(s) from context
        sentences = re.split(r'[.!?]', context)
        relevant = [s.strip() for s in sentences 
                    if concept.lower() in s.lower() and len(s.strip()) > 20]
        
        if relevant:
            key_info = '. '.join(relevant[:2]).strip()
            if not key_info.endswith('.'):
                key_info += '.'
        else:
            key_info = context[:300].strip()
            if not key_info.endswith('.'):
                key_info += '...'
        
        explanation = (
            f"The correct answer is: {display_answer}\n\n"
            f"Explanation: {key_info}\n\n"
            f"Key Concept: {concept} - Understanding this concept is important "
            f"as it relates to the core material covered in the document."
        )
        
        return explanation

    def _generate_explanation(self, context: str, answer: str) -> str:
        """Generate explanation for the correct answer (legacy)"""
        display_answer = answer[:200] + "..." if len(answer) > 200 else answer
        display_context = context[:250] + "..." if len(context) > 250 else context
        return f"The correct answer is '{display_answer}'. This is based on the following information: {display_context}"
    
    def _extract_answer_keywords(self, answer: str) -> List[str]:
        """Extract key words from answer for partial matching"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'and',
                     'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        words = re.findall(r'\b\w+\b', answer.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:5]
    
    def _get_context(self, text: str, concept: str, chars: int = 500) -> str:
        """Get context around a concept in text"""
        pattern = re.compile(re.escape(concept), re.IGNORECASE)
        match = pattern.search(text)
        
        if not match:
            return ""
        
        start = max(0, match.start() - chars // 2)
        end = min(len(text), match.end() + chars // 2)
        
        return text[start:end].strip()
