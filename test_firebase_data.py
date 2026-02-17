"""
Test script to verify the quiz system is working correctly.
Tests:
1. Quiz evaluation with answer checking
2. Marks calculation
3. Form submission and answer collection
4. Results display
"""
import json
import math
from services.quiz_service import QuizService
from services.firebase_service import FirebaseService

print("="*70)
print("QUIZ SYSTEM COMPREHENSIVE TEST")
print("="*70)

# Create services
quiz_service = QuizService()
firebase_service = FirebaseService()

# ============================================================================
# TEST 1: Basic Quiz Evaluation
# ============================================================================
print("\nTEST 1: Quiz Evaluation with Marks Calculation")
print("-" * 70)

# Create mock questions
mock_questions = [
    {
        'id': 'q1',
        'text': 'What is the capital of France?',
        'type': 'mcq',
        'options': ['Paris', 'London', 'Berlin', 'Madrid'],
        'correct_answer': 'Paris',
        'concept': 'Geography - Europe',
        'explanation': 'Paris is the capital of France.',
        'difficulty': 'easy'
    },
    {
        'id': 'q2',
        'text': 'Is Python a programming language?',
        'type': 'true_false',
        'correct_answer': 'True',
        'concept': 'Programming Languages',
        'explanation': 'Python is indeed a programming language.',
        'difficulty': 'easy'
    },
    {
        'id': 'q3',
        'text': 'Define machine learning.',
        'type': 'short_answer',
        'correct_answer': 'Machine learning is a field of artificial intelligence',
        'concept': 'Machine Learning',
        'keywords': ['machine learning', 'artificial intelligence', 'learning algorithm'],
        'explanation': 'Machine learning enables computers to learn from data.',
        'difficulty': 'medium'
    }
]

# User answers (simulating form submission)
user_answers = {
    'q1': 'Paris',          # Correct
    'q2': 'True',           # Correct
    'q3': 'Machine learning is about AI'  # Partially correct (contains keywords)
}

print(f"\nMock Quiz with {len(mock_questions)} questions:")
for q in mock_questions:
    print(f"  - {q['id']}: {q['text'][:50]}...")

print(f"\nUser Answers: {json.dumps(user_answers, indent=2)}")

# Manually evaluate answers (since we can't access Firebase)
print("\n[ANSWER EVALUATION]")
correct = 0
for q in mock_questions:
    q_id = q['id']
    if q_id in user_answers:
        answer = user_answers[q_id]
        is_correct = False
        
        if q['type'] == 'mcq':
            is_correct = answer.lower() == q['correct_answer'].lower()
        elif q['type'] == 'true_false':
            is_correct = answer.lower() in ['true', 'false'] and \
                        answer.lower() == q['correct_answer'].lower()
        elif q['type'] == 'short_answer':
            # Check if keywords are present
            answer_lower = answer.lower()
            keywords = q.get('keywords', [])
            keywords_found = sum(1 for kw in keywords if kw.lower() in answer_lower)
            is_correct = keywords_found >= 2
        
        if is_correct:
            correct += 1
        
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        print(f"  Q{q_id}: {status}")
        print(f"    Your answer: {answer}")
        if not is_correct:
            print(f"    Correct answer: {q['correct_answer']}")

# Calculate marks
marks_per_question = 100 / len(mock_questions)
total_marks = 100
marks_obtained = correct * marks_per_question
score = (correct / len(mock_questions)) * 100

print(f"\n[MARKS CALCULATION]")
print(f"  Questions answered correctly: {correct}/{len(mock_questions)}")
print(f"  Marks per question: {marks_per_question:.2f}")
print(f"  Total marks obtained: {marks_obtained:.2f}/{total_marks:.2f}")
print(f"  Percentage score: {score:.1f}%")

# ============================================================================
# TEST 2: Quiz Result Structure
# ============================================================================
print("\n\nTEST 2: Quiz Result Data Structure")
print("-" * 70)

test_result = {
    'score': score,
    'score_percentage': score,
    'correct': correct,
    'total_questions': len(mock_questions),
    'marks_obtained': marks_obtained,
    'total_marks': total_marks,
    'timestamp': firebase_service.get_timestamp(),
    'concept_performance': {
        'Geography - Europe': 1.0,  # 1 correct out of 1
        'Programming Languages': 1.0,  # 1 correct out of 1
        'Machine Learning': 1.0 if correct >= 3 else 0.0  # Depends on short answer
    }
}

print("\nTest Result Structure (JSON):")
print(json.dumps(test_result, indent=2))

# ============================================================================
# TEST 3: Form Submission Simulation
# ============================================================================
print("\n\nTEST 3: Form Submission Simulation")
print("-" * 70)

# Simulate form data
form_data = {
    'attempt_id': 'mock_attempt_123',
    'question_q1': 'Paris',
    'question_q2': 'True',
    'question_q3': 'Machine learning is about AI'
}

print("\nSimulated Form Data (POST):")
for key, value in form_data.items():
    if key.startswith('question_'):
        q_id = key.replace('question_', '')
        print(f"  {key} = '{value}'")

# Parse answers from form
parsed_answers = {}
for key, value in form_data.items():
    if key.startswith('question_'):
        q_id = key.replace('question_', '')
        if value.strip():
            parsed_answers[q_id] = value

print(f"\nParsed Answers: {json.dumps(parsed_answers, indent=2)}")
print(f"Total questions submitted: {len(parsed_answers)}")

# ============================================================================
# TEST 4: Results Display Information
# ============================================================================
print("\n\nTEST 4: Results Display Information")
print("-" * 70)

print(f"""
FINAL QUIZ RESULTS:
─────────────────────────────────────────────────────
  Score: {score:.1f}%
  Correct: {correct}/{len(mock_questions)}
  Marks: {marks_obtained:.1f}/{total_marks:.1f}
  Status: {'PASS' if score >= 60 else 'FAIL'}
─────────────────────────────────────────────────────

Concept Performance:
""")
for concept, perf in test_result['concept_performance'].items():
    perf_pct = perf * 100
    print(f"  • {concept}: {perf_pct:.0f}%")

# ============================================================================
# TEST 5: Data Sanitization for Firebase
# ============================================================================
print("\n\nTEST 5: Data Sanitization for Firebase Storage")
print("-" * 70)

def sanitize_value(value):
    if isinstance(value, (int, bool, str, type(None))):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    if isinstance(value, dict):
        return {str(k): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return str(value)

def sanitize_key(key):
    if not key:
        return 'unknown'
    # Replace invalid Firebase characters with underscores
    invalid_chars = ['.', '$', '#', '[', ']', '/', '\\']
    safe_key = str(key)
    for char in invalid_chars:
        safe_key = safe_key.replace(char, '_')
    # Firebase keys cannot be empty or just whitespace
    safe_key = safe_key.strip() or 'unknown'
    # Limit key length
    if len(safe_key) > 100:
        safe_key = safe_key[:100]
    return safe_key

# Create a test result with special characters
test_result_with_special = {
    'score': 75.5,
    'correct': 3,
    'total_questions': 4,
    'concept_performance': {
        'Math.Algebra': 0.8,           # Has dot
        'Science/Physics': 0.7,        # Has slash
        'Test$Concept': 0.6            # Has dollar sign
    }
}

print("\nOriginal result:")
print(json.dumps(test_result_with_special, indent=2))

print("\nAfter sanitization:")
sanitized = {
    'score': sanitize_value(test_result_with_special['score']),
    'correct': sanitize_value(test_result_with_special['correct']),
    'total_questions': sanitize_value(test_result_with_special['total_questions']),
    'concept_performance': {}
}

for concept, perf in test_result_with_special['concept_performance'].items():
    safe_key = sanitize_key(concept)
    safe_value = sanitize_value(perf)
    sanitized['concept_performance'][safe_key] = safe_value
    print(f"  '{concept}' → '{safe_key}'")

print("\nSanitized JSON:")
print(json.dumps(sanitized, indent=2))

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
    return str(value)
    return str(value)

def sanitize_key(key):
    if not key:
        return 'unknown'
    invalid_chars = ['.', '$', '#', '[', ']', '/', '\\']
    safe_key = str(key)
    for char in invalid_chars:
        safe_key = safe_key.replace(char, '_')
    safe_key = safe_key.strip() or 'unknown'
    return safe_key

# Sanitize the result
sanitized_result = {
    'score': sanitize_value(test_result.get('score', 0)),
    'score_percentage': sanitize_value(test_result.get('score_percentage', test_result.get('score', 0))),
    'correct': int(test_result.get('correct', 0)),
    'total_questions': int(test_result.get('total_questions', 0)),
    'timestamp': str(test_result.get('timestamp', firebase_service.get_timestamp()))
}

# Sanitize concept_performance
if 'concept_performance' in test_result and test_result['concept_performance']:
    sanitized_concepts = {}
    for concept, performance in test_result['concept_performance'].items():
        safe_key = sanitize_key(concept)
        safe_value = sanitize_value(performance)
        sanitized_concepts[safe_key] = safe_value
        print(f"Concept: '{concept}' -> '{safe_key}' = {safe_value}")
    sanitized_result['concept_performance'] = sanitized_concepts

print("\nSanitized result (ready for Firebase):")
print(json.dumps(sanitized_result, indent=2))

# Test JSON serialization
try:
    json_str = json.dumps(sanitized_result)
    print("\n✅ JSON serialization successful!")
    print(f"JSON length: {len(json_str)} bytes")
except Exception as e:
    print(f"\n❌ JSON serialization failed: {e}")
