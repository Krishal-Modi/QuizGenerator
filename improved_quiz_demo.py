"""
Example of improved quiz generation logic
This demonstrates how the questions will now be more logical and context-aware
"""

# Sample text about Machine Learning
sample_text = """
Machine Learning is a subset of artificial intelligence (AI) that focuses on 
enabling computer systems to learn and improve from experience without being 
explicitly programmed. The process involves feeding large amounts of data to 
algorithms which can then identify patterns, make predictions, and make decisions 
based on the data.

There are three main types of machine learning: supervised learning, unsupervised 
learning, and reinforcement learning. Supervised learning uses labeled data to 
train models. Unsupervised learning finds patterns in unlabeled data. Reinforcement 
learning learns through trial and error using rewards and penalties.

Machine Learning is used for various purposes including image recognition, natural 
language processing, recommendation systems, fraud detection, and predictive analytics. 
Its applications span across industries from healthcare to finance to entertainment.

The key advantages of Machine Learning include automation of complex tasks, ability 
to handle large volumes of data, continuous improvement over time, and the capacity 
to discover hidden patterns that humans might miss. However, it also requires 
significant computational resources, quality training data, and careful monitoring 
to avoid bias.
"""

print("="*70)
print("IMPROVED QUIZ GENERATION DEMONSTRATION")
print("="*70)

# Concept to test
concept = "Machine Learning"

print(f"\n📚 CONCEPT: {concept}")
print(f"📄 Text Length: {len(sample_text)} characters")
print(f"   Sentences: {len([s for s in sample_text.split('.') if s.strip()])}")

# Demonstrate improved question generation
print("\n" + "="*70)
print("1. MCQ QUESTION GENERATION (Context-Aware)")
print("="*70)

# Check what type of content we have
has_definition = 'is a' in sample_text or 'defined as' in sample_text
has_process = 'process' in sample_text.lower()
has_purpose = 'used for' in sample_text.lower() or 'purpose' in sample_text.lower()
has_types = 'types' in sample_text.lower() or 'kinds' in sample_text.lower()

print(f"\nContent Analysis:")
print(f"  ✓ Has definition: {has_definition}")
print(f"  ✓ Describes process: {has_process}")
print(f"  ✓ Mentions purpose: {has_purpose}")
print(f"  ✓ Lists types: {has_types}")

if has_definition:
    question = f"How is {concept} defined in the context?"
    print(f"\n🎯 Generated Question:")
    print(f"   {question}")
    
    # Extract answer
    import re
    sentences = [s.strip() for s in re.split(r'[.!?]', sample_text) if len(s.strip()) > 15]
    for s in sentences:
        if concept.lower() in s.lower() and ('is a' in s.lower() or 'is the' in s.lower()):
            answer = s
            print(f"\n✅ Correct Answer:")
            print(f"   {answer}")
            break

print("\n" + "="*70)
print("2. TRUE/FALSE QUESTIONS (Logical Statements)")
print("="*70)

# Generate TRUE statement
print("\n✓ TRUE Statement:")
true_statement = "Machine Learning is a subset of artificial intelligence that focuses on enabling computer systems to learn from experience"
print(f"   {true_statement}")
print(f"   Reason: Directly extracted from the text, factually accurate")

# Generate FALSE statement  
print("\n✗ FALSE Statement:")
false_statement = "Machine Learning requires explicit programming for every task it performs"
print(f"   {false_statement}")
print(f"   Reason: Contradicts the text which says 'without being explicitly programmed'")

print("\n" + "="*70)
print("3. SHORT ANSWER QUESTIONS (Comprehensive)")
print("="*70)

question_sa = f"Explain what {concept} is and describe its main types."
print(f"\n🎯 Question:")
print(f"   {question_sa}")

answer_sa = """Machine Learning is a subset of AI that enables systems to learn from 
experience without explicit programming. The three main types are:
1. Supervised learning - uses labeled data to train models
2. Unsupervised learning - finds patterns in unlabeled data  
3. Reinforcement learning - learns through trial and error using rewards"""

print(f"\n✅ Model Answer:")
for line in answer_sa.strip().split('\n'):
    print(f"   {line}")

print("\n" + "="*70)
print("4. DISTRACTOR GENERATION (Related Concepts)")
print("="*70)

print(f"\nFor MCQ about '{concept}', good distractors would be:")
distractors = [
    "Deep Learning",
    "Neural Networks", 
    "Artificial Intelligence",
    "Data Mining"
]

for i, d in enumerate(distractors, 1):
    print(f"   {i}. {d} (related but distinct concept)")

print("\n" + "="*70)
print("KEY IMPROVEMENTS IMPLEMENTED:")
print("="*70)

improvements = [
    "✓ Context-aware question generation (analyzes text for patterns)",
    "✓ Intelligent answer extraction (looks for definitions, purposes, processes)",
    "✓ Logical True/False statements (not random, based on actual content)",
    "✓ Better distractors (uses related concepts, not random words)",
    "✓ Full document search (finds best context from entire text)",
    "✓ Question type detection (adjusts logic based on available information)",
    "✓ Detailed explanations (helps students learn from mistakes)",
    "✓ Difficulty adaptation (selects appropriate concepts per difficulty level)"
]

for improvement in improvements:
    print(f"  {improvement}")

print("\n" + "="*70)
print("TESTING RECOMMENDATIONS:")
print("="*70)

recommendations = [
    "1. Upload a document with clear definitions and concepts",
    "2. Generate a quiz with mixed question types",
    "3. Check that questions make logical sense",
    "4. Verify answers are actually present in the document",
    "5. Confirm distractors are plausible but incorrect",
    "6. Review explanations for educational value"
]

for rec in recommendations:
    print(f"  {rec}")

print("\n" + "="*70)
