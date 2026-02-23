"""Test script to verify mastery chart data generation"""

# Simulate concept mastery data
concept_mastery = {
    'Python Programming': 0.85,
    'Data Structures': 0.72,
    'Algorithms': 0.65,
    'Machine Learning': 0.58,
    'Web Development': 0.90
}

# Simulate quiz history
quiz_history = [
    {'score_percentage': 75, 'concepts': ['Python Programming', 'Data Structures']},
    {'score_percentage': 82, 'concepts': ['Python Programming', 'Algorithms']},
    {'score_percentage': 68, 'concepts': ['Machine Learning']},
    {'score_percentage': 90, 'concepts': ['Web Development']},
]

print("Testing mastery chart data generation...")
print(f"Concept Mastery: {concept_mastery}")
print(f"Quiz History: {len(quiz_history)} quizzes")

# Test the calculation
mastery_labels = [str(c)[:40] for c in list(concept_mastery.keys())[:10]]
mastery_data = []

for c in list(concept_mastery.keys())[:10]:
    val = concept_mastery.get(c, 0)
    try:
        val = float(val) * 100
        val = min(max(val, 0), 100)  # Clamp 0-100
    except (ValueError, TypeError):
        val = 0
    mastery_data.append(round(val, 1))

print("\n=== Generated Data ===")
print(f"Labels: {mastery_labels}")
print(f"Data: {mastery_data}")
print(f"Length check: labels={len(mastery_labels)}, data={len(mastery_data)}")

# Test with numpy
try:
    import numpy as np
    print("\n✓ NumPy is available")
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("\n✗ NumPy is NOT available - install with: pip install numpy")

print("\n=== Test Complete ===")
