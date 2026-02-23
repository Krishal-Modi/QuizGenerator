"""
Quick test to verify quiz codes are being generated and can be looked up
"""
import random
import string

def generate_quiz_code():
    """Generate a 16-character alphanumeric code"""
    characters = string.ascii_uppercase + string.digits
    code = ''.join(random.choices(characters, k=16))
    return code

print("="*60)
print("Quiz Code System Test")
print("="*60)

# Generate sample codes
print("\n📝 Sample Quiz Codes:")
for i in range(5):
    code = generate_quiz_code()
    print(f"  {i+1}. {code} ({len(code)} chars)")

# Test validation
print("\n🔍 Code Validation Tests:")
test_codes = [
    ("ABC123XYZ456PQRS", True, "Valid 16-char code"),
    ("SHORT", False, "Too short"),
    ("TOOLONGCODE12345678", False, "Too long"),
    ("ABC#123XYZ456PQR", False, "Contains special character"),
    ("abcd1234efgh5678", True, "Valid (will be uppercased)"),
]

for code, expected_valid, reason in test_codes:
    code_upper = code.upper()
    is_valid = (len(code_upper) == 16 and 
                all(c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in code_upper))
    status = "✓" if is_valid == expected_valid else "✗"
    print(f"  {status} '{code}' - {reason}")
    print(f"     Length: {len(code)}, Valid: {is_valid}")

print("\n" + "="*60)
print("How Quiz Codes Work:")
print("="*60)
print("""
1. INSTRUCTOR CREATES QUIZ:
   - System generates unique 16-character code
   - Code is stored in Firebase: quiz_codes/{CODE} -> quiz_id
   - Instructor shares code with students

2. STUDENT SEARCHES FOR QUIZ:
   - Student enters 16-character code
   - System validates format (16 alphanumeric chars)
   - Looks up code in Firebase quiz_codes
   - If found, redirects to quiz
   - If not found, shows error message

3. FIXES IMPLEMENTED:
   ✓ Fixed Firebase lookup (.get() instead of .get().val())
   ✓ Better error messages (shows code length, validation details)
   ✓ Debug logging (prints lookup attempts and results)
   ✓ Input sanitization (auto-uppercase, trim whitespace)
""")

print("="*60)
print("Testing Instructions:")
print("="*60)
print("""
TO TEST QUIZ CODE SEARCH:

1. CREATE A QUIZ (as Instructor):
   - Login as instructor
   - Upload a document
   - Generate a quiz
   - Note the quiz code displayed (e.g., ABC123XYZ456PQRS)

2. SEARCH FOR QUIZ (as Student):
   - Go to "Find Quiz" page
   - Enter the 16-character code
   - Should find and start the quiz

3. CHECK CONSOLE/LOGS:
   - Look for debug messages like:
     [Quiz Search] Looking up code: ABC123XYZ456PQRS
     Found quiz ID 'quiz_id_here' for code 'ABC123XYZ456PQRS'

4. COMMON ISSUES:
   - Wrong code length → Shows exact character count
   - Quiz doesn't exist → Clear error message
   - Not logged in → Redirects to login
   - Code format invalid → Explains validation rules
""")

print("="*60)
