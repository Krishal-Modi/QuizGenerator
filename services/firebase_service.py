"""
Firebase Service - Database operations with Firebase
"""
import os
import json
import secrets
import string
from datetime import datetime
from typing import Dict, List, Optional, Any


class FirebaseService:
    """
    Service for Firebase Realtime Database and Authentication operations.
    
    Firebase Initialization:
    1. Create a Firebase project at console.firebase.google.com
    2. Enable Authentication (Email/Password)
    3. Create Realtime Database
    4. Download service account key for Admin SDK
    5. Set environment variables in .env file
    """
    
    def __init__(self):
        self._db = None
        self._auth = None
        self._initialized = False
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK and Pyrebase"""
        try:
            import firebase_admin
            from firebase_admin import credentials, db
            import pyrebase
            
            # Check if already initialized
            if not firebase_admin._apps:
                # Initialize Admin SDK
                admin_sdk_path = os.getenv('FIREBASE_ADMIN_SDK_PATH', 'config/firebase-admin-sdk.json')
                if os.path.exists(admin_sdk_path):
                    cred = credentials.Certificate(admin_sdk_path)
                    firebase_admin.initialize_app(cred, {
                        'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
                    })
            
            self._db = db
            
            # Initialize Pyrebase for client-side auth
            config = {
                'apiKey': os.getenv('FIREBASE_API_KEY'),
                'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN'),
                'projectId': os.getenv('FIREBASE_PROJECT_ID'),
                'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
                'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
                'appId': os.getenv('FIREBASE_APP_ID'),
                'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
            }
            
            firebase = pyrebase.initialize_app(config)
            self._auth = firebase.auth()
            self._initialized = True
            
        except Exception as e:
            print(f"Warning: Firebase not initialized: {e}")
            print("Running in demo mode with mock data.")
            self._initialized = False
    
    def get_timestamp(self) -> str:
        """Get current timestamp as ISO string"""
        return datetime.utcnow().isoformat()
    
    # ===================== AUTHENTICATION =====================
    
    def sign_in(self, email: str, password: str) -> Optional[Dict]:
        """Sign in user with email and password"""
        if not self._initialized:
            # Demo mode
            return {
                'localId': 'demo_user_1',
                'email': email,
                'idToken': 'demo_token'
            }
        
        try:
            if not email or not password:
                raise Exception("Email and password are required.")
            
            user = self._auth.sign_in_with_email_and_password(email, password)
            return user
        except Exception as e:
            error_msg = str(e)
            # Re-raise with the original error message for proper error handling
            raise Exception(error_msg)
    
    def create_user(self, email: str, password: str) -> Optional[Dict]:
        """Create new user account"""
        if not self._initialized:
            return {'localId': 'demo_user_1', 'email': email}
        
        try:
            user = self._auth.create_user_with_email_and_password(email, password)
            return user
        except Exception as e:
            raise Exception(f"User creation failed: {str(e)}")
    
    def send_password_reset(self, email: str):
        """Send password reset email"""
        if not self._initialized:
            return True
        
        try:
            self._auth.send_password_reset_email(email)
            return True
        except Exception as e:
            raise Exception(f"Password reset failed: {str(e)}")
    
    # ===================== USER DATA =====================
    
    def get_user_data(self, user_id: str) -> Dict:
        """Get user profile and data"""
        if not self._initialized:
            return self._get_demo_user_data()
        
        try:
            ref = self._db.reference(f'users/{user_id}')
            data = ref.get()
            return data or {}
        except Exception as e:
            print(f"Error getting user data: {e}")
            return {}
    
    def save_user_data(self, user_id: str, data: Dict):
        """Save user profile data"""
        if not self._initialized:
            return True
        
        try:
            ref = self._db.reference(f'users/{user_id}')
            ref.set(data)
            return True
        except Exception as e:
            raise Exception(f"Failed to save user data: {str(e)}")
    
    def update_user_data(self, user_id: str, data: Dict):
        """Update specific user data fields"""
        if not self._initialized:
            return True
        
        try:
            ref = self._db.reference(f'users/{user_id}')
            ref.update(data)
            return True
        except Exception as e:
            raise Exception(f"Failed to update user data: {str(e)}")
    
    # ===================== DOCUMENTS =====================
    
    def save_document(self, user_id: str, doc_data: Dict) -> str:
        """Save uploaded document and return document ID"""
        if not self._initialized:
            return 'demo_doc_1'
        
        try:
            ref = self._db.reference(f'documents/{user_id}')
            new_doc = ref.push(doc_data)
            return new_doc.key
        except Exception as e:
            raise Exception(f"Failed to save document: {str(e)}")
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID"""
        if not self._initialized:
            return self._get_demo_document()
        
        try:
            # Search across all users' documents
            ref = self._db.reference('documents')
            docs = ref.get()
            
            if docs:
                for user_docs in docs.values():
                    if doc_id in user_docs:
                        doc = user_docs[doc_id]
                        doc['id'] = doc_id
                        return doc
            return None
        except Exception as e:
            print(f"Error getting document: {e}")
            return None
    
    def get_instructor_documents(self, user_id: str) -> List[Dict]:
        """Get all documents uploaded by instructor"""
        if not self._initialized:
            return [self._get_demo_document()]
        
        try:
            ref = self._db.reference(f'documents/{user_id}')
            docs = ref.get()
            
            if docs:
                return [{'id': k, **v} for k, v in docs.items()]
            return []
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []
    
    # ===================== QUIZZES =====================
    
    def generate_unique_quiz_code(self, max_attempts: int = 10) -> str:
        """Generate 16-digit unique quiz code (alphanumeric)"""
        if not self._initialized:
            return secrets.token_hex(8).upper()  # 16 hex characters
        
        try:
            # Try up to max_attempts times to get a unique code
            for _ in range(max_attempts):
                # Generate 16-character alphanumeric code
                chars = string.ascii_uppercase + string.digits
                code = ''.join(secrets.choice(chars) for _ in range(16))
                
                # Check if code already exists
                ref = self._db.reference('quiz_codes')
                existing = ref.child(code).get().val()
                
                if existing is None:  # Code doesn't exist, it's unique
                    return code
            
            # Fallback: use UUID-based code if collision happens
            return secrets.token_hex(8).upper()
        except Exception as e:
            print(f"Error generating quiz code: {e}")
            return secrets.token_hex(8).upper()
    
    def register_quiz_code(self, quiz_code: str, quiz_id: str):
        """Register quiz code mapping to quiz ID"""
        if not self._initialized:
            return True
        
        try:
            ref = self._db.reference('quiz_codes')
            ref.child(quiz_code).set(quiz_id)
            return True
        except Exception as e:
            print(f"Error registering quiz code: {e}")
            return False

    def add_to_quiz_history(self, user_id: str, history_entry: Dict):
        """Add an entry to user's quiz history"""
        if not self._initialized:
            return True
        
        try:
            ref = self._db.reference(f'users/{user_id}/quiz_history')
            ref.push(history_entry)
            return True
        except Exception as e:
            print(f"Error adding to quiz history: {e}")
            return False

    def get_quiz_by_code(self, quiz_code: str) -> Optional[str]:
        """Get quiz ID from quiz code"""
        if not self._initialized:
            return 'demo_quiz_1'
        
        try:
            ref = self._db.reference('quiz_codes')
            quiz_id = ref.child(quiz_code).get().val()
            return quiz_id
        except Exception as e:
            print(f"Error getting quiz by code: {e}")
            return None
    
    def create_quiz(self, quiz_data: Dict, questions: List[Dict]) -> str:
        """Create new quiz with questions and generate unique quiz code"""
        if not self._initialized:
            return 'demo_quiz_1'
        
        try:
            # Generate unique quiz code
            quiz_code = self.generate_unique_quiz_code()
            quiz_data['quiz_code'] = quiz_code
            
            # Save quiz data
            ref = self._db.reference('quizzes')
            new_quiz = ref.push(quiz_data)
            quiz_id = new_quiz.key
            
            # Register code mapping
            self.register_quiz_code(quiz_code, quiz_id)
            
            # Save questions (don't store redundant 'id' field — Firebase push key is the ID)
            q_ref = self._db.reference(f'questions/{quiz_id}')
            for i, q in enumerate(questions):
                q_data = {k: v for k, v in q.items() if k != 'id'}
                q_ref.push(q_data)
            
            return quiz_id
        except Exception as e:
            raise Exception(f"Failed to create quiz: {str(e)}")
    
    def get_quiz(self, quiz_id: str) -> Optional[Dict]:
        """Get quiz by ID"""
        if not self._initialized:
            return self._get_demo_quiz()
        
        try:
            ref = self._db.reference(f'quizzes/{quiz_id}')
            quiz = ref.get()
            if quiz:
                quiz['id'] = quiz_id
            return quiz
        except Exception as e:
            print(f"Error getting quiz: {e}")
            return None
    
    def get_quiz_questions(self, quiz_id: str) -> List[Dict]:
        """Get all questions for a quiz"""
        if not self._initialized:
            return self._get_demo_questions()
        
        try:
            ref = self._db.reference(f'questions/{quiz_id}')
            questions = ref.get()
            
            if questions and isinstance(questions, dict):
                result = []
                for k, v in questions.items():
                    if isinstance(v, dict):
                        q = dict(v)  # copy to avoid mutating cached data
                        q['id'] = k  # Use Firebase push key as canonical ID
                        result.append(q)
                    else:
                        print(f"[WARNING] Skipping corrupted question {k} in quiz {quiz_id}")
                return result
            return []
        except Exception as e:
            print(f"Error getting questions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_question(self, quiz_id: str, question_id: str) -> Optional[Dict]:
        """Get specific question"""
        questions = self.get_quiz_questions(quiz_id)
        for q in questions:
            if q['id'] == question_id:
                return q
        return None
    
    def update_quiz_questions(self, quiz_id: str, questions: List[Dict]):
        """Update quiz questions"""
        if not self._initialized:
            return True
        
        try:
            ref = self._db.reference(f'questions/{quiz_id}')
            for q in questions:
                q_id = q.pop('id', None)
                if q_id:
                    ref.child(q_id).update(q)
            return True
        except Exception as e:
            raise Exception(f"Failed to update questions: {str(e)}")
    
    def get_available_quizzes(self, user_id: str) -> List[Dict]:
        """Get quizzes available to a user (public + own)"""
        if not self._initialized:
            return [self._get_demo_quiz()]
        
        try:
            ref = self._db.reference('quizzes')
            quizzes = ref.get()
            
            if quizzes:
                return [{'id': k, **v} for k, v in quizzes.items()
                       if not v.get('is_password_protected') or v.get('instructor_id') == user_id]
            return []
        except Exception as e:
            print(f"Error getting quizzes: {e}")
            return []
    
    def get_instructor_quizzes(self, user_id: str) -> List[Dict]:
        """Get quizzes created by instructor"""
        if not self._initialized:
            return [self._get_demo_quiz()]
        
        try:
            ref = self._db.reference('quizzes')
            quizzes = ref.order_by_child('instructor_id').equal_to(user_id).get()
            
            if quizzes:
                return [{'id': k, **v} for k, v in quizzes.items()]
            return []
        except Exception as e:
            print(f"Error getting instructor quizzes: {e}")
            return []
    
    def delete_quiz(self, quiz_id: str) -> bool:
        """Delete a quiz and all associated data"""
        if not self._initialized:
            return True
        
        try:
            # Delete quiz data
            quiz_ref = self._db.reference(f'quizzes/{quiz_id}')
            quiz_ref.delete()
            
            # Delete questions
            questions_ref = self._db.reference(f'questions/{quiz_id}')
            questions_ref.delete()
            
            # Delete quiz code mapping
            quiz = self.get_quiz(quiz_id)
            if quiz and 'quiz_code' in quiz:
                code_ref = self._db.reference(f'quiz_codes/{quiz["quiz_code"]}')
                code_ref.delete()
            
            # Note: We don't delete attempts/results as they contain student history
            # Those will remain for record-keeping but will show as orphaned
            
            print(f"[DELETE] Successfully deleted quiz {quiz_id}")
            return True
        except Exception as e:
            print(f"Error deleting quiz {quiz_id}: {e}")
            raise Exception(f"Failed to delete quiz: {str(e)}")
    
    # ===================== QUIZ ATTEMPTS =====================
    
    def create_quiz_attempt(self, user_id: str, quiz_id: str, questions: List[Dict]) -> str:
        """Create new quiz attempt"""
        if not self._initialized:
            return 'demo_attempt_1'
        
        try:
            ref = self._db.reference(f'attempts/{user_id}/{quiz_id}')
            attempt_data = {
                'started_at': self.get_timestamp(),
                'questions': [q['id'] for q in questions],
                'status': 'in_progress'
            }
            new_attempt = ref.push(attempt_data)
            return new_attempt.key
        except Exception as e:
            raise Exception(f"Failed to create attempt: {str(e)}")
    
    def save_quiz_result(self, user_id: str, quiz_id: str, attempt_id: str, result: Dict):
        """Save quiz attempt result"""
        if not self._initialized:
            return True
        
        try:
            import math
            import json
            
            # Helper function to ensure value is JSON serializable
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
            
            # Helper function to sanitize Firebase keys
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
                # Limit key length to prevent overly long concept names
                if len(safe_key) > 100:
                    safe_key = safe_key[:100]
                return safe_key
            
            # Create a sanitized copy of result for Firebase storage
            sanitized_result = {
                'score': sanitize_value(result.get('score', 0)),
                'score_percentage': sanitize_value(result.get('score_percentage', result.get('score', 0))),
                'correct': int(result.get('correct', 0)),
                'total_questions': int(result.get('total_questions', 0)),
                'marks_obtained': sanitize_value(result.get('marks_obtained', 0)),
                'total_marks': sanitize_value(result.get('total_marks', 100)),
                'timestamp': str(result.get('timestamp', self.get_timestamp()))
            }
            
            # Sanitize concept_performance
            if 'concept_performance' in result and result['concept_performance']:
                sanitized_concepts = {}
                for concept, performance in result['concept_performance'].items():
                    safe_key = sanitize_key(concept)
                    safe_value = sanitize_value(performance)
                    sanitized_concepts[safe_key] = safe_value
                sanitized_result['concept_performance'] = sanitized_concepts
            
            # Sanitize detailed results for Firebase storage (question-by-question review)
            sanitized_details = []
            for item in result.get('results', []):
                if isinstance(item, dict):
                    sanitized_details.append({
                        'question_id': str(item.get('question_id', '')),
                        'question_text': str(item.get('question_text', 'N/A'))[:500],
                        'question_type': str(item.get('question_type', 'mcq')),
                        'is_correct': bool(item.get('is_correct', False)),
                        'user_answer': str(item.get('user_answer', ''))[:300],
                        'correct_answer': str(item.get('correct_answer', ''))[:300],
                        'concept': sanitize_key(str(item.get('concept', 'General'))),
                        'explanation': str(item.get('explanation', ''))[:500],
                        'marks_obtained': int(item.get('marks_obtained', 0)),
                        'marks_total': int(item.get('marks_total', 1)),
                    })
            sanitized_result['details'] = sanitized_details
            
            if 'time_taken' in result:
                sanitized_result['time_taken'] = int(result.get('time_taken', 0))
            if 'time_taken_display' in result:
                sanitized_result['time_taken_display'] = str(result.get('time_taken_display', 'N/A'))
            
            # Use set() to create/overwrite the entire record
            ref = self._db.reference(f'attempts/{user_id}/{quiz_id}/{attempt_id}')
            ref.set({
                'completed_at': self.get_timestamp(),
                'status': 'completed',
                'result': sanitized_result
            })
            
            # Update concept mastery with sanitized keys
            if 'concept_performance' in result and result['concept_performance']:
                mastery_ref = self._db.reference(f'users/{user_id}/concept_mastery')
                current = mastery_ref.get() or {}
                
                for concept, performance in result['concept_performance'].items():
                    safe_key = sanitize_key(concept)
                    safe_performance = sanitize_value(performance)
                    
                    old_score = current.get(safe_key, 0.5)
                    old_score = sanitize_value(old_score)
                    
                    # Exponential moving average
                    new_score = 0.7 * float(old_score) + 0.3 * float(safe_performance)
                    new_score = sanitize_value(new_score)
                    current[safe_key] = new_score
                
                mastery_ref.set(current)
            
            return True
        except Exception as e:
            # Log the error with more detail
            print(f"Error saving quiz result: {str(e)}")
            print(f"Result data: {result}")
            raise Exception(f"Failed to save result: {str(e)}")
    
    def get_quiz_result(self, user_id: str, quiz_id: str, attempt_id: str) -> Optional[Dict]:
        """Get quiz attempt result with detailed breakdown"""
        if not self._initialized:
            return self._get_demo_result()
        
        try:
            ref = self._db.reference(f'attempts/{user_id}/{quiz_id}/{attempt_id}')
            attempt_data = ref.get()
            
            if attempt_data and 'result' in attempt_data:
                result = attempt_data['result']
                # Restore 'results' (detailed breakdown) from 'details' if available
                if 'details' in result and 'results' not in result:
                    result['results'] = result['details']
                return result
            return attempt_data
        except Exception as e:
            print(f"Error getting result: {e}")
            return None
    
    def get_quiz_history(self, user_id: str) -> List[Dict]:
        """Get user's quiz history"""
        if not self._initialized:
            return []
        
        try:
            ref = self._db.reference(f'users/{user_id}/quiz_history')
            history = ref.get()
            
            if history:
                return list(history.values())
            return []
        except Exception as e:
            print(f"Error getting history: {e}")
            return []
    
    def get_recent_quizzes(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get user's recent quiz attempts"""
        history = self.get_quiz_history(user_id)
        history.sort(key=lambda x: x.get('timestamp', x.get('completed_at', '')), reverse=True)
        return history[:limit]
    
    # ===================== ANALYTICS =====================
    
    def get_quiz_statistics(self, quiz_id: str) -> Dict:
        """Get statistics for a quiz"""
        if not self._initialized:
            return {
                'total_attempts': 5,
                'average_score': 75.5,
                'completion_rate': 0.85
            }
        
        try:
            # Aggregate stats from all attempts
            ref = self._db.reference('attempts')
            all_attempts = ref.get() or {}
            
            quiz_attempts = []
            for user_attempts in all_attempts.values():
                if quiz_id in user_attempts:
                    for attempt in user_attempts[quiz_id].values():
                        if attempt.get('status') == 'completed':
                            quiz_attempts.append(attempt)
            
            if not quiz_attempts:
                return {'total_attempts': 0, 'average_score': 0}
            
            scores = [a.get('result', {}).get('score', 0) for a in quiz_attempts]
            
            return {
                'total_attempts': len(quiz_attempts),
                'average_score': sum(scores) / len(scores) if scores else 0,
                'highest_score': max(scores) if scores else 0,
                'lowest_score': min(scores) if scores else 0
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def get_instructor_analytics(self, user_id: str) -> Dict:
        """Get aggregated analytics for instructor"""
        quizzes = self.get_instructor_quizzes(user_id)
        
        total_attempts = 0
        total_score = 0
        quiz_stats = []
        
        for quiz in quizzes:
            stats = self.get_quiz_statistics(quiz['id'])
            quiz_stats.append({**quiz, **stats})
            total_attempts += stats.get('total_attempts', 0)
            total_score += stats.get('average_score', 0) * stats.get('total_attempts', 0)
        
        return {
            'total_quizzes': len(quizzes),
            'total_attempts': total_attempts,
            'overall_average': total_score / total_attempts if total_attempts > 0 else 0,
            'quiz_stats': quiz_stats
        }
    
    def get_learning_curve(self, user_id: str) -> List[Dict]:
        """Get learning progress over time"""
        history = self.get_quiz_history(user_id)
        history.sort(key=lambda x: x.get('completed_at', ''))
        
        return [
            {'date': h.get('completed_at', ''), 'score': h.get('score', 0)}
            for h in history
        ]
    
    def get_user_knowledge_graph(self, user_id: str) -> Dict:
        """Get knowledge graph for user's studied concepts"""
        if not self._initialized:
            return self._get_demo_knowledge_graph()
        
        try:
            # Get all documents user has quizzed on
            user_data = self.get_user_data(user_id)
            concept_mastery = user_data.get('concept_mastery', {})
            
            # Get related knowledge graph from documents
            # This is simplified - full implementation would merge graphs
            nodes = [{'id': c, 'label': c, 'mastery': m} 
                    for c, m in concept_mastery.items()]
            
            return {'nodes': nodes, 'edges': []}
        except Exception as e:
            print(f"Error getting knowledge graph: {e}")
            return {'nodes': [], 'edges': []}
    
    # ===================== DEMO DATA =====================
    
    def _get_demo_user_data(self) -> Dict:
        """Get demo user data for testing without Firebase"""
        return {
            'name': 'Demo User',
            'email': 'demo@example.com',
            'role': 'student',
            'concept_mastery': {
                'Search Algorithms': 0.8,
                'BFS': 0.75,
                'DFS': 0.65,
                'A* Search': 0.45,
                'Heuristics': 0.5,
                'Knowledge Graphs': 0.3,
                'Machine Learning': 0.6
            },
            'quiz_history': [
                {'quiz_id': 'demo_1', 'score': 85, 'completed_at': '2026-02-01'},
                {'quiz_id': 'demo_2', 'score': 72, 'completed_at': '2026-02-05'}
            ]
        }
    
    def _get_demo_document(self) -> Dict:
        """Get demo document for testing"""
        return {
            'id': 'demo_doc_1',
            'filename': 'Introduction_to_AI.pdf',
            'concepts': ['Search Algorithms', 'BFS', 'DFS', 'A* Search', 
                        'Heuristics', 'Knowledge Graphs', 'Machine Learning'],
            'knowledge_graph': self._get_demo_knowledge_graph()
        }
    
    def _get_demo_quiz(self) -> Dict:
        """Get demo quiz for testing"""
        return {
            'id': 'demo_quiz_1',
            'name': 'Introduction to AI - Chapter 3',
            'num_questions': 10,
            'is_adaptive': True,
            'is_password_protected': False,
            'difficulty': 'mixed'
        }
    
    def _get_demo_questions(self) -> List[Dict]:
        """Get demo questions for testing"""
        return [
            {
                'id': 'q1',
                'text': 'Which search algorithm guarantees the shortest path in an unweighted graph?',
                'type': 'mcq',
                'options': ['BFS', 'DFS', 'A*', 'Hill Climbing'],
                'correct_answer': 'BFS',
                'concept': 'Search Algorithms',
                'difficulty': 'medium'
            },
            {
                'id': 'q2',
                'text': 'DFS uses a stack data structure for traversal.',
                'type': 'true_false',
                'correct_answer': 'True',
                'concept': 'DFS',
                'difficulty': 'easy'
            },
            {
                'id': 'q3',
                'text': 'What is a heuristic in the context of search algorithms?',
                'type': 'short_answer',
                'correct_answer': 'A function that estimates the cost to reach the goal',
                'concept': 'Heuristics',
                'difficulty': 'medium'
            }
        ]
    
    def _get_demo_result(self) -> Dict:
        """Get demo result for testing"""
        return {
            'score': 80,
            'total_questions': 5,
            'correct': 4,
            'time_taken': 300,
            'concept_performance': {
                'Search Algorithms': 1.0,
                'BFS': 0.5,
                'DFS': 1.0,
                'Heuristics': 0.75
            }
        }
    
    def _get_demo_knowledge_graph(self) -> Dict:
        """Get demo knowledge graph for testing"""
        return {
            'nodes': [
                {'id': 'AI', 'label': 'Artificial Intelligence'},
                {'id': 'Search', 'label': 'Search Algorithms'},
                {'id': 'BFS', 'label': 'Breadth-First Search'},
                {'id': 'DFS', 'label': 'Depth-First Search'},
                {'id': 'A*', 'label': 'A* Search'},
                {'id': 'Heuristics', 'label': 'Heuristics'},
                {'id': 'KG', 'label': 'Knowledge Graphs'}
            ],
            'edges': [
                {'from': 'AI', 'to': 'Search'},
                {'from': 'Search', 'to': 'BFS'},
                {'from': 'Search', 'to': 'DFS'},
                {'from': 'Search', 'to': 'A*'},
                {'from': 'A*', 'to': 'Heuristics'},
                {'from': 'AI', 'to': 'KG'}
            ]
        }
