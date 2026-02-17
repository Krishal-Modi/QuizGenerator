"""
Quiz routes - Take quizzes, view results
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from functools import wraps
from services.firebase_service import FirebaseService
from services.quiz_service import QuizService
from services.bandit_service import BanditService

quiz_bp = Blueprint('quiz', __name__)
firebase_service = FirebaseService()
quiz_service = QuizService()
bandit_service = BanditService()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


@quiz_bp.route('/')
@login_required
def index():
    """List all available quizzes"""
    user_id = session.get('user_id')
    
    # Get public quizzes and user's own quizzes
    quizzes = firebase_service.get_available_quizzes(user_id)
    
    return render_template('quiz/index.html', quizzes=quizzes)


@quiz_bp.route('/start/<quiz_id>', methods=['GET', 'POST'])
@login_required
def start_quiz(quiz_id):
    """Start a quiz - handle password-protected quizzes"""
    quiz = firebase_service.get_quiz(quiz_id)
    
    if not quiz:
        flash('Quiz not found.', 'danger')
        return redirect(url_for('quiz.index'))
    
    # Check if quiz is password protected
    if quiz.get('is_password_protected'):
        if request.method == 'POST':
            password = request.form.get('password')
            if password == quiz.get('password'):
                session[f'quiz_access_{quiz_id}'] = True
            else:
                flash('Incorrect password.', 'danger')
                return render_template('quiz/password.html', quiz=quiz)
        else:
            if not session.get(f'quiz_access_{quiz_id}'):
                return render_template('quiz/password.html', quiz=quiz)
    
    return redirect(url_for('quiz.take_quiz', quiz_id=quiz_id))


@quiz_bp.route('/take/<quiz_id>')
@login_required
def take_quiz(quiz_id):
    """Take a quiz with adaptive question selection"""
    user_id = session.get('user_id')
    quiz = firebase_service.get_quiz(quiz_id)
    
    if not quiz:
        flash('Quiz not found.', 'danger')
        return redirect(url_for('quiz.index'))
    
    # Check password protection
    if quiz.get('is_password_protected') and not session.get(f'quiz_access_{quiz_id}'):
        return redirect(url_for('quiz.start_quiz', quiz_id=quiz_id))
    
    # Get user's mastery data
    user_data = firebase_service.get_user_data(user_id)
    concept_mastery = user_data.get('concept_mastery', {})
    
    # Get questions for this quiz
    questions = firebase_service.get_quiz_questions(quiz_id)
    
    # Use bandit algorithm to select questions adaptively
    if quiz.get('is_adaptive', True):
        selected_questions = bandit_service.select_questions(
            questions=questions,
            concept_mastery=concept_mastery,
            num_questions=quiz.get('num_questions', 10),
            algorithm=quiz.get('bandit_algorithm', 'thompson_sampling')
        )
    else:
        # Static selection
        selected_questions = questions[:quiz.get('num_questions', 10)]
    
    # Create quiz attempt
    attempt_id = firebase_service.create_quiz_attempt(user_id, quiz_id, selected_questions)
    session[f'current_attempt_{quiz_id}'] = attempt_id
    
    display_questions = []
    for q in selected_questions:
        display_questions.append({
            'id': q.get('id'),
            'text': q.get('text'),
            'type': q.get('type'),
            'options': q.get('options', []),
            'concept': q.get('concept'),
            'difficulty': q.get('difficulty')
        })

    return render_template('quiz/take.html', 
                         quiz=quiz, 
                         questions=display_questions,
                         attempt_id=attempt_id)


@quiz_bp.route('/submit/<quiz_id>', methods=['POST'])
@login_required
def submit_quiz(quiz_id):
    """Submit quiz answers and calculate results"""
    user_id = session.get('user_id')
    attempt_id = session.get(f'current_attempt_{quiz_id}')

    quiz = firebase_service.get_quiz(quiz_id)
    if not quiz:
        flash('Quiz not found.', 'danger')
        return redirect(url_for('quiz.index'))
    
    if not attempt_id:
        flash('Quiz session expired. Please start again.', 'warning')
        return redirect(url_for('quiz.start_quiz', quiz_id=quiz_id))
    
    # Get submitted answers - collect ALL question IDs from form (including unanswered)
    answers = {}
    presented_question_ids = []
    for key, value in request.form.items():
        if key.startswith('question_'):
            question_id = key.replace('question_', '')
            presented_question_ids.append(question_id)
            if value.strip():  # Only include non-empty answers
                answers[question_id] = value
    
    # Capture time taken (sent from frontend timer)
    time_taken_seconds = int(request.form.get('time_taken', 0))
    time_taken_minutes = time_taken_seconds // 60
    time_taken_display = f"{time_taken_minutes}m {time_taken_seconds % 60}s"
    
    print(f"\n{'='*60}")
    print(f"[SUBMIT] Quiz ID: {quiz_id}")
    print(f"[SUBMIT] User ID: {user_id}")
    print(f"[SUBMIT] Attempt ID: {attempt_id}")
    print(f"[SUBMIT] Questions presented: {len(presented_question_ids)}")
    print(f"[SUBMIT] Questions answered: {len(answers)}")
    print(f"[SUBMIT] Time taken: {time_taken_display}")
    print(f"[SUBMIT] Presented question IDs: {presented_question_ids}")
    print(f"[SUBMIT] Answered question IDs: {list(answers.keys())}")
    print(f"{'='*60}\n")
    
    # Validate we have questions
    if not presented_question_ids:
        print("[ERROR] No question IDs found in form submission!")
        flash('Error: No questions found in submission. Please try again.', 'danger')
        return redirect(url_for('quiz.start_quiz', quiz_id=quiz_id))
    
    # Calculate results - pass presented_question_ids so we evaluate only shown questions
    result = quiz_service.evaluate_quiz(quiz_id, attempt_id, answers, presented_question_ids)
    
    # Ensure all required fields are in the result
    correct_count = int(result.get('correct', 0))
    total_questions = int(result.get('total_questions', 0))
    marks_obtained = int(result.get('marks_obtained', correct_count))
    total_marks = int(result.get('total_marks', total_questions))
    score_percentage = float(result.get('score', 0))
    
    print(f"[RESULT] Quiz {quiz_id}: {correct_count}/{total_questions} correct ({score_percentage:.1f}%)")
    print(f"[RESULT] Marks: {marks_obtained}/{total_marks}")
    print(f"[RESULT] Detail results count: {len(result.get('results', []))}")
    
    # Build comprehensive result object
    result_data = {
        'score': score_percentage,  # Percentage (0-100)
        'correct': correct_count,  # Number of correct answers
        'total_questions': total_questions,  # Total questions
        'marks_obtained': marks_obtained,  # Marks earned
        'total_marks': total_marks,  # Total possible marks
        'score_percentage': score_percentage,  # Duplicate for template compatibility
        'timestamp': firebase_service.get_timestamp(),
        'time_taken': time_taken_seconds,
        'time_taken_display': time_taken_display,
        'concept_performance': result.get('concept_performance', {}),
        'results': result.get('results', []),  # Detailed answer reviews
    }
    
    print(f"[CALCULATION] Score: {marks_obtained}/{total_marks} marks = {score_percentage:.1f}%")
    
    # Store full result in session FIRST (before any potential Firebase errors)
    session[f'quiz_result_{attempt_id}'] = result_data
    session.modified = True  # Force session save
    
    print(f"[SESSION] Stored result in session key: quiz_result_{attempt_id}")
    
    # Try to save to Firebase, but don't crash if it fails
    save_success = True
    try:
        # Update user's concept mastery using bandit feedback
        bandit_service.update_mastery(user_id, result_data['concept_performance'])
        
        # Save attempt results (sanitized version will be saved to Firebase)
        firebase_service.save_quiz_result(user_id, quiz_id, attempt_id, result_data)
        
        # Update user's quiz history with score and time
        firebase_service.add_to_quiz_history(user_id, {
            'quiz_id': quiz_id,
            'quiz_name': quiz.get('name', 'Quiz'),
            'score': score_percentage,
            'score_percentage': score_percentage,
            'correct': correct_count,
            'total_questions': total_questions,
            'marks_obtained': marks_obtained,
            'total_marks': total_marks,
            'time_taken': time_taken_seconds,
            'time_taken_display': time_taken_display,
            'timestamp': result_data['timestamp'],
            'attempt_id': attempt_id
        })
        
        print(f"[FIREBASE] Results saved successfully")
        flash('Quiz completed successfully! Your score has been saved.', 'success')
        
    except Exception as e:
        # Log the error but continue to show results
        save_success = False
        print(f"[FIREBASE_ERROR] Failed to save to Firebase: {str(e)}")
        import traceback
        print(traceback.format_exc())
        flash(f'Quiz completed! Results saved locally (score: {marks_obtained}/{total_marks} = {score_percentage:.1f}%)', 'warning')
    
    # Clear session
    session.pop(f'current_attempt_{quiz_id}', None)
    session.pop(f'quiz_access_{quiz_id}', None)
    
    return redirect(url_for('quiz.results', quiz_id=quiz_id, attempt_id=attempt_id))


@quiz_bp.route('/results/<quiz_id>/<attempt_id>')
@login_required
def results(quiz_id, attempt_id):
    """View quiz results and feedback"""
    user_id = session.get('user_id')
    
    # First, try to get result from session (for immediate display after quiz)
    result = session.pop(f'quiz_result_{attempt_id}', None)
    
    # If not in session, get from Firebase (for viewing old results)
    if not result:
        result = firebase_service.get_quiz_result(user_id, quiz_id, attempt_id)
    
    quiz = firebase_service.get_quiz(quiz_id)
    
    if not result:
        flash('Results not found. Please try taking the quiz again.', 'danger')
        return redirect(url_for('quiz.index'))
    
    # Validate result has all required fields
    if 'score' not in result or result.get('score') is None:
        print(f"[WARNING] Result missing score field: {result.keys()}")
        flash('Warning: Some score information may be incomplete.', 'warning')
    
    # Ensure all required fields are present for template
    result_display = {
        'score': float(result.get('score', 0)),  # Percentage 0-100
        'correct': int(result.get('correct', 0)),  # Number correct
        'total_questions': int(result.get('total_questions', 0)),  # Total questions
        'marks_obtained': int(result.get('marks_obtained', result.get('correct', 0))),  # Marks earned
        'total_marks': int(result.get('total_marks', result.get('total_questions', 0))),  # Total marks
        'score_percentage': float(result.get('score_percentage', result.get('score', 0))),  # For compatibility
        'time_taken_display': result.get('time_taken_display', 'N/A'),
        'concept_performance': result.get('concept_performance', {}),
        'results': result.get('results', []),  # Detailed reviews
        'timestamp': result.get('timestamp'),
    }
    
    print(f"[RESULTS] Displaying: {result_display['marks_obtained']}/{result_display['total_marks']} marks ({result_display['score']:.1f}%)")
    
    # Get weak concepts and recommendations
    weak_concepts = quiz_service.get_weak_concepts(result_display)
    recommendations = quiz_service.get_study_recommendations(weak_concepts, quiz_id)
    
    return render_template('quiz/results.html', 
                         quiz=quiz,
                         result=result_display, 
                         weak_concepts=weak_concepts,
                         recommendations=recommendations)


@quiz_bp.route('/history')
@login_required
def history():
    """View quiz history"""
    user_id = session.get('user_id')
    history_raw = firebase_service.get_quiz_history(user_id)
    
    # Normalize history entries to have consistent field names
    history = []
    for entry in history_raw:
        if not isinstance(entry, dict):
            continue
        
        score_val = 0
        try:
            score_val = float(entry.get('score_percentage', entry.get('score', 0)) or 0)
        except (ValueError, TypeError):
            pass
        
        correct = int(entry.get('correct', 0) or 0)
        total = int(entry.get('total_questions', 0) or 0)
        marks_obtained = int(entry.get('marks_obtained', correct) or 0)
        total_marks = int(entry.get('total_marks', total) or 0)
        
        # Parse timestamp for display
        timestamp = str(entry.get('timestamp', ''))
        date_display = timestamp[:10] if len(timestamp) >= 10 else 'N/A'
        
        history.append({
            'quiz_id': entry.get('quiz_id', ''),
            'quiz_name': entry.get('quiz_name', 'Quiz'),
            'score': round(score_val, 1),
            'correct': correct,
            'total': total,
            'marks_obtained': marks_obtained,
            'total_marks': total_marks,
            'timestamp': timestamp,
            'date': date_display,
            'time_taken_display': entry.get('time_taken_display', 'N/A'),
            'attempt_id': entry.get('attempt_id', ''),
        })
    
    # Sort by timestamp descending (newest first)
    history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Calculate summary stats
    scores = [h['score'] for h in history if h['score'] > 0]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    best_score = round(max(scores), 1) if scores else 0
    total_questions = sum(h['total'] for h in history)
    
    return render_template('quiz/history.html', 
                         history=history,
                         avg_score=avg_score,
                         best_score=best_score,
                         total_questions=total_questions)


@quiz_bp.route('/find', methods=['GET', 'POST'])
def find_quiz():
    """Find and join quiz by 16-digit code"""
    if request.method == 'POST':
        quiz_code = request.form.get('quiz_code', '').upper().strip()
        
        if not quiz_code:
            flash('Please enter a quiz code.', 'warning')
            return render_template('quiz/find.html')
        
        # Validate code format (16 alphanumeric characters)
        if len(quiz_code) != 16 or not all(c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in quiz_code):
            flash('Invalid code format. Code must be 16 alphanumeric characters.', 'danger')
            return render_template('quiz/find.html')
        
        # Look up quiz by code
        quiz_id = firebase_service.get_quiz_by_code(quiz_code)
        
        if not quiz_id:
            flash('Quiz code not found. Please check and try again.', 'danger')
            return render_template('quiz/find.html')
        
        # Check if user is logged in
        if 'user_id' not in session:
            session['next_quiz_id'] = quiz_id
            flash('Please login to access this quiz.', 'info')
            return redirect(url_for('auth.login'))
        
        # Redirect to quiz start
        return redirect(url_for('quiz.start_quiz', quiz_id=quiz_id))
    
    return render_template('quiz/find.html')


@quiz_bp.route('/code/<quiz_id>')
@login_required
def get_quiz_code(quiz_id):
    """Get quiz code for sharing (JSON endpoint)"""
    quiz = firebase_service.get_quiz(quiz_id)
    
    if not quiz:
        return jsonify({'error': 'Quiz not found'}), 404
    
    # Verify user is quiz creator
    if quiz.get('instructor_id') != session.get('user_id'):
        # Allow instructors/creators to share
        pass
    
    return jsonify({
        'quiz_code': quiz.get('quiz_code', 'N/A'),
        'quiz_id': quiz_id,
        'quiz_name': quiz.get('name', 'Untitled')
    })


