"""
Dashboard routes - User dashboard, progress tracking
"""
from flask import Blueprint, render_template, session, redirect, url_for, flash
from functools import wraps
from services.firebase_service import FirebaseService

dashboard_bp = Blueprint('dashboard', __name__)
firebase_service = FirebaseService()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


@dashboard_bp.route('/')
@login_required
def index():
    """Main dashboard"""
    user_id = session.get('user_id')
    user_data = firebase_service.get_user_data(user_id)
    
    # Get summary statistics
    quiz_history_raw = user_data.get('quiz_history', {})
    if isinstance(quiz_history_raw, dict):
        quiz_history_list = [v for v in quiz_history_raw.values() if isinstance(v, dict)]
    elif isinstance(quiz_history_raw, list):
        quiz_history_list = [q for q in quiz_history_raw if isinstance(q, dict)]
    else:
        quiz_history_list = []
    
    stats = {
        'total_quizzes': len(quiz_history_list),
        'average_score': calculate_average_score(quiz_history_list),
        'concepts_mastered': count_mastered_concepts(user_data.get('concept_mastery', {})),
        'total_concepts': len(user_data.get('concept_mastery', {}))
    }
    
    # Get recent activity
    recent_quizzes = firebase_service.get_recent_quizzes(user_id, limit=5)
    
    # Get weak concepts for review
    weak_concepts = get_weak_concepts(user_data.get('concept_mastery', {}))
    
    return render_template('dashboard/index.html', 
                         user=user_data,
                         stats=stats,
                         recent_quizzes=recent_quizzes,
                         weak_concepts=weak_concepts)


@dashboard_bp.route('/progress')
@login_required
def progress():
    """Detailed progress view"""
    user_id = session.get('user_id')
    user_data = firebase_service.get_user_data(user_id)
    
    # Get concept mastery details
    concept_mastery = user_data.get('concept_mastery', {})
    quiz_history_raw = user_data.get('quiz_history', {})
    
    # Firebase stores push() items as dicts {push_id: {data}}
    # Convert to list properly
    if isinstance(quiz_history_raw, dict):
        quiz_history = [v for v in quiz_history_raw.values() if isinstance(v, dict)]
    elif isinstance(quiz_history_raw, list):
        quiz_history = [q for q in quiz_history_raw if isinstance(q, dict)]
    else:
        quiz_history = []
    
    # Sort quiz history by timestamp
    if quiz_history:
        quiz_history_sorted = sorted(
            quiz_history, 
            key=lambda x: x.get('timestamp', '') if isinstance(x, dict) else '', 
            reverse=False
        )
    else:
        quiz_history_sorted = []
    
    # Get statistics including total time
    total_time_seconds = sum(
        q.get('time_taken', 0) for q in quiz_history if isinstance(q, dict)
    )
    total_time_hours = total_time_seconds // 3600
    total_time_minutes = (total_time_seconds % 3600) // 60
    if total_time_hours > 0:
        total_time_display = f"{total_time_hours}h {total_time_minutes}m"
    elif total_time_minutes > 0:
        total_time_display = f"{total_time_minutes}m"
    else:
        total_time_display = "0m"
    
    # Calculate best score
    all_scores = []
    for q in quiz_history:
        if isinstance(q, dict):
            try:
                s = float(q.get('score_percentage', q.get('score', 0)) or 0)
                all_scores.append(s)
            except (ValueError, TypeError):
                pass
    
    best_score = max(all_scores) if all_scores else 0
    
    stats = {
        'total_quizzes': len(quiz_history),
        'avg_score': calculate_average_score(quiz_history),
        'concepts_mastered': count_mastered_concepts(concept_mastery),
        'total_time': total_time_display,
        'best_score': round(best_score, 1),
        'total_correct': sum(q.get('correct', 0) for q in quiz_history if isinstance(q, dict)),
        'total_questions_attempted': sum(q.get('total_questions', 0) for q in quiz_history if isinstance(q, dict))
    }
    
    # Prepare mastery chart data - safely handle non-numeric values
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
    
    # Prepare performance over time chart data with quiz names and dates
    performance_labels = []
    performance_data = []
    
    for i, q in enumerate(quiz_history_sorted):
        # Format: "Quiz Name (MM/DD)" or just "Quiz #" if no name
        quiz_name = str(q.get('quiz_name', f'Quiz {i+1}'))
        timestamp = str(q.get('timestamp', ''))
        
        # Shorten quiz name if too long
        if len(quiz_name) > 20:
            quiz_name = quiz_name[:17] + '...'
        
        # Add date if available
        if timestamp and len(timestamp) >= 10:
            date_str = timestamp[:10]  # Get YYYY-MM-DD
            try:
                date_parts = date_str.split('-')
                if len(date_parts) == 3:
                    label = f"{quiz_name} ({date_parts[1]}/{date_parts[2]})"
                else:
                    label = quiz_name
            except:
                label = quiz_name
        else:
            label = quiz_name
        
        performance_labels.append(label)
        try:
            score = float(q.get('score_percentage', q.get('score', 0)) or 0)
        except (ValueError, TypeError):
            score = 0
        performance_data.append(round(score, 1))
    
    # Get top concepts (highest mastery)
    top_concepts = sorted(
        [{'name': c, 'mastery': int(m * 100)} for c, m in concept_mastery.items()],
        key=lambda x: x['mastery'],
        reverse=True
    )[:5]
    
    # Get weak concepts
    weak_concepts = get_weak_concepts(concept_mastery)
    
    return render_template('dashboard/progress.html',
                         stats=stats,
                         concept_mastery=concept_mastery,
                         quiz_history=quiz_history,
                         mastery_labels=mastery_labels,
                         mastery_data=mastery_data,
                         performance_labels=performance_labels,
                         performance_data=performance_data,
                         top_concepts=top_concepts,
                         weak_concepts=weak_concepts)


@dashboard_bp.route('/knowledge-graph')
@login_required
def knowledge_graph():
    """View concept knowledge graph with mastery overlay"""
    user_id = session.get('user_id')
    user_data = firebase_service.get_user_data(user_id)
    
    concept_mastery = user_data.get('concept_mastery', {})
    
    # Get knowledge graph data (will be populated when user uploads notes)
    graph_data = firebase_service.get_user_knowledge_graph(user_id)
    
    return render_template('dashboard/knowledge_graph.html',
                         graph_data=graph_data,
                         concept_mastery=concept_mastery)


@dashboard_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile settings"""
    user_id = session.get('user_id')
    user_data = firebase_service.get_user_data(user_id)
    
    return render_template('dashboard/profile.html', user=user_data)


# Helper functions
def calculate_average_score(quiz_history):
    """Calculate average score from quiz history"""
    if not quiz_history:
        return 0
    
    # Filter to only dictionary items and extract scores
    scores = [q.get('score', 0) for q in quiz_history if isinstance(q, dict) and 'score' in q]
    return sum(scores) / len(scores) if scores else 0


def count_mastered_concepts(concept_mastery):
    """Count concepts with mastery >= 0.7"""
    return sum(1 for score in concept_mastery.values() if score >= 0.7)


def get_weak_concepts(concept_mastery, threshold=0.5, limit=5):
    """Get concepts with mastery below threshold"""
    weak = [(concept, score) for concept, score in concept_mastery.items() 
            if score < threshold]
    weak.sort(key=lambda x: x[1])
    return weak[:limit]
