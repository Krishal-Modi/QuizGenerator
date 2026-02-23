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
    
    # Prepare mastery chart data with ML-enhanced analytics
    # Calculate detailed concept performance metrics
    try:
        concept_performance = calculate_concept_performance(quiz_history, concept_mastery)
    except Exception as e:
        print(f"Error calculating concept performance: {e}")
        concept_performance = {}
    
    mastery_labels = []
    mastery_data = []
    mastery_trend = []  # Learning trend indicator
    mastery_predicted = []  # Predicted next performance
    
    if concept_mastery:
        mastery_labels = [str(c)[:40] for c in list(concept_mastery.keys())[:10]]
        
        for c in list(concept_mastery.keys())[:10]:
            val = concept_mastery.get(c, 0)
            try:
                val = float(val) * 100
                val = min(max(val, 0), 100)  # Clamp 0-100
            except (ValueError, TypeError):
                val = 0
            mastery_data.append(round(val, 1))
            
            # Calculate learning trend (positive/negative growth)
            perf = concept_performance.get(c, {})
            mastery_trend.append(perf.get('trend', 0))
            mastery_predicted.append(perf.get('predicted', val))
    
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
    
    # Debug output
    print(f"[Progress Page Debug]")
    print(f"  Total quizzes: {len(quiz_history)}")
    print(f"  Concepts tracked: {len(concept_mastery)}")
    print(f"  Mastery labels: {len(mastery_labels)}")
    print(f"  Mastery data points: {len(mastery_data)}")
    print(f"  Has data for graph: {len(mastery_labels) > 0 and len(mastery_data) > 0}")
    if mastery_labels:
        print(f"  First concept: {mastery_labels[0]} = {mastery_data[0]}%")
    
    return render_template('dashboard/progress.html',
                         stats=stats,
                         concept_mastery=concept_mastery,
                         quiz_history=quiz_history,
                         mastery_labels=mastery_labels,
                         mastery_data=mastery_data,
                         mastery_trend=mastery_trend,
                         mastery_predicted=mastery_predicted,
                         performance_labels=performance_labels,
                         performance_data=performance_data,
                         top_concepts=top_concepts,
                         weak_concepts=weak_concepts,
                         concept_performance=concept_performance)


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


def calculate_concept_performance(quiz_history, concept_mastery):
    """Calculate ML-enhanced performance metrics for each concept"""
    try:
        import numpy as np
    except ImportError:
        print("NumPy not available, using basic calculations")
        # Fallback to basic calculations without NumPy
        performance = {}
        for concept, current_mastery in concept_mastery.items():
            current_percent = float(current_mastery) * 100
            performance[concept] = {
                'current': round(current_percent, 1),
                'trend': 0,
                'predicted': round(current_percent, 1),
                'learning_rate': 0,
                'consistency': 50,
                'attempts': 0
            }
        return performance
    
    from collections import defaultdict
    
    performance = {}
    concept_history = defaultdict(list)
    
    # Collect historical performance for each concept
    for quiz in quiz_history:
        if not isinstance(quiz, dict):
            continue
        
        quiz_concepts = quiz.get('concepts', [])
        score = quiz.get('score_percentage', quiz.get('score', 0)) or 0
        
        # If quiz doesn't track individual concepts, use overall concepts
        if not quiz_concepts:
            quiz_concepts = list(concept_mastery.keys())
        
        for concept in quiz_concepts:
            concept_history[concept].append(float(score))
    
    # Calculate metrics for each concept
    for concept, current_mastery in concept_mastery.items():
        scores = concept_history.get(concept, [])
        current_percent = float(current_mastery) * 100
        
        if len(scores) >= 2:
            # Calculate learning trend (slope of performance over time)
            x = np.arange(len(scores))
            y = np.array(scores)
            
            # Simple linear regression for trend
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                trend = round(slope, 2)
                
                # Predict next performance using trend
                predicted = current_percent + (trend * 2)
                predicted = min(max(predicted, 0), 100)
            else:
                trend = 0
                predicted = current_percent
            
            # Calculate learning rate (recent improvement)
            if len(scores) >= 3:
                recent_avg = np.mean(scores[-3:])
                early_avg = np.mean(scores[:3]) if len(scores) > 3 else scores[0]
                learning_rate = recent_avg - early_avg
            else:
                learning_rate = 0
            
            # Calculate consistency (standard deviation - lower is more consistent)
            consistency = 100 - min(np.std(scores), 100)
            
        else:
            trend = 0
            predicted = current_percent
            learning_rate = 0
            consistency = 50
        
        performance[concept] = {
            'current': round(current_percent, 1),
            'trend': trend,
            'predicted': round(predicted, 1),
            'learning_rate': round(learning_rate, 1),
            'consistency': round(consistency, 1),
            'attempts': len(scores)
        }
    
    return performance
