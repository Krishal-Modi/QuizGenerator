"""
Authentication routes - Login, Signup, Logout
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from services.firebase_service import FirebaseService

auth_bp = Blueprint('auth', __name__)
firebase_service = FirebaseService()


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        try:
            user = firebase_service.sign_in(email, password)
            if user:
                session['user_id'] = user['localId']
                session['email'] = user['email']
                session['id_token'] = user['idToken']
                
                # Check user role
                user_data = firebase_service.get_user_data(user['localId'])
                session['role'] = user_data.get('role', 'student')
                session['name'] = user_data.get('name', email.split('@')[0])
                
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard.index'))
            else:
                flash('Invalid email or password. Please try again.', 'danger')
        except Exception as e:
            error_str = str(e).lower()
            if 'auth/user-not-found' in error_str or 'unable to locate' in error_str:
                flash('No account found with this email. Please sign up first.', 'danger')
            elif 'auth/wrong-password' in error_str or 'invalid password' in error_str:
                flash('Incorrect password. Please try again.', 'danger')
            elif 'auth/invalid-email' in error_str or 'invalid email' in error_str:
                flash('Invalid email address. Please check and try again.', 'danger')
            elif 'too many login attempts' in error_str:
                flash('Too many failed login attempts. Please try again later.', 'danger')
            else:
                flash('Login failed. Please check your credentials and try again.', 'danger')
    
    return render_template('auth/login.html')


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration page"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role', 'student')  # student or instructor
        
        # Validation
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('auth/signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('auth/signup.html')
        
        try:
            user = firebase_service.create_user(email, password)
            if user:
                # Save additional user data
                user_data = {
                    'name': name,
                    'email': email,
                    'role': role,
                    'created_at': firebase_service.get_timestamp(),
                    'concept_mastery': {},
                    'quiz_history': []
                }
                firebase_service.save_user_data(user['localId'], user_data)
                
                flash('Account created successfully! Please login.', 'success')
                return redirect(url_for('auth.login'))
        except Exception as e:
            error_str = str(e).lower()
            
            # Provide user-friendly error messages
            if 'email_exists' in error_str:
                flash('This email is already registered. Please login or use a different email.', 'danger')
            elif 'weak_password' in error_str:
                flash('Password is too weak. Use at least 6 characters with letters and numbers.', 'danger')
            elif 'invalid_email' in error_str:
                flash('Please enter a valid email address.', 'danger')
            else:
                flash(f'Signup failed: {str(e)}', 'danger')
    
    return render_template('auth/signup.html')


@auth_bp.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Password reset request"""
    if request.method == 'POST':
        email = request.form.get('email')
        try:
            firebase_service.send_password_reset(email)
            flash('Password reset email sent. Check your inbox.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            flash(f'Failed to send reset email: {str(e)}', 'danger')
    
    return render_template('auth/forgot_password.html')
