"""
Setup verification script - Checks if all dependencies and configurations are correct
"""
import os
import sys

def check_environment():
    """Check if .env file exists and has required variables"""
    print("Checking environment configuration...")
    
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        return False
    
    required_vars = [
        'FIREBASE_API_KEY',
        'FIREBASE_AUTH_DOMAIN',
        'FIREBASE_PROJECT_ID',
        'FIREBASE_DATABASE_URL',
        'SECRET_KEY'
    ]
    
    from dotenv import load_dotenv
    load_dotenv()
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    
    print("✓ Environment variables configured")
    return True


def check_firebase_config():
    """Check if Firebase admin SDK file exists"""
    print("\nChecking Firebase configuration...")
    
    admin_sdk_path = os.getenv('FIREBASE_ADMIN_SDK_PATH', 'config/firebase-admin-sdk.json')
    if not os.path.exists(admin_sdk_path):
        print(f"❌ Firebase admin SDK file not found at: {admin_sdk_path}")
        return False
    
    print("✓ Firebase admin SDK file exists")
    return True


def check_python_packages():
    """Check if required Python packages are installed"""
    print("\nChecking Python packages...")
    
    required_packages = [
        'flask',
        'firebase_admin',
        'pyrebase',
        'spacy',
        'transformers',
        'torch',
        'nltk',
        'pdfplumber',
        'networkx'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✓ All required packages installed")
    return True


def check_spacy_model():
    """Check if spacy language model is installed"""
    print("\nChecking spacy language model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ Spacy model 'en_core_web_sm' installed")
        return True
    except OSError:
        print("❌ Spacy model 'en_core_web_sm' not found")
        print("   Run: python -m spacy download en_core_web_sm")
        return False


def check_directories():
    """Check if required directories exist"""
    print("\nChecking required directories...")
    
    required_dirs = [
        'uploads',
        'static',
        'templates',
        'config',
        'services',
        'routes'
    ]
    
    missing = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing.append(directory)
    
    if missing:
        print(f"❌ Missing directories: {', '.join(missing)}")
        return False
    
    print("✓ All required directories exist")
    return True


def check_firebase_connection():
    """Test Firebase connection"""
    print("\nTesting Firebase connection...")
    
    try:
        from services.firebase_service import FirebaseService
        firebase = FirebaseService()
        
        if firebase._initialized:
            print("✓ Firebase connection successful")
            return True
        else:
            print("⚠️  Firebase not initialized (running in demo mode)")
            return True
    except Exception as e:
        print(f"❌ Firebase connection failed: {e}")
        return False


def check_app_import():
    """Test if app can be imported"""
    print("\nTesting application import...")
    
    try:
        from app import app
        print("✓ Application imports successfully")
        return True
    except Exception as e:
        print(f"❌ Application import failed: {e}")
        return False


def main():
    """Run all checks"""
    print("="*60)
    print("Quiz Generator - Setup Verification")
    print("="*60)
    
    checks = [
        check_environment,
        check_firebase_config,
        check_python_packages,
        check_spacy_model,
        check_directories,
        check_firebase_connection,
        check_app_import
    ]
    
    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All checks passed ({passed}/{total})")
        print("\n✅ Your application is ready to run!")
        print("   Start the server with: python app.py")
        print("   Then visit: http://127.0.0.1:5000")
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print("\n⚠️  Please fix the issues above before running the application.")
    
    print("="*60)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
