import os
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Firebase
admin_sdk_path = 'config/firebase-admin-sdk.json'
if not firebase_admin._apps:
    cred = credentials.Certificate(admin_sdk_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
    })

# Delete all main collections
collections = ['users', 'documents', 'quiz_codes', 'quizzes', 'questions']

print('⚠️  WARNING: This will delete ALL data from Firebase!')
confirm = input('Type YES to confirm deletion: ')

if confirm == 'YES':
    for collection in collections:
        try:
            ref = db.reference(collection)
            ref.delete()
            print(f'✓ Deleted: {collection}')
        except Exception as e:
            print(f'✗ Error deleting {collection}: {e}')
    print('\n✓ Database cleanup complete!')
else:
    print('Deletion cancelled.')
