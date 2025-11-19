"""
Example web API code using Flask.
"""

from flask import Flask, jsonify, request
from functools import wraps
import jwt
from datetime import datetime, timedelta


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'


# In-memory database
users_db = {}
posts_db = []


def token_required(f):
    """Decorator to require authentication token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['user_id']
        except:
            return jsonify({'message': 'Token is invalid'}), 401

        return f(current_user, *args, **kwargs)

    return decorated


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.get_json()

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not all([username, email, password]):
        return jsonify({'message': 'Missing required fields'}), 400

    if username in users_db:
        return jsonify({'message': 'User already exists'}), 409

    users_db[username] = {
        'email': email,
        'password': password,  # In production, hash this!
        'created_at': datetime.utcnow().isoformat()
    }

    return jsonify({'message': 'User created successfully'}), 201


@app.route('/api/login', methods=['POST'])
def login():
    """Login and get authentication token."""
    data = request.get_json()

    username = data.get('username')
    password = data.get('password')

    user = users_db.get(username)

    if not user or user['password'] != password:
        return jsonify({'message': 'Invalid credentials'}), 401

    token = jwt.encode({
        'user_id': username,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }, app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({'token': token})


@app.route('/api/posts', methods=['GET'])
def get_posts():
    """Get all posts."""
    return jsonify({'posts': posts_db})


@app.route('/api/posts', methods=['POST'])
@token_required
def create_post(current_user):
    """Create a new post."""
    data = request.get_json()

    title = data.get('title')
    content = data.get('content')

    if not all([title, content]):
        return jsonify({'message': 'Missing required fields'}), 400

    post = {
        'id': len(posts_db) + 1,
        'title': title,
        'content': content,
        'author': current_user,
        'created_at': datetime.utcnow().isoformat()
    }

    posts_db.append(post)

    return jsonify(post), 201


@app.route('/api/posts/<int:post_id>', methods=['GET'])
def get_post(post_id):
    """Get a specific post."""
    post = next((p for p in posts_db if p['id'] == post_id), None)

    if not post:
        return jsonify({'message': 'Post not found'}), 404

    return jsonify(post)


@app.route('/api/posts/<int:post_id>', methods=['DELETE'])
@token_required
def delete_post(current_user, post_id):
    """Delete a post."""
    post = next((p for p in posts_db if p['id'] == post_id), None)

    if not post:
        return jsonify({'message': 'Post not found'}), 404

    if post['author'] != current_user:
        return jsonify({'message': 'Unauthorized'}), 403

    posts_db.remove(post)

    return jsonify({'message': 'Post deleted successfully'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
