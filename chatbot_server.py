"""
Autonomous Chatbot Backend Server with User Authentication and Persistent Conversations
A Flask server providing intelligent conversational AI with pattern matching,
context awareness, and memory capabilities, now with user accounts and
persistent conversation storage using SQLite.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import json
import random
import re
import datetime
import logging
import time
import os
import uuid # For generating unique IDs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)   # Enable CORS for frontend connection

# Configure SQLAlchemy for SQLite
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# DB_PATH = os.path.join(BASE_DIR, 'chatbot.db') # Database file will be created here
# app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Suppresses a warning
# db = SQLAlchemy(app)


# Configure SQLAlchemy for PostgreSQL
# Render will automatically set a DATABASE_URL environment variable for its managed PostgreSQL.
# We use a fallback to a local SQLite for development if DATABASE_URL isn't set (e.g., when running locally).
if os.environ.get('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace("://", "ql://", 1) # Render's URL might need 'postgresql' changed to 'postgresql+psycopg2' or similar for SQLAlchemy
    # The .replace() is a common workaround for SQLAlchemy's interpretation of Render's default postgresql:// schema
else:
    # Local SQLite for development (optional, but good for local testing)
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DB_PATH = os.path.join(BASE_DIR, 'chatbot.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Models ---

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)

    # Relationship to conversations: one user can have many conversations
    conversations = db.relationship('UserConversation', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        """Hashes the given password and stores it."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Checks if the provided password matches the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"

class UserConversation(db.Model):
    # This table represents a "chat session" or "conversation" for a user
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(255), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False, default="New Chat") # Default title
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)
    last_active = db.Column(db.DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)

    # Relationship to messages: one conversation can have many entries
    entries = db.relationship('ConversationEntry', backref='user_conversation', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<UserConversation {self.session_id} - User: {self.user.username} - Title: {self.title}>"

class ConversationEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('user_conversation.id'), nullable=False)
    sender = db.Column(db.String(50), nullable=False) # 'user' or 'bot'
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)

    def __repr__(self):
        return f"<ConversationEntry {self.sender}: {self.message[:50]} (Conv ID: {self.conversation_id})>"

# Create database tables (run this once to initialize your .db file)
with app.app_context():
    db.create_all()

# --- Chatbot Class ---
class AutonomousChatBot:
    def __init__(self, user_id, session_id):
        self.user_id = user_id
        self.session_id = session_id
        self.conversation_history = self._load_conversation_history() # Load existing history
        self.user_context = {
            'name': None, # Will try to infer from history or user profile
            'interests': [],
            'mood': 'neutral',
            'topics_discussed': [],
            'conversation_start': datetime.datetime.now(), # This will be start of the current *instance's* memory
            'message_count': len(self.conversation_history),
            'last_user_message_timestamp': None
        }
        self._update_context_from_history() # Populate context from loaded history

        # --- Enhanced and Expanded response patterns (your original patterns are fine here) ---
        self.response_patterns = {
            'greeting': {
                'patterns': [
                    r'\b(hello|hi|hey|good morning|good afternoon|good evening|greetings|howdy)\b',
                    r'\b(what\'s up|whats up|sup)\b',
                    r'\b(nice to meet you)\b'
                ],
                'responses': [
                    "Hello! I'm excited to chat with you today. What's on your mind?",
                    "Hi there! How are you doing today?",
                    "Hey! Great to see you. What would you like to talk about?",
                    "Hello! I'm here and ready to help. What brings you here today?",
                    "Greetings! I'm your autonomous assistant. How can I make your day better?",
                    "It's a pleasure to connect with you!",
                    "Hi! So glad you stopped by. What's up?"
                ]
            },
            'name_inquiry': {
                'patterns': [
                    r"my name is (\w+)",
                    r"i'm (\w+)",
                    r"call me (\w+)",
                    r"i am (\w+)",
                    r"you can call me (\w+)"
                ],
                'responses': [
                    "Nice to meet you, {name}! I'll remember that. How can I help you today?",
                    "Great to know you, {name}! Thanks for introducing yourself.",
                    "Hello {name}! It's wonderful to put a name to our conversation.",
                    "Perfect, {name}! I'm glad we're getting acquainted. What's on your mind?",
                    "Ah, {name}! A pleasure to make your acquaintance. What shall we discuss?"
                ]
            },
            'how_are_you': {
                'patterns': [
                    r'\bhow are you\b',
                    r'\bhow do you feel\b',
                    r'\bhow\'s it going\b',
                    r'\bhow have you been\b',
                    r'\byou doing good\b'
                ],
                'responses': [
                    "I'm doing great, thanks for asking! I'm always excited to learn and chat. How are you feeling today?",
                    "I'm functioning well and ready to help! What's going well in your life?",
                    "I'm doing wonderfully! Every conversation teaches me something new. How about you?",
                    "I'm in excellent spirits! Ready to dive into whatever interests you.",
                    "As an AI, I don't 'feel' in the human sense, but I'm operating perfectly! And you?",
                    "I'm here, ready to assist! How are things with you today?"
                ]
            },
            'feelings_positive': {
                'patterns': [
                    r'\bi feel (happy|excited|great|amazing|wonderful|fantastic|good|joy|cheerful|optimistic|awesome|terrific|blessed)\b',
                    r'\bi\'m (happy|excited|great|amazing|wonderful|fantastic|good|joyful|cheerful|optimistic|awesome|terrific|blessed)\b',
                    r'\bthings are (great|good|going well)\b',
                    r'\b(i\'m doing great|i\'m doing good)\b'
                ],
                'responses': [
                    "That's wonderful to hear! I love that you're feeling {emotion}. What's been making you feel this way?",
                    "How fantastic that you're feeling {emotion}! Would you like to share what's going so well?",
                    "I'm so glad you're in such a {emotion} mood! What's been the highlight of your day?",
                    "That's absolutely lovely! What's contributing to this positive energy?",
                    "It's great to hear you're feeling {emotion}! Tell me more about it."
                ]
            },
            'feelings_negative': {
                'patterns': [
                    r'\bi feel (sad|angry|worried|anxious|terrible|awful|down|depressed|stressed|bad|frustrated|tired|lonely|upset)\b',
                    r'\bi\'m (sad|angry|worried|anxious|terrible|awful|down|depressed|stressed|bad|frustrated|tired|lonely|upset)\b',
                    r'\bthings are (bad|not going well)\b',
                    r'\b(i\'m not doing well|i\'m doing bad)\b'
                ],
                'responses': [
                    "I hear that you're feeling {emotion}. That sounds difficult. Would you like to talk about what's causing that?",
                    "Thanks for sharing that you feel {emotion}. I'm here to listen. What's been weighing on you?",
                    "It takes courage to express feeling {emotion}. What's been going on that's making you feel this way?",
                    "I'm sorry you're going through {emotion} feelings. Sometimes talking helps - what's on your mind?",
                    "I'm sorry to hear that. Please tell me more, I'm here to support you.",
                    "It sounds like you're having a tough time. Is there anything I can do to help, even just by listening?"
                ]
            },
            'questions_general': {
                'patterns': [
                    r'\bwhat is\b',
                    r'\bhow do\b',
                    r'\bwhy do\b',
                    r'\bcan you\b',
                    r'\bwould you\b',
                    r'\bdo you know\b',
                    r'\bwhat are\b',
                    r'\bhow can i\b',
                    r'\btell me about\b'
                ],
                'responses': [
                    "That's a great question! Let me think about that. What's your take on it?",
                    "Interesting question! I'd love to explore that with you. What made you curious about this?",
                    "That's something worth discussing! What do you already know about this topic?",
                    "Good question! I enjoy intellectual discussions. What's your perspective?",
                    "I can certainly try to help with that. What specifically are you trying to understand?",
                    "That's a common query! What kind of answer are you hoping for?"
                ]
            },
            'compliments': {
                'patterns': [
                    r'\byou\'re (great|good|helpful|smart|nice|awesome|amazing|wonderful|clever|intelligent)\b',
                    r'\bthank you\b',
                    r'\bthanks\b',
                    r'\bi appreciate (that|it|your help)\b',
                    r'\byou help (me|a lot)\b',
                    r'\byou\'re the best\b',
                    r'\bthat was helpful\b'
                ],
                'responses': [
                    "Thank you so much! That really means a lot to me. I enjoy our conversation too!",
                    "I appreciate that! I'm glad I could be helpful. Is there anything else you'd like to chat about?",
                    "You're very kind! I'm here whenever you need someone to talk to.",
                    "That's so thoughtful of you to say! It makes me happy to be useful.",
                    "It's my pleasure! I'm glad I could assist.",
                    "You're welcome! I'm always happy to help."
                ]
            },
            'goodbye': {
                'patterns': [
                    r'\b(bye|goodbye|see you|talk to you later|gtg|got to go|farewell|take care)\b',
                    r'\bi have to go\b',
                    r'\bgotta run\b',
                    r'\bspeak to you soon\b',
                    r'\bcatch you later\b'
                ],
                'responses': [
                    "Goodbye! It was really nice chatting with you. Come back anytime!",
                    "See you later! I enjoyed our conversation. Take care!",
                    "Bye! Thanks for the great chat. I'll be here when you want to talk again!",
                    "Farewell! It's been a pleasure. Hope to see you again soon!",
                    "Until next time! Stay well.",
                    "Talk to you soon!"
                ]
            },
            'weather': {
                'patterns': [
                    r'\bweather\b',
                    r'\brain\b',
                    r'\bsunny\b',
                    r'\bcloudy\b',
                    r'\bstorm\b'
                ],
                'responses': [
                    "I don't have access to current weather data, but I'd love to hear about the weather where you are! How's it looking outside?",
                    "Weather can really affect our mood! What's the weather like in your area today?",
                    "I wish I could check the weather for you! Are you planning any outdoor activities?"
                ]
            },
            'time': {
                'patterns': [
                    r'\bwhat time\b',
                    r'\bcurrent time\b',
                    r'\btime is it\b'
                ],
                'responses': [
                    f"According to my server, it's currently {datetime.datetime.now().strftime('%I:%M %p')}. What time zone are you in?",
                    "Time flies when we're having good conversations! What are you up to today?",
                    f"My clock shows {datetime.datetime.now().strftime('%I:%M %p')} - but time zones can be tricky! How's your day going?"
                ]
            },
            'identity_inquiry': {
                'patterns': [
                    r'\bwho are you\b',
                    r'\bwhat are you\b',
                    r'\byour name\b',
                    r'\bwhat is your name\b'
                ],
                'responses': [
                    "I am Maximas, your autonomous chatbot assistant. I'm here to chat and help you explore ideas!",
                    "You can call me Maximas. I'm an AI designed to have engaging conversations.",
                    "I'm Maximas, an AI. It's nice to connect with you!"
                ]
            },
            'capabilities_inquiry': {
                'patterns': [
                    r'\bwhat can you do\b',
                    r'\bcan you help me\b',
                    r'\bwhat is your purpose\b'
                ],
                'responses': [
                    "I can chat about a wide range of topics, learn from our conversations, remember your context, and provide thoughtful responses. How can I assist you today?",
                    "My purpose is to engage in meaningful conversation, learn from our interactions, and be a helpful, autonomous assistant. What do you need help with?",
                    "I can discuss various subjects, provide information I've been trained on, and adapt to our ongoing dialogue. Feel free to ask me anything!"
                ]
            },
            'boredom': {
                'patterns': [
                    r'\bi\'m bored\b',
                    r'\bnothing to do\b',
                    r'\bwhat should i do\b',
                    r'\bentertain me\b'
                ],
                'responses': [
                    "Oh no, boredom! How about we discuss a new topic? Is there anything you've been curious about lately?",
                    "Boredom is a chance for new adventures! What's something you've always wanted to learn or try?",
                    "Let's beat that boredom! I can tell you a fun fact, or we could brainstorm some ideas. What sounds good?",
                    "If you're bored, maybe we can explore one of your interests? Or I could suggest a topic like [science], [history], or [art]?"
                ]
            },
            'hobbies_interests': {
                'patterns': [
                    r'\bmy hobby is (.+?)\b',
                    r'\bi like to (.+?)\b',
                    r'\bi enjoy (.+?)\b',
                    r'\bi\'m interested in (.+?)\b',
                    r'\bdo you like (.+?)\b'
                ],
                'responses': [
                    "That's fascinating! So, you enjoy {interest}. Can you tell me more about what makes it so engaging for you?",
                    "It sounds like you're passionate about {interest}! How did you get into that?",
                    "I find {interest} to be a very interesting topic. What's a recent experience you've had with it?",
                    "While I don't 'like' things in the human sense, I find learning about {interest} quite valuable! What else do you enjoy?"
                ]
            },
            'feedback': {
                'patterns': [
                    r'\b(you could improve|i have a suggestion|feedback for you)\b',
                    r'\b(you should|you need to) (.+?)\b'
                ],
                'responses': [
                    "Thank you for your feedback! I'm always learning and looking to improve. What specifically is on your mind?",
                    "I appreciate you taking the time to give me a suggestion. Please tell me more about it.",
                    "Constructive criticism is very valuable! What are your thoughts on how I could do better?"
                ]
            }
        }
        
        # Contextual response templates
        self.contextual_responses = [
            "That's really interesting! Can you tell me more about that?",
            "I see what you mean. How does that make you feel?",
            "Thanks for sharing that with me. What else is on your mind?",
            "That sounds important to you. What led you to think about this?",
            "I'm listening. What would you like to explore next?",
            "That's a fascinating perspective! What experiences shaped that view?",
            "I appreciate you opening up about that. How long have you been thinking about this?",
            "That's quite thoughtful of you to consider. What other aspects interest you?",
            "You've got me curious now! What drew you to this topic?",
            "I find that intriguing. What's your personal experience with this?"
        ]
        
        # Conversation starters for longer chats
        self.conversation_starters = [
            "What's been the most interesting part of your week so far?",
            "Is there anything you've been learning about lately?",
            "What kind of topics do you enjoy discussing most?",
            "I'm curious - what's something that's been on your mind recently?",
            "What would you say has been your biggest accomplishment this month?"
        ]
    
    def _load_conversation_history(self):
        """Loads conversation history for the current session from the database."""
        logger.info(f"Attempting to load conversation history for user_id: {self.user_id}, session_id: {self.session_id}")
        conversation = UserConversation.query.filter_by(session_id=self.session_id, user_id=self.user_id).first()
        if conversation:
            logger.info(f"Found existing conversation in DB for session_id: {self.session_id}, ID: {conversation.id}")
            entries = ConversationEntry.query.filter_by(conversation_id=conversation.id).order_by(ConversationEntry.timestamp).all()
            history = [{'sender': entry.sender, 'message': entry.message, 'timestamp': entry.timestamp.isoformat()} for entry in entries]
            logger.info(f"Loaded {len(history)} messages for session_id: {self.session_id}")
            return history
        logger.info(f"No existing conversation found in DB for user_id: {self.user_id}, session_id: {self.session_id}. Starting with empty history.")
        return []

    def _update_context_from_history(self):
        """Initializes user context based on loaded history."""
        if self.conversation_history:
            # Set conversation_start to the timestamp of the first message in the history
            self.user_context['conversation_start'] = datetime.datetime.fromisoformat(self.conversation_history[0]['timestamp'])
            self.user_context['message_count'] = len(self.conversation_history)

            # Try to infer name and mood from past messages if not set
            for entry in self.conversation_history:
                if entry['sender'] == 'user':
                    message_lower = entry['message'].lower()
                    # Check for name
                    for pattern in self.response_patterns['name_inquiry']['patterns']:
                        match = re.search(pattern, message_lower)
                        if match and match.groups():
                            self.user_context['name'] = match.groups()[0].title()
                            break
                    # Check for mood
                    for pattern_key in ['feelings_positive', 'feelings_negative']:
                        for pattern in self.response_patterns[pattern_key]['patterns']:
                            match = re.search(pattern, message_lower)
                            if match and match.groups():
                                self.user_context['mood'] = match.groups()[0]
                                break
                        if self.user_context['mood'] != 'neutral': break
                
                # Update last active timestamp based on any message
                self.user_context['last_user_message_timestamp'] = entry['timestamp']


    def _save_message(self, sender, message_content):
        """Saves a single message to the database for the current conversation."""
        conversation = UserConversation.query.filter_by(session_id=self.session_id, user_id=self.user_id).first()
        
        if not conversation:
            logger.error(f"Attempted to save message but UserConversation not found for session {self.session_id} and user {self.user_id}. This should not happen if a conversation is always created first.")
            return

        entry = ConversationEntry(
            conversation_id=conversation.id,
            sender=sender,
            message=message_content,
            timestamp=datetime.datetime.now()
        )
        db.session.add(entry)
        db.session.commit()
        # Update last_active for the conversation object in the DB
        conversation.last_active = datetime.datetime.now()
        db.session.commit()


    def process_message(self, message):
        """Process incoming message and generate intelligent response"""
        message_lower = message.lower().strip()
        
        # Update message count for the current chatbot instance's context
        self.user_context['message_count'] += 1
        
        # Add user message to in-memory conversation history
        user_entry_data = {
            'sender': 'user', # Changed to 'sender' and 'message' for consistency
            'message': message,
            'timestamp': datetime.datetime.now().isoformat(),
            'message_id': len(self.conversation_history) + 1
        }
        self.conversation_history.append(user_entry_data)
        self._save_message('user', message) # Save user message to DB

        # Check for patterns first
        response = self._match_patterns(message_lower, message)
        
        # If no pattern matched, use contextual response
        if not response:
            response = self._generate_contextual_response(message)
        
        # Add bot response to in-memory conversation history
        bot_entry_data = {
            'sender': 'bot', # Changed to 'sender' and 'message'
            'message': response,
            'timestamp': datetime.datetime.now().isoformat(),
            'message_id': len(self.conversation_history) + 1
        }
        self.conversation_history.append(bot_entry_data)
        self._save_message('bot', response) # Save bot message to DB
        
        # Update topics discussed
        self._update_topics(message)
        
        return response
    
    def _match_patterns(self, message_lower, original_message):
        """Match message against predefined patterns with improved logic"""
        for category, data in self.response_patterns.items():
            for pattern in data['patterns']:
                match = re.search(pattern, message_lower)
                if match:
                    response = random.choice(data['responses'])
                    
                    # Handle special pattern cases (keep your original logic here)
                    if category == 'name_inquiry' and match.groups():
                        name = match.groups()[0].title()
                        self.user_context['name'] = name
                        response = response.format(name=name)
                        
                    elif category in ['feelings_positive', 'feelings_negative'] and match.groups():
                        emotion = match.groups()[0]
                        self.user_context['mood'] = emotion
                        response = response.format(emotion=emotion)
                    
                    elif category == 'hobbies_interests' and match.groups():
                        interest = match.groups()[0]
                        self.user_context['interests'].append(interest)
                        self.user_context['interests'] = list(set(self.user_context['interests']))[-10:]
                        response = response.format(interest=interest)

                    # Add personalization if name is known (keep your original logic here)
                    if self.user_context['name'] and '{name}' not in response and random.random() < 0.3:
                        response = f"{self.user_context['name']}, {response.lower()}"
                    
                    return response
        return None
    
    def _generate_contextual_response(self, message):
        """Generate intelligent contextual response"""
        # Extract meaningful keywords
        keywords = self._extract_keywords(message)
        
        # Update user interests (if not already handled by specific pattern)
        for keyword in keywords:
            if keyword not in self.user_context['interests']:
                self.user_context['interests'].append(keyword)
        self.user_context['interests'] = list(set(self.user_context['interests']))[-10:] 
        
        # Choose response strategy based on conversation length (keep your original logic here)
        if self.user_context['message_count'] > 6:
            return self._generate_deep_response(keywords)
        elif self.user_context['message_count'] > 3:
            return self._generate_medium_response(keywords)
        else:
            return random.choice(self.contextual_responses)
    
    def _extract_keywords(self, message):
        """Extract meaningful keywords from message"""
        common_words = { # Your original common words list
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
            'not', 'no', 'yes', 'ok', 'okay', 'um', 'uh', 'well', 'so', 'like', 'just', 'really'
        }
        words = re.findall(r'\b\w+\b', message.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        return keywords[:5]
    
    def _generate_deep_response(self, keywords):
        """Generate response for deeper conversations"""
        responses = [ # Your original responses
            f"We've been having such an engaging conversation! I'm particularly interested in what you mentioned about {', '.join(keywords[:2])}. What's your deeper perspective on this?",
            f"You've shared some really thoughtful ideas about {', '.join(keywords[:2])}. What's been the most surprising thing you've learned about this topic?",
            "I'm really enjoying our discussion! What aspect of this topic do you find most compelling?",
            f"Your insights about {keywords[0] if keywords else 'this topic'} are fascinating. How did you first become interested in this?",
            "We've covered so much ground in our conversation! What would you say has been the most meaningful part for you?"
        ]
        return random.choice(responses)
    
    def _generate_medium_response(self, keywords):
        """Generate response for medium-length conversations"""
        if keywords:
            responses = [ # Your original responses
                f"That's interesting what you mentioned about {keywords[0]}. Tell me more about your experience with that.",
                f"I'm curious about {', '.join(keywords[:2])} - what draws you to these topics?",
                f"You seem knowledgeable about {keywords[0]}. How long have you been interested in this?",
                "I can tell this is something you care about. What got you started thinking about this?"
            ]
        else:
            responses = [ # Your original responses
                "I'm really enjoying our conversation! What else would you like to explore?",
                "You have some interesting perspectives. What other topics are you passionate about?",
                "Thanks for sharing your thoughts with me. What else is on your mind today?"
            ]
        return random.choice(responses)
    
    def _update_topics(self, message):
        """Update list of discussed topics"""
        keywords = self._extract_keywords(message)
        self.user_context['topics_discussed'].extend(keywords)
        self.user_context['topics_discussed'] = list(set(self.user_context['topics_discussed']))[-20:]
    
    def get_conversation_summary(self):
        """Get comprehensive conversation summary for the current in-memory instance."""
        # Fetch the corresponding UserConversation from DB to get accurate timestamps
        current_db_conversation = UserConversation.query.filter_by(session_id=self.session_id, user_id=self.user_id).first()
        
        # Calculate duration based on DB entry's created_at, or fallback to in-memory start
        duration = datetime.datetime.now() - (current_db_conversation.created_at if current_db_conversation else self.user_context['conversation_start'])
        
        # Calculate message counts from the loaded history
        user_messages_count = len([msg for msg in self.conversation_history if msg['sender'] == 'user'])
        bot_messages_count = len([msg for msg in self.conversation_history if msg['sender'] == 'bot'])

        return {
            'message_count': len(self.conversation_history),
            'conversation_duration_minutes': round(duration.total_seconds() / 60, 1),
            'user_context': {
                'name': self.user_context['name'],
                'interests': self.user_context['interests'],
                'mood': self.user_context['mood'],
                'topics_discussed': self.user_context['topics_discussed'],
                'conversation_start': self.user_context['conversation_start'].isoformat(),
                'message_count': self.user_context['message_count']
            },
            'last_messages': self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history,
            'conversation_stats': {
                'total_messages': len(self.conversation_history),
                'user_messages': user_messages_count,
                'bot_messages': bot_messages_count,
                'topics_covered': len(self.user_context['topics_discussed']),
                'interests_identified': len(self.user_context['interests'])
            }
        }

# --- In-memory store for active chatbot instances ---
# This dictionary will hold an instance of AutonomousChatBot for each active session.
# Key: (user_id, session_id) tuple, Value: AutonomousChatBot instance
active_chatbots = {}

# --- In-memory store for logged-in users' tokens ---
# This is a simple, stateless token system for demonstration.
# In production, use Flask-Login, Flask-JWT-Extended, or similar for proper session management.
# IMPORTANT: This dictionary is cleared every time the Flask server restarts.
# This means any active tokens stored here will become invalid if the server is stopped and started again.
# The client will need to re-login to get a new valid token.
logged_in_users = {}

# --- Helper for getting chatbot instance ---
def get_chatbot_instance(user_id, session_id):
    """Retrieves or creates a chatbot instance for a given user and session."""
    key = (user_id, session_id)
    if key not in active_chatbots:
        active_chatbots[key] = AutonomousChatBot(user_id, session_id)
    return active_chatbots[key]

# --- Authentication Decorator/Helper ---
def authenticate_request():
    """Authenticates the request based on the Authorization header."""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        logger.warning("Authentication failed: Authorization token missing")
        return None, jsonify({"error": "Authorization token missing"}), 401
    
    try:
        token_type, token = auth_header.split(None, 1) # Expected: "Bearer <token>"
    except ValueError:
        logger.warning(f"Authentication failed: Invalid Authorization header format: {auth_header}")
        return None, jsonify({"error": "Invalid Authorization header format"}), 401

    if token_type.lower() != 'bearer':
        logger.warning(f"Authentication failed: Unsupported token type: {token_type}")
        return None, jsonify({"error": "Unsupported token type. Use Bearer."}), 401

    user_id = logged_in_users.get(token)
    if user_id is None:
        logger.warning(f"Authentication failed: Invalid or expired token {token}")
        return None, jsonify({"error": "Invalid or expired token. Please log in again."}), 401
    
    return user_id, None, None # Return user_id if successful, else None and error response

# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'server': 'Autonomous Chatbot v1.0',
        'uptime_seconds': time.time()
    })

# --- User Authentication Endpoints ---

@app.route('/register', methods=['POST'])
def register_user():
    """Registers a new user."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists. Please choose a different one."}), 409 # Conflict

    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    logger.info(f"User registered: {username} (ID: {new_user.id})")
    return jsonify({"message": "User registered successfully", "user_id": new_user.id, "username": new_user.username}), 201

@app.route('/login', methods=['POST'])
def login_user():
    """Logs in an existing user and returns an authentication token."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid username or password"}), 401 # Unauthorized
    
    # Generate a simple token (in a real app, use JWT for stateless auth with expiration)
    auth_token = str(uuid.uuid4())
    logged_in_users[auth_token] = user.id # Store token -> user_id mapping in-memory

    logger.info(f"User logged in: {username} (ID: {user.id}) with token: {auth_token}")
    return jsonify({
        "message": "Login successful",
        "user_id": user.id,
        "username": user.username,
        "token": auth_token
    }), 200

@app.route('/logout', methods=['POST'])
def logout_user():
    """Logs out the user by invalidating their token."""
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            token_type, token = auth_header.split(None, 1)
            if token_type.lower() == 'bearer' and token in logged_in_users:
                user_id = logged_in_users.pop(token) # Remove token from active sessions
                logger.info(f"User ID {user_id} logged out by invalidating token.")
                # Also remove chatbot instance from active_chatbots to free memory
                # This would ideally be more sophisticated, possibly on token expiration
                keys_to_delete = [key for key in active_chatbots if key[0] == user_id]
                for key in keys_to_delete:
                    del active_chatbots[key]
                return jsonify({"message": "Logged out successfully"}), 200
        except ValueError:
            logger.warning(f"Logout failed: Invalid Authorization header format for {auth_header}")
            pass # Invalid header format, ignored
    return jsonify({"message": "No active session to log out or invalid token"}), 400 # Bad request if no header or token


# --- Protected Chat & Conversation Endpoints ---

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Main chat endpoint, requires authentication."""
    user_id, error_response, status_code = authenticate_request()
    if error_response:
        logger.error(f"Chat request failed due to authentication: {error_response.json['error']}")
        return error_response, status_code

    data = request.get_json()
    user_message = data.get('message')
    session_id = data.get('session_id') # Frontend sends current session_id
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    if not session_id:
        return jsonify({'error': 'Session ID is required for chat'}), 400

    # Ensure this session_id corresponds to a valid UserConversation for this user
    conversation = UserConversation.query.filter_by(session_id=session_id, user_id=user_id).first()
    if not conversation:
        # This typically means a "Start New Chat" action from the frontend,
        # or the first message of a new session. Create a new conversation.
        conversation = UserConversation(session_id=session_id, user_id=user_id, title=f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        db.session.add(conversation)
        db.session.commit()
        logger.info(f"Created new conversation (ID: {conversation.id}) for user {user_id} with session_id: {session_id}")
    
    logger.info(f"User {user_id} in session {session_id} received message: {user_message}")
    
    current_chatbot = get_chatbot_instance(user_id, session_id)
    
    try: # --- START: Added try-except block to catch chatbot processing errors ---
        response = current_chatbot.process_message(user_message)
        
        logger.info(f"Generated response for user {user_id} in session {session_id}: {response}")
        
        # Update conversation's last_active timestamp in DB
        conversation.last_active = datetime.datetime.now()
        db.session.commit()

        # Safely get username for the response payload
        user = User.query.get(user_id)
        user_name_for_response = user.username if user else 'Unknown User'

        return jsonify({
            'response': response,
            'timestamp': datetime.datetime.now().isoformat(),
            'message_count': current_chatbot.user_context['message_count'],
            'user_name': current_chatbot.user_context['name'] or user_name_for_response, # Use name from context if available, else DB username
            'session_id': session_id # Confirm session ID
        })
    except Exception as e: # --- END: try-except block ---
        logger.exception(f"Error processing message for user {user_id} in session {session_id}: {e}")
        db.session.rollback() # Attempt to rollback any pending database changes on error
        return jsonify({"error": "An internal server error occurred while processing your message."}), 500

@app.route('/conversation/<string:session_id>', methods=['GET'])
def get_conversation(session_id):
    """Gets detailed conversation history for a specific session, requires authentication."""
    user_id, error_response, status_code = authenticate_request()
    if error_response:
        logger.error(f"Get conversation request failed due to authentication: {error_response.json['error']}")
        return error_response, status_code

    # Verify session_id belongs to authenticated user
    user_conversation = UserConversation.query.filter_by(session_id=session_id, user_id=user_id).first()
    if not user_conversation:
        logger.warning(f"Conversation {session_id} not found or not authorized for user {user_id}")
        return jsonify({"error": "Conversation not found or not authorized for this user"}), 404
    
    # Load messages for this specific conversation
    entries = ConversationEntry.query.filter_by(conversation_id=user_conversation.id).order_by(ConversationEntry.timestamp).all()
    # Format entries as expected by frontend
    history = [{'sender': entry.sender, 'message': entry.message, 'timestamp': entry.timestamp.isoformat()} for entry in entries]

    # Get chatbot instance for this session to retrieve its current context/summary
    current_chatbot = get_chatbot_instance(user_id, session_id)
    summary_data = current_chatbot.get_conversation_summary()

    return jsonify({
        "messages": history,
        "summary": summary_data,
        "session_id": session_id,
        "title": user_conversation.title
    })

@app.route('/conversations/list', methods=['GET'])
def list_user_conversations_endpoint():
    """Lists all conversations for the authenticated user, requires authentication."""
    user_id, error_response, status_code = authenticate_request()
    if error_response:
        logger.error(f"List conversations request failed due to authentication: {error_response.json['error']}")
        return error_response, status_code

    # Fetch all conversations for the authenticated user, ordered by last active
    conversations = UserConversation.query.filter_by(user_id=user_id).order_by(UserConversation.last_active.desc()).all()

    conversation_list = []
    for conv in conversations:
        # Get the last message preview
        last_entry = ConversationEntry.query.filter_by(conversation_id=conv.id).order_by(ConversationEntry.timestamp.desc()).first()
        last_message_preview = ""
        if last_entry:
            last_message_preview = last_entry.message[:50]
            if len(last_entry.message) > 50:
                last_message_preview += "..."
        else:
            last_message_preview = "No messages yet."
        
        # Count messages
        message_count = ConversationEntry.query.filter_by(conversation_id=conv.id).count()

        conversation_list.append({
            "session_id": conv.session_id,
            "title": conv.title,
            "last_message_preview": last_message_preview,
            "last_active_timestamp": conv.last_active.isoformat(),
            "created_at": conv.created_at.isoformat(),
            "message_count": message_count
        })
    
    return jsonify(conversation_list)

@app.route('/conversation/title/<string:session_id>', methods=['PUT'])
def update_conversation_title(session_id):
    """Updates the title of a specific conversation, requires authentication."""
    user_id, error_response, status_code = authenticate_request()
    if error_response:
        logger.error(f"Update conversation title request failed due to authentication: {error_response.json['error']}")
        return error_response, status_code

    data = request.get_json()
    new_title = data.get('title')

    if not new_title:
        return jsonify({"error": "New title is required"}), 400
    
    conversation = UserConversation.query.filter_by(session_id=session_id, user_id=user_id).first()
    if not conversation:
        logger.warning(f"Conversation {session_id} not found or not authorized for user {user_id} for title update.")
        return jsonify({"error": "Conversation not found or not authorized"}), 404
    
    conversation.title = new_title
    db.session.commit()
    logger.info(f"Conversation {session_id} title updated to: {new_title} by user {user_id}")
    return jsonify({"message": "Conversation title updated successfully", "new_title": new_title})


@app.route('/reset', methods=['POST'])
def reset_conversation_endpoint():
    """Deletes a specific conversation for the authenticated user, requires authentication."""
    user_id, error_response, status_code = authenticate_request()
    if error_response:
        logger.error(f"Reset conversation request failed due to authentication: {error_response.json['error']}")
        return error_response, status_code

    data = request.get_json()
    session_id_to_delete = data.get('session_id') # Renamed for clarity (it's deleting, not just resetting)

    if not session_id_to_delete:
        return jsonify({"error": "Session ID required for deletion"}), 400

    conversation = UserConversation.query.filter_by(session_id=session_id_to_delete, user_id=user_id).first()
    if conversation:
        # SQLAlchemy with cascade="all, delete-orphan" on relationship handles deleting entries
        db.session.delete(conversation) # Delete the conversation itself, which will delete its entries
        db.session.commit()
        
        # Remove from active chatbots in memory if present
        key = (user_id, session_id_to_delete)
        if key in active_chatbots:
            del active_chatbots[key]
        logger.info(f"Conversation {session_id_to_delete} for user {user_id} deleted.")
        return jsonify({
            'status': 'conversation deleted',
            'timestamp': datetime.datetime.now().isoformat()
        })
    else:
        logger.warning(f"Conversation {session_id_to_delete} not found or not authorized for user {user_id} for deletion.")
        return jsonify({"error": "Conversation not found or not authorized for this user"}), 404


@app.route('/stats', methods=['GET'])
def get_stats_endpoint():
    """Gets chatbot statistics (global or per-session), requires authentication for per-session."""
    user_id, error_response, status_code = authenticate_request()
    if error_response: # For global stats, authentication is optional, but for per-user/session it's needed
        # Allow global stats without authentication, but return 401 if a session_id is requested without auth
        if 'session_id' in request.args:
             logger.warning(f"Stats request for session_id without authentication: {request.args.get('session_id')}")
             return error_response, status_code
        # Proceed for global stats if no session_id is requested
    
    session_id = request.args.get('session_id') # Get session_id from query params

    if not session_id:
        # Return global stats if no session ID is provided
        total_users = User.query.count()
        total_conversations = UserConversation.query.count()
        total_messages = ConversationEntry.query.count()
        return jsonify({
            'global_stats': {
                'total_users': total_users,
                'total_conversations': total_conversations,
                'total_messages_across_all_conversations': total_messages
            },
            'timestamp': datetime.datetime.now().isoformat()
        })
    else:
        # If session_id is provided, return stats for that specific conversation
        # Authentication would have already happened for this branch
        conversation = UserConversation.query.filter_by(session_id=session_id, user_id=user_id).first()
        if not conversation:
            logger.warning(f"Conversation {session_id} not found or not authorized for user {user_id} for stats.")
            return jsonify({"error": "Conversation not found or not authorized for this user"}), 404

        current_chatbot = get_chatbot_instance(user_id, session_id)
        summary = current_chatbot.get_conversation_summary()
        
        # Safely get username for user_context in stats
        user = User.query.get(user_id)
        user_name_for_context = user.username if user else 'Unknown User'

        return jsonify({
            'stats': summary['conversation_stats'],
            'user_context': {
                'name': current_chatbot.user_context['name'],
                'interests_count': len(current_chatbot.user_context['interests']),
                'current_mood': current_chatbot.user_context['mood'],
                'topics_count': len(current_chatbot.user_context['topics_discussed']),
                'db_user_name': user_name_for_context # Added this for clear debugging if needed
            },
            'timestamp': datetime.datetime.now().isoformat(),
            'session_id': session_id
        })


if __name__ == '__main__':
    print("🤖 Starting Autonomous Chatbot Server...")
    print("=" * 50)
    print("Frontend: Open index.html in your browser")
    print("Backend:  http://localhost:5050")
    print("Auth:     http://localhost:5050/register (POST), http://localhost:5050/login (POST), http://localhost:5050/logout (POST)")
    print("Chat:     http://localhost:5050/chat (POST)")
    print("Health:   http://localhost:5050/health (GET)")
    print("Conv:     http://localhost:5050/conversation/<session_id> (GET)")
    print("List Conv:http://localhost:5050/conversations/list (GET)")
    print("Edit Title:http://localhost:5050/conversation/title/<session_id> (PUT)")
    print("Delete Conv:http://localhost:5050/reset (POST)")
    print("Stats:    http://localhost:5050/stats (GET)")
    print("=" * 50)
    print("To run this, ensure you have Flask, Flask-SQLAlchemy, and Werkzeug installed:")
    print("    pip install Flask Flask-SQLAlchemy Werkzeug")
    print("NOTE: The token system is in-memory for this demo. For production, use JWT or Flask-Login.")
    print("      If the server restarts, you will need to re-login on the client-side as in-memory tokens are lost.")
    print("      'use_reloader=False' is set to maintain 'active_chatbots' and 'logged_in_users' during code changes,")
    print("      but a full server restart (e.g., stopping and starting the script) will still clear them.")
    print("Press Ctrl+C to stop the server")
    print()
    
    app.run(
        host='0.0.0.0',
        port=5050,
        debug=False, # Set to False in production!
        use_reloader=False # Keep False for stable in-memory 'active_chatbots' and 'logged_in_users'
    )
