#!/usr/bin/env python3
"""
Autonomous Chatbot Backend Server
A Flask server providing intelligent conversational AI with pattern matching,
context awareness, and memory capabilities.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import random
import re
import datetime
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

class AutonomousChatBot:
    def __init__(self):
        self.conversation_history = []
        self.user_context = {
            'name': None,
            'interests': [],
            'mood': 'neutral',
            'topics_discussed': [],
            'conversation_start': datetime.datetime.now(),
            'message_count': 0
        }
        
        # Enhanced response patterns with more intelligence
        self.response_patterns = {
            'greeting': {
                'patterns': [
                    r'\b(hello|hi|hey|good morning|good afternoon|good evening|greetings|howdy)\b',
                    r'\b(what\'s up|whats up|sup)\b'
                ],
                'responses': [
                    "Hello! I'm excited to chat with you today. What's on your mind?",
                    "Hi there! How are you doing today?",
                    "Hey! Great to see you. What would you like to talk about?",
                    "Hello! I'm here and ready to help. What brings you here today?",
                    "Greetings! I'm your autonomous assistant. How can I make your day better?"
                ]
            },
            'name_inquiry': {
                'patterns': [
                    r"my name is (\w+)",
                    r"i'm (\w+)",
                    r"call me (\w+)",
                    r"i am (\w+)"
                ],
                'responses': [
                    "Nice to meet you, {name}! I'll remember that. How can I help you today?",
                    "Great to know you, {name}! Thanks for introducing yourself.",
                    "Hello {name}! It's wonderful to put a name to our conversation.",
                    "Perfect, {name}! I'm glad we're getting acquainted."
                ]
            },
            'how_are_you': {
                'patterns': [
                    r'\bhow are you\b',
                    r'\bhow do you feel\b',
                    r'\bhow\'s it going\b',
                    r'\bhow have you been\b'
                ],
                'responses': [
                    "I'm doing great, thanks for asking! I'm always excited to learn and chat. How are you feeling today?",
                    "I'm functioning well and ready to help! What's going well in your life?",
                    "I'm doing wonderfully! Every conversation teaches me something new. How about you?",
                    "I'm in excellent spirits! Ready to dive into whatever interests you."
                ]
            },
            'feelings_positive': {
                'patterns': [
                    r'\bi feel (happy|excited|great|amazing|wonderful|fantastic|good|joy|cheerful)\b',
                    r'\bi\'m (happy|excited|great|amazing|wonderful|fantastic|good|joyful|cheerful)\b'
                ],
                'responses': [
                    "That's wonderful to hear! I love that you're feeling {emotion}. What's been making you feel this way?",
                    "How fantastic that you're feeling {emotion}! Would you like to share what's going so well?",
                    "I'm so glad you're in such a {emotion} mood! What's been the highlight of your day?"
                ]
            },
            'feelings_negative': {
                'patterns': [
                    r'\bi feel (sad|angry|worried|anxious|terrible|awful|down|depressed|stressed)\b',
                    r'\bi\'m (sad|angry|worried|anxious|terrible|awful|down|depressed|stressed)\b'
                ],
                'responses': [
                    "I hear that you're feeling {emotion}. That sounds difficult. Would you like to talk about what's causing that?",
                    "Thanks for sharing that you feel {emotion}. I'm here to listen. What's been weighing on you?",
                    "It takes courage to express feeling {emotion}. What's been going on that's making you feel this way?",
                    "I'm sorry you're going through {emotion} feelings. Sometimes talking helps - what's on your mind?"
                ]
            },
            'questions_general': {
                'patterns': [
                    r'\bwhat is\b',
                    r'\bhow do\b',
                    r'\bwhy do\b',
                    r'\bcan you\b',
                    r'\bwould you\b',
                    r'\bdo you know\b'
                ],
                'responses': [
                    "That's a great question! Let me think about that. What's your take on it?",
                    "Interesting question! I'd love to explore that with you. What made you curious about this?",
                    "That's something worth discussing! What do you already know about this topic?",
                    "Good question! I enjoy intellectual discussions. What's your perspective?"
                ]
            },
            'compliments': {
                'patterns': [
                    r'\byou\'re (great|good|helpful|smart|nice|awesome|amazing|wonderful)\b',
                    r'\bthank you\b',
                    r'\bthanks\b',
                    r'\bi appreciate\b',
                    r'\byou help\b'
                ],
                'responses': [
                    "Thank you so much! That really means a lot to me. I enjoy our conversation too!",
                    "I appreciate that! I'm glad I could be helpful. Is there anything else you'd like to chat about?",
                    "You're very kind! I'm here whenever you need someone to talk to.",
                    "That's so thoughtful of you to say! It makes me happy to be useful."
                ]
            },
            'goodbye': {
                'patterns': [
                    r'\b(bye|goodbye|see you|talk to you later|gtg|got to go|farewell|take care)\b',
                    r'\bi have to go\b',
                    r'\bgotta run\b'
                ],
                'responses': [
                    "Goodbye! It was really nice chatting with you. Come back anytime!",
                    "See you later! I enjoyed our conversation. Take care!",
                    "Bye! Thanks for the great chat. I'll be here when you want to talk again!",
                    "Farewell! It's been a pleasure. Hope to see you again soon!"
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
    
    def process_message(self, message):
        """Process incoming message and generate intelligent response"""
        message_lower = message.lower().strip()
        
        # Update message count
        self.user_context['message_count'] += 1
        
        # Add to conversation history
        self.conversation_history.append({
            'user': message,
            'timestamp': datetime.datetime.now().isoformat(),
            'message_id': len(self.conversation_history) + 1
        })
        
        # Check for patterns first
        response = self._match_patterns(message_lower, message)
        
        # If no pattern matched, use contextual response
        if not response:
            response = self._generate_contextual_response(message)
        
        # Add response to history
        self.conversation_history.append({
            'bot': response,
            'timestamp': datetime.datetime.now().isoformat(),
            'message_id': len(self.conversation_history) + 1
        })
        
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
                    
                    # Handle special pattern cases
                    if category == 'name_inquiry' and match.groups():
                        name = match.groups()[0].title()
                        self.user_context['name'] = name
                        response = response.format(name=name)
                        
                    elif category in ['feelings_positive', 'feelings_negative'] and match.groups():
                        emotion = match.groups()[0]
                        self.user_context['mood'] = emotion
                        response = response.format(emotion=emotion)
                    
                    # Add personalization if name is known
                    if self.user_context['name'] and '{name}' not in response:
                        if random.random() < 0.3:  # 30% chance to add name
                            response = f"{self.user_context['name']}, {response.lower()}"
                    
                    return response
        return None
    
    def _generate_contextual_response(self, message):
        """Generate intelligent contextual response"""
        # Extract meaningful keywords
        keywords = self._extract_keywords(message)
        
        # Update user interests
        self.user_context['interests'].extend(keywords)
        self.user_context['interests'] = list(set(self.user_context['interests']))[-10:]  # Keep last 10 interests
        
        # Choose response strategy based on conversation length
        if self.user_context['message_count'] > 6:  # Deep conversation
            return self._generate_deep_response(keywords)
        elif self.user_context['message_count'] > 3:  # Medium conversation
            return self._generate_medium_response(keywords)
        else:  # Early conversation
            return random.choice(self.contextual_responses)
    
    def _extract_keywords(self, message):
        """Extract meaningful keywords from message"""
        # Remove common words
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
            'not', 'no', 'yes', 'ok', 'okay', 'um', 'uh', 'well', 'so', 'like', 'just', 'really'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', message.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        
        return keywords[:5]  # Return top 5 keywords
    
    def _generate_deep_response(self, keywords):
        """Generate response for deeper conversations"""
        responses = [
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
            responses = [
                f"That's interesting what you mentioned about {keywords[0]}. Tell me more about your experience with that.",
                f"I'm curious about {', '.join(keywords[:2])} - what draws you to these topics?",
                f"You seem knowledgeable about {keywords[0]}. How long have you been interested in this?",
                "I can tell this is something you care about. What got you started thinking about this?"
            ]
        else:
            responses = [
                "I'm really enjoying our conversation! What else would you like to explore?",
                "You have some interesting perspectives. What other topics are you passionate about?",
                "Thanks for sharing your thoughts with me. What else is on your mind today?"
            ]
        return random.choice(responses)
    
    def _update_topics(self, message):
        """Update list of discussed topics"""
        keywords = self._extract_keywords(message)
        self.user_context['topics_discussed'].extend(keywords)
        # Keep unique topics, limited to last 20
        self.user_context['topics_discussed'] = list(set(self.user_context['topics_discussed']))[-20:]
    
    def get_conversation_summary(self):
        """Get comprehensive conversation summary"""
        duration = datetime.datetime.now() - self.user_context['conversation_start']
        
        return {
            'message_count': len(self.conversation_history),
            'conversation_duration_minutes': round(duration.total_seconds() / 60, 1),
            'user_context': self.user_context,
            'last_messages': self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history,
            'conversation_stats': {
                'total_messages': len(self.conversation_history),
                'user_messages': len([msg for msg in self.conversation_history if 'user' in msg]),
                'bot_messages': len([msg for msg in self.conversation_history if 'bot' in msg]),
                'topics_covered': len(self.user_context['topics_discussed']),
                'interests_identified': len(self.user_context['interests'])
            }
        }

# Global chatbot instance
chatbot = AutonomousChatBot()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'server': 'Autonomous Chatbot v1.0',
        'uptime_seconds': time.time()
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with enhanced error handling"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        logger.info(f"Received message: {user_message}")
        
        # Process message through chatbot
        response = chatbot.process_message(user_message)
        
        logger.info(f"Generated response: {response}")
        
        return jsonify({
            'response': response,
            'timestamp': datetime.datetime.now().isoformat(),
            'message_count': chatbot.user_context['message_count'],
            'user_name': chatbot.user_context['name']
        })
    
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        return jsonify({
            'error': 'I encountered an error processing your message. Please try again.',
            'timestamp': datetime.datetime.now().isoformat()
        }), 500

@app.route('/conversation', methods=['GET'])
def get_conversation():
    """Get detailed conversation history and analytics"""
    try:
        summary = chatbot.get_conversation_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation history and start fresh"""
    try:
        global chatbot
        chatbot = AutonomousChatBot()
        logger.info("Conversation reset")
        return jsonify({
            'status': 'conversation reset',
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error resetting conversation: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get chatbot statistics"""
    try:
        summary = chatbot.get_conversation_summary()
        return jsonify({
            'stats': summary['conversation_stats'],
            'user_context': {
                'name': chatbot.user_context['name'],
                'interests_count': len(chatbot.user_context['interests']),
                'current_mood': chatbot.user_context['mood'],
                'topics_count': len(chatbot.user_context['topics_discussed'])
            },
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🤖 Starting Autonomous Chatbot Server...")
    print("=" * 50)
    print("Frontend: Open index.html in your browser")
    print("Backend:  http://localhost:5000")
    print("Health:   http://localhost:5000/health")
    print("Stats:    http://localhost:5000/stats")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    app.run(
        host='0.0.0.0',
        port=5050,
        debug=True,
        use_reloader=True
    )