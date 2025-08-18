import praw
import asyncio
import random
import time
import requests
import json
from datetime import datetime, timedelta
from supabase import create_client, Client
import re
from typing import Dict, List, Optional, Set
import logging
from dataclasses import dataclass
import html
import unicodedata
from difflib import SequenceMatcher
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserStats:
    """User statistics data class"""
    username: str
    total_score: int = 0
    correct_answers: int = 0
    total_questions: int = 0
    streak: int = 0
    max_streak: int = 0
    last_active: datetime = None
    daily_score: int = 0
    weekly_score: int = 0
    monthly_score: int = 0

class AnswerMatcher:
    """Advanced answer matching system with flexible comparison"""
    
    def __init__(self):
        # Common abbreviations and their full forms
        self.abbreviations = {
            'usa': 'united states of america',
            'us': 'united states',
            'uk': 'united kingdom',
            'ussr': 'soviet union',
            'wwi': 'world war 1',
            'ww1': 'world war 1',
            'wwii': 'world war 2',
            'ww2': 'world war 2',
            'nyc': 'new york city',
            'la': 'los angeles',
            'sf': 'san francisco',
            'dc': 'washington dc',
            'ca': 'california',
            'ny': 'new york',
            'jr': 'junior',
            'sr': 'senior',
            'dr': 'doctor',
            'mr': 'mister',
            'mrs': 'missus',
            'ms': 'miss',
            'prof': 'professor',
            'st': 'saint',
            'mt': 'mount',
            'ft': 'fort',
            'co': 'company',
            'corp': 'corporation',
            'inc': 'incorporated',
            'ltd': 'limited',
            'etc': 'et cetera',
            'vs': 'versus',
            'aka': 'also known as'
        }
        
        # Roman numerals
        self.roman_numerals = {
            'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5',
            'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10',
            'xi': '11', 'xii': '12', 'xiii': '13', 'xiv': '14', 'xv': '15',
            'xvi': '16', 'xvii': '17', 'xviii': '18', 'xix': '19', 'xx': '20'
        }
        
        # Words to ignore in comparison
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'as', 'is', 'was', 'are', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'shall'
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove HTML entities
        text = html.unescape(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove accents and diacritics
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        # Replace common punctuation and symbols
        replacements = {
            '&': 'and',
            '+': 'plus',
            '%': 'percent',
            '$': 'dollar',
            '€': 'euro',
            '£': 'pound',
            '¥': 'yen',
            '@': 'at',
            '#': 'number',
            '°': 'degree',
            '½': '0.5',
            '¼': '0.25',
            '¾': '0.75',
            '²': '2',
            '³': '3'
        }
        
        for symbol, replacement in replacements.items():
            text = text.replace(symbol, replacement)
        
        # Remove extra punctuation but keep some important ones
        text = re.sub(r'[^\w\s\.\-\']', ' ', text)
        
        # Handle abbreviations
        words = text.split()
        normalized_words = []
        for word in words:
            clean_word = word.strip('.,!?;:"()[]{}')
            if clean_word in self.abbreviations:
                normalized_words.append(self.abbreviations[clean_word])
            elif clean_word in self.roman_numerals:
                normalized_words.append(self.roman_numerals[clean_word])
            else:
                normalized_words.append(word)
        
        text = ' '.join(normalized_words)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def extract_key_words(self, text: str) -> Set[str]:
        """Extract key words for comparison"""
        normalized = self.normalize_text(text)
        words = normalized.split()
        
        # Remove stop words and very short words
        key_words = set()
        for word in words:
            clean_word = word.strip('.,!?;:"()[]{}')
            if len(clean_word) > 1 and clean_word not in self.stop_words:
                key_words.add(clean_word)
        
        return key_words

    def similarity_score(self, str1: str, str2: str) -> float:
        """Calculate similarity score between two strings"""
        return SequenceMatcher(None, str1, str2).ratio()

    def is_match(self, user_answer: str, correct_answer: str, threshold: float = 0.8) -> tuple[bool, float, str]:
        """
        Check if user answer matches correct answer
        Returns: (is_match, confidence_score, match_type)
        """
        if not user_answer or not correct_answer:
            return False, 0.0, "empty"
        
        # Normalize both answers
        user_norm = self.normalize_text(user_answer)
        correct_norm = self.normalize_text(correct_answer)
        
        # Exact match after normalization
        if user_norm == correct_norm:
            return True, 1.0, "exact"
        
        # Check if user answer contains the correct answer or vice versa
        if user_norm in correct_norm or correct_norm in user_norm:
            return True, 0.95, "contains"
        
        # Key words matching
        user_words = self.extract_key_words(user_answer)
        correct_words = self.extract_key_words(correct_answer)
        
        if user_words and correct_words:
            # Check if all important words match
            intersection = user_words.intersection(correct_words)
            union = user_words.union(correct_words)
            
            if union:
                word_similarity = len(intersection) / len(union)
                if word_similarity >= 0.7:  # 70% of words match
                    return True, word_similarity, "keywords"
        
        # Fuzzy string matching
        similarity = self.similarity_score(user_norm, correct_norm)
        if similarity >= threshold:
            return True, similarity, "fuzzy"
        
        # Check for partial matches in multi-word answers
        if len(correct_words) > 1:
            user_text_clean = ' '.join(user_words)
            correct_text_clean = ' '.join(correct_words)
            
            # Check if user got the main words right
            essential_words = [w for w in correct_words if len(w) > 3]  # Important words are usually longer
            if essential_words:
                matched_essential = sum(1 for word in essential_words if word in user_words)
                essential_ratio = matched_essential / len(essential_words)
                if essential_ratio >= 0.8:  # 80% of essential words match
                    return True, essential_ratio, "essential"
        
        # Handle initials and abbreviations (like "G.H. Hardy" vs "GH Hardy")
        def remove_dots_and_spaces(text):
            return re.sub(r'[\s\.]', '', text.lower())
        
        user_compact = remove_dots_and_spaces(user_answer)
        correct_compact = remove_dots_and_spaces(correct_answer)
        
        if user_compact == correct_compact:
            return True, 0.9, "initials"
        
        # Check for common name variations
        if self.check_name_variations(user_answer, correct_answer):
            return True, 0.85, "name_variation"
        
        return False, similarity, "no_match"

    def check_name_variations(self, user_answer: str, correct_answer: str) -> bool:
        """Check for common name variations"""
        user_words = self.normalize_text(user_answer).split()
        correct_words = self.normalize_text(correct_answer).split()
        
        # For person names, check if first and last names match (ignoring middle names/initials)
        if len(user_words) >= 2 and len(correct_words) >= 2:
            user_first = user_words[0]
            user_last = user_words[-1]
            correct_first = correct_words[0]
            correct_last = correct_words[-1]
            
            # Check if first and last names match
            if (self.similarity_score(user_first, correct_first) > 0.8 and 
                self.similarity_score(user_last, correct_last) > 0.8):
                return True
        
        return False

    def get_match_explanation(self, match_type: str, confidence: float) -> str:
        """Get human-readable explanation of the match"""
        explanations = {
            "exact": "Perfect match!",
            "contains": "Correct answer found in response!",
            "keywords": f"Key words matched! (Confidence: {confidence:.0%})",
            "fuzzy": f"Close enough! (Similarity: {confidence:.0%})",
            "essential": f"Got the main parts right! (Confidence: {confidence:.0%})",
            "initials": "Correct with different formatting!",
            "name_variation": "Correct name variation!",
            "no_match": "Not quite right"
        }
        return explanations.get(match_type, "Match found!")

# Initialize the answer matcher
answer_matcher = AnswerMatcher()

class TriviaBot:
    def __init__(self, reddit_config: dict, supabase_config: dict):
        # Reddit setup
        self.reddit = praw.Reddit(
            client_id=reddit_config['client_id'],
            client_secret=reddit_config['client_secret'],
            username=reddit_config['username'],
            password=reddit_config['password'],
            user_agent=reddit_config['user_agent']
        )
        
        # Supabase setup
        self.supabase: Client = create_client(
            supabase_config['url'],
            supabase_config['key']
        )
        
        # Bot state
        self.current_question = None
        self.current_answer = None
        self.question_start_time = None
        self.waiting_for_answer = False
        self.user_stats: Dict[str, UserStats] = {}
        self.processed_comments = set()
        self.question_answers = []  # Store all answers for current question
        
        # AI responses for engagement
        self.streak_responses = {
            3: ["Nice streak going! 🔥", "You're on fire! 🎯", "Three in a row! 💪"],
            5: ["Wow! 5 correct answers in a row! Looks like you're enjoying the game @{username}! ☕"],
            7: ["INCREDIBLE! 7 streak! You're absolutely crushing it! 🏆", "Seven straight! Are you even human? 🤖"],
            10: ["LEGENDARY! 10 in a row! Hall of Fame material! 👑", "Perfect 10! Time to go pro! 🎓"]
        }
        
        self.encouragement_responses = [
            "Don't give up! The next one might be easier! 💪",
            "Good try! Learning is part of the fun! 📚",
            "Close one! You're getting better! 🎯",
            "Keep going! Every expert was once a beginner! 🌟"
        ]
        
        self.coffee_responses = [
            "Then buy me a coffee! ☕ (Just kidding... or am I? 😏)",
            "Coffee donations accepted! Just kidding, keep playing! ☕😄",
            "I run on coffee and good vibes! Keep the questions coming! ☕✨"
        ]

    def setup_database(self):
        """Initialize database tables"""
        try:
            # Create users table
            self.supabase.table('trivia_users').select('*').limit(1).execute()
        except:
            # Table doesn't exist, create it
            logger.info("Setting up database tables...")
            # Note: In real implementation, you'd create tables through Supabase dashboard
            # or use SQL migrations
            pass

    def get_trivia_question(self) -> dict:
        """Fetch trivia question from Open Trivia Database"""
        try:
            # Weighted difficulty selection (70% medium, 20% hard, 10% easy)
            difficulty_weights = ['medium'] * 70 + ['hard'] * 20 + ['easy'] * 10
            difficulty = random.choice(difficulty_weights)
            
            url = f"https://opentdb.com/api.php?amount=1&difficulty={difficulty}&type=boolean,multiple"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data['response_code'] == 0:
                question_data = data['results'][0]
                
                # Decode HTML entities
                question = html.unescape(question_data['question'])
                correct_answer = html.unescape(question_data['correct_answer'])
                
                # For boolean questions
                if question_data['type'] == 'boolean':
                    return {
                        'question': question,
                        'correct_answer': correct_answer,
                        'difficulty': difficulty,
                        'category': html.unescape(question_data['category']),
                        'type': 'boolean'
                    }
                else:
                    # For multiple choice, we'll use the correct answer only
                    return {
                        'question': question,
                        'correct_answer': correct_answer,
                        'difficulty': difficulty,
                        'category': html.unescape(question_data['category']),
                        'type': 'open'
                    }
        except Exception as e:
            logger.error(f"Error fetching trivia question: {e}")
            
        # Fallback questions
        fallback_questions = [
            {
                'question': "What is the capital of France?",
                'correct_answer': "Paris",
                'difficulty': 'easy',
                'category': 'Geography',
                'type': 'open'
            },
            {
                'question': "Which planet is known as the Red Planet?",
                'correct_answer': "Mars",
                'difficulty': 'easy',
                'category': 'Science',
                'type': 'open'
            },
            {
                'question': "Who wrote 'Romeo and Juliet'?",
                'correct_answer': "William Shakespeare",
                'difficulty': 'medium',
                'category': 'Literature',
                'type': 'open'
            }
        ]
        return random.choice(fallback_questions)

    def format_question(self, question_data: dict) -> str:
        """Format question for Reddit post"""
        
        question_type_text = ""
        if question_data['type'] == 'boolean':
            question_type_text = "**Type:** True/False"
        else:
            question_type_text = "**Type:** Open Answer"
        
        post_text = f"""🧠 **RAPID TRIVIA!** 🧠

**Category:** {question_data['category']}
**Difficulty:** {question_data['difficulty'].title()}
{question_type_text}

**Question:** {question_data['question']}

---
⚡ **SPEED ROUND:** Type your answer! 
⏱️ **Time limit:** 45 seconds
💡 *Flexible matching - variations of the correct answer are accepted!*

**Examples of accepted formats:**
• Exact: "William Shakespeare"
• Initials: "W. Shakespeare" or "W Shakespeare" 
• Short: "Shakespeare"
• Common variations accepted automatically!

🏆 **Current Leaderboard:**
{self.get_top_players_text()}

---
*Bot by u/YourUsername | Smart Answer Matching 🤖*
"""
        return post_text

    def get_top_players_text(self, limit: int = 3) -> str:
        """Get formatted top players for display"""
        if not self.user_stats:
            return "*No players yet! Be the first!*"
        
        sorted_users = sorted(self.user_stats.values(), 
                            key=lambda x: (x.total_score, x.max_streak), 
                            reverse=True)[:limit]
        
        leaderboard = []
        for i, user in enumerate(sorted_users, 1):
            emoji = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
            leaderboard.append(f"{emoji} u/{user.username}: {user.total_score} points (Max streak: {user.max_streak})")
        
        return "\n".join(leaderboard)

    def get_top_matching_answers(self, correct_answer: str, limit: int = 3) -> str:
        """Get top 3 users with answers most similar to the correct answer"""
        if not self.question_answers:
            return ""
        
        # Calculate similarity scores for all answers
        answer_scores = []
        for answer_data in self.question_answers:
            is_match, confidence, match_type = answer_matcher.is_match(
                answer_data['answer'], 
                correct_answer
            )
            answer_scores.append({
                'username': answer_data['username'],
                'answer': answer_data['answer'],
                'confidence': confidence,
                'match_type': match_type,
                'is_correct': is_match
            })
        
        # Sort by confidence score (highest first)
        answer_scores.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get top 3 matching answers (excluding the winner if they got it exactly right)
        top_matches = []
        for score_data in answer_scores[:limit + 2]:  # Get a few extra in case we need to filter
            if len(top_matches) >= limit:
                break
            # Only include if they have some similarity (confidence > 0.3)
            if score_data['confidence'] > 0.3:
                top_matches.append(score_data)
        
        if not top_matches:
            return ""
        
        # Format the top matching answers
        match_text = "\n**🎯 Top Matching Answers:**\n"
        for i, match in enumerate(top_matches, 1):
            emoji = ["🥈", "🥉", "🏅"][i-1] if i <= 3 else f"{i}."
            explanation = answer_matcher.get_match_explanation(match['match_type'], match['confidence'])
            match_text += f"{emoji} u/{match['username']}: \"{match['answer']}\" - {explanation}\n"
        
        return match_text

    def update_user_stats(self, username: str, correct: bool, response_time: float):
        """Update user statistics"""
        if username not in self.user_stats:
            self.user_stats[username] = UserStats(username=username, last_active=datetime.now())
        
        user = self.user_stats[username]
        user.total_questions += 1
        user.last_active = datetime.now()
        
        if correct:
            points = self.calculate_points(response_time)
            user.total_score += points
            user.correct_answers += 1
            user.streak += 1
            user.max_streak = max(user.max_streak, user.streak)
            user.daily_score += points
            user.weekly_score += points
            user.monthly_score += points
        else:
            user.streak = 0
        
        # Save to database
        self.save_user_to_db(user)

    def calculate_points(self, response_time: float) -> int:
        """Calculate points based on response time (faster = more points) - Rapid Fire Mode"""
        base_points = 10
        if response_time < 5:  # Super fast (under 5 seconds)
            return base_points + 10
        elif response_time < 10:  # Very fast (under 10 seconds)
            return base_points + 7
        elif response_time < 20:  # Fast (under 20 seconds)
            return base_points + 5
        elif response_time < 35:  # Moderate (under 35 seconds)
            return base_points + 3
        else:  # Just made it (35-45 seconds)
            return base_points + 1

    def save_user_to_db(self, user: UserStats):
        """Save user statistics to Supabase"""
        try:
            data = {
                'username': user.username,
                'total_score': user.total_score,
                'correct_answers': user.correct_answers,
                'total_questions': user.total_questions,
                'current_streak': user.streak,
                'max_streak': user.max_streak,
                'last_active': user.last_active.isoformat(),
                'daily_score': user.daily_score,
                'weekly_score': user.weekly_score,
                'monthly_score': user.monthly_score,
                'updated_at': datetime.now().isoformat()
            }
            
            # Upsert user data
            result = self.supabase.table('trivia_users').upsert(data).execute()
            logger.info(f"Saved user {user.username} to database")
            
        except Exception as e:
            logger.error(f"Error saving user to database: {e}")

    def get_ai_response(self, username: str, streak: int, correct: bool, content: str = "") -> Optional[str]:
        """Generate AI response based on context"""
        content_lower = content.lower()
        
        # Check for coffee/enjoyment interaction
        if "yes" in content_lower and any(word in content_lower for word in ["enjoy", "fun", "like", "love"]):
            return random.choice(self.coffee_responses)
        
        if correct:
            # Streak-based responses
            if streak in self.streak_responses:
                response = random.choice(self.streak_responses[streak])
                return response.format(username=username)
            elif streak > 10 and streak % 5 == 0:  # Every 5 after 10
                return f"🔥 {streak} STRAIGHT! u/{username} is unstoppable! 🔥"
        else:
            # Encouragement for wrong answers (occasionally)
            if random.random() < 0.3:  # 30% chance
                return random.choice(self.encouragement_responses)
        
        return None

    def is_correct_answer(self, user_answer: str, correct_answer: str) -> tuple[bool, float, str]:
        """Check if user answer is correct using advanced matching"""
        return answer_matcher.is_match(user_answer, correct_answer)

    def post_question(self, subreddit_name: str):
        """Post a new trivia question"""
        try:
            question_data = self.get_trivia_question()
            post_text = self.format_question(question_data)
            
            subreddit = self.reddit.subreddit(subreddit_name)
            submission = subreddit.submit(
                title=f"⚡ Rapid Trivia #{random.randint(1000, 9999)} - {question_data['category']} ({question_data['difficulty'].title()}) ⏱️",
                selftext=post_text
            )
            
            # Store current question data and reset answer tracking
            self.current_question = submission
            self.current_answer = question_data['correct_answer']
            self.question_start_time = datetime.now()
            self.waiting_for_answer = True
            self.question_answers = []  # Reset answers for new question
            
            logger.info(f"Posted new question: {question_data['question']}")
            
        except Exception as e:
            logger.error(f"Error posting question: {e}")

    def monitor_comments(self):
        """Monitor comments for answers"""
        if not self.current_question or not self.waiting_for_answer:
            return
        
        try:
            self.current_question.comments.replace_more(limit=0)
            
            for comment in self.current_question.comments:
                if comment.id in self.processed_comments:
                    continue
                
                self.processed_comments.add(comment.id)
                
                # Skip bot's own comments
                if comment.author and comment.author.name == self.reddit.user.me().name:
                    continue
                
                username = comment.author.name if comment.author else "Anonymous"
                response_time = (datetime.now() - self.question_start_time).total_seconds()
                
                # Store all answers for later analysis
                self.question_answers.append({
                    'username': username,
                    'answer': comment.body,
                    'response_time': response_time
                })
                
                # Check if answer is correct using advanced matching
                is_correct, confidence, match_type = self.is_correct_answer(
                    comment.body, 
                    self.current_answer
                )
                
                if is_correct:
                    # First correct answer wins!
                    self.update_user_stats(username, True, response_time)
                    user = self.user_stats[username]
                    
                    # Celebrate the winner with match explanation
                    points = self.calculate_points(response_time)
                    match_explanation = answer_matcher.get_match_explanation(match_type, confidence)
                    
                    celebration = f"🎉 **CORRECT!** 🎉\n\n"
                    celebration += f"**Winner:** u/{username}\n"
                    celebration += f"**Your Answer:** {comment.body.strip()}\n"
                    celebration += f"**Correct Answer:** {self.current_answer}\n"
                    celebration += f"**Match Type:** {match_explanation}\n"
                    celebration += f"**Points earned:** {points} (⚡{response_time:.1f}s)\n"
                    celebration += f"**Current streak:** {user.streak} 🔥\n"
                    
                    # Add top 3 matching answers
                    top_matches = self.get_top_matching_answers(self.current_answer)
                    if top_matches:
                        celebration += top_matches
                    
                    celebration += "\n"
                    
                    # Add AI response if applicable
                    ai_response = self.get_ai_response(username, user.streak, True)
                    if ai_response:
                        celebration += f"*{ai_response}*\n\n"
                    
                    celebration += "---\n*⏱️ Next question in 10 seconds!*"
                    
                    comment.reply(celebration)
                    
                    # Mark question as answered
                    self.waiting_for_answer = False
                    logger.info(f"Question answered correctly by {username} in {response_time:.1f}s ({match_type})")
                    break
                    
                else:
                    # Wrong answer - occasionally provide feedback on close answers
                    if confidence > 0.6:  # Close but not quite right
                        feedback_responses = [
                            f"Close! You got {confidence:.0%} of it right. Keep trying! 💪",
                            f"Almost there! {confidence:.0%} similarity to the correct answer. 🎯",
                            f"You're on the right track! {confidence:.0%} match. Don't give up! 🌟"
                        ]
                        if random.random() < 0.4:  # 40% chance to give feedback
                            comment.reply(random.choice(feedback_responses))
                    
                    self.update_user_stats(username, False, response_time)
                    user = self.user_stats[username]
                    
                    # Occasionally respond to wrong answers
                    ai_response = self.get_ai_response(username, user.streak, False, comment.body)
                    if ai_response and random.random() < 0.2:  # 20% chance for regular encouragement
                        comment.reply(ai_response)
                        
        except Exception as e:
            logger.error(f"Error monitoring comments: {e}")

    def reset_daily_stats(self):
        """Reset daily statistics"""
        for user in self.user_stats.values():
            user.daily_score = 0
        logger.info("Reset daily statistics")

    def reset_weekly_stats(self):
        """Reset weekly statistics"""
        for user in self.user_stats.values():
            user.weekly_score = 0
        logger.info("Reset weekly statistics")

    def reset_monthly_stats(self):
        """Reset monthly statistics"""
        for user in self.user_stats.values():
            user.monthly_score = 0
        logger.info("Reset monthly statistics")

    def handle_timeout(self, subreddit_name: str):
        """Handle question timeout (45 seconds passed)"""
        try:
            if self.current_question:
                timeout_message = f"""⏰ **TIME'S UP!** ⏰

**Correct Answer:** {self.current_answer}

*No one got it this time! Don't worry, here comes the next one!*

---
*Next question in 10 seconds...*
"""
                
                # Post timeout comment on current question
                self.current_question.comments.replace_more(limit=0)
                self.current_question.reply(timeout_message)
                
                logger.info(f"Question timed out. Answer was: {self.current_answer}")
            
            # Mark as no longer waiting
            self.waiting_for_answer = False
            
            # Wait 10 seconds then post next question
            time.sleep(10)
            self.post_question(subreddit_name)
            
        except Exception as e:
            logger.error(f"Error handling timeout: {e}")
            # Still post next question even if timeout message fails
            self.waiting_for_answer = False
            time.sleep(10)
            self.post_question(subreddit_name)
        """Reset monthly statistics"""
        for user in self.user_stats.values():
            user.monthly_score = 0
        logger.info("Reset monthly statistics")

    def run(self, subreddit_name: str, rapid_fire_mode: bool = True):
        """Main bot loop with rapid-fire quiz mode"""
        logger.info("Starting Reddit Trivia Bot in rapid-fire mode...")
        self.setup_database()
        
        last_daily_reset = datetime.now().date()
        last_weekly_reset = datetime.now().isocalendar()[1]
        last_monthly_reset = datetime.now().month
        
        # Post first question immediately
        self.post_question(subreddit_name)
        
        while True:
            try:
                current_date = datetime.now()
                
                # Check for daily reset
                if current_date.date() > last_daily_reset:
                    self.reset_daily_stats()
                    last_daily_reset = current_date.date()
                
                # Check for weekly reset
                if current_date.isocalendar()[1] != last_weekly_reset:
                    self.reset_weekly_stats()
                    last_weekly_reset = current_date.isocalendar()[1]
                
                # Check for monthly reset
                if current_date.month != last_monthly_reset:
                    self.reset_monthly_stats()
                    last_monthly_reset = current_date.month
                
                if self.waiting_for_answer and self.current_question:
                    # Check for timeout (45 seconds)
                    time_elapsed = (datetime.now() - self.question_start_time).total_seconds()
                    
                    if time_elapsed >= 45:
                        # Time's up! Post timeout message and next question
                        self.handle_timeout(subreddit_name)
                    else:
                        # Monitor for answers
                        self.monitor_comments()
                        if not self.waiting_for_answer:  # Someone answered correctly
                            # Wait 10 seconds then post next question
                            logger.info("Answer found! Waiting 10 seconds for next question...")
                            time.sleep(10)
                            self.post_question(subreddit_name)
                
                time.sleep(2)  # Check every 2 seconds for faster response
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(30)  # Wait 30 seconds before retrying

# Configuration
def main():
    reddit_config = {
        'client_id': os.environ.get('REDDIT_CLIENT_ID', 'YOUR_CLIENT_ID'),
        'client_secret': os.environ.get('REDDIT_CLIENT_SECRET', 'YOUR_CLIENT_SECRET'),
        'username': os.environ.get('REDDIT_USERNAME', 'YOUR_BOT_USERNAME'),
        'password': os.environ.get('REDDIT_PASSWORD', 'YOUR_BOT_PASSWORD'),
        'user_agent': 'RapidTriviaBot v1.0 by /u/YourUsername'
    }
    
    supabase_config = {
        'url': os.environ.get('SUPABASE_URL', 'YOUR_SUPABASE_URL'),
        'key': os.environ.get('SUPABASE_KEY', 'YOUR_SUPABASE_ANON_KEY')
    }
    
    subreddit_name = os.environ.get('SUBREDDIT_NAME', 'test_trivia')
    
    bot = TriviaBot(reddit_config, supabase_config)
    bot.run(subreddit_name, rapid_fire_mode=True)  # Rapid fire mode enabled!

if __name__ == "__main__":
    main()
