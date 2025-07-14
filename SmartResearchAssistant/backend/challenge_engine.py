from typing import List, Dict, Any, Optional, Tuple
import re
import logging
from .llm import LLMClient

logger = logging.getLogger(__name__)

class ChallengeEngine:
    """Handles generation and evaluation of challenge questions using LLMs."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the Challenge Engine.
        
        Args:
            llm_client: Pre-initialized LLM client (optional)
        """
        self.llm = llm_client or LLMClient()
        self.generated_questions = []
    
    def _parse_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        """Parse questions and answers from the model's response."""
        qa_pairs = []
        current_qa = {}
        
        # Split by lines and process each line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            # Check for question pattern
            q_match = re.match(r'^\s*Question\s*\d*[:.]?\s*(.*)$', line, re.IGNORECASE)
            if q_match:
                if current_qa:
                    qa_pairs.append(current_qa)
                current_qa = {'question': q_match.group(1).strip(), 'ideal_answer': ''}
                continue
                
            # Check for answer pattern
            a_match = re.match(r'^\s*(?:Answer|Ideal Answer|Ideal_Answer|IdealAnswer)\s*\d*[:.]?\s*(.*)$', 
                             line, re.IGNORECASE)
            if a_match and current_qa:
                current_qa['ideal_answer'] = a_match.group(1).strip()
        
        # Add the last QA pair if it exists
        if current_qa and current_qa.get('question') and current_qa.get('ideal_answer'):
            qa_pairs.append(current_qa)
            
        return qa_pairs
    
    def generate_questions(self, text: str, num_questions: int = 3) -> List[Dict[str, str]]:
        """
        Generate comprehension questions from the given text.
        
        Args:
            text: The text to generate questions from
            num_questions: Number of questions to generate (1-5)
            
        Returns:
            List of question dictionaries with 'question' and 'ideal_answer' keys
        """
        try:
            # Ensure num_questions is within reasonable bounds
            num_questions = max(1, min(5, num_questions))
            
            prompt = f"""[INST] <<SYS>>
You are a helpful teaching assistant that creates high-quality comprehension questions.
Your task is to generate {num_questions} questions based on the provided text.

Guidelines:
1. Create questions that test understanding of key concepts, relationships, and implications.
2. Include at least one question that requires synthesizing multiple ideas.
3. Make sure each question is clear, specific, and answerable from the text.
4. Provide a concise, accurate answer for each question.
5. Format your response as:

Question 1: [Your first question]
Answer 1: [The ideal answer]

Question 2: [Your second question]
Answer 2: [The ideal answer]

And so on...
<</SYS>>

Text to generate questions from:
{text}

Please generate {num_questions} comprehension questions with answers based on the text above.
Focus on important concepts and their relationships.
[/INST]"""
            
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # Parse the response into questions and answers
            qa_pairs = self._parse_qa_pairs(response)
            
            # Format the questions with empty user_answer and feedback
            questions = []
            for i, qa in enumerate(qa_pairs[:num_questions], 1):
                if 'question' in qa and 'ideal_answer' in qa:
                    questions.append({
                        'question': f"{i}. {qa['question'].strip()}",
                        'ideal_answer': qa['ideal_answer'].strip(),
                        'user_answer': '',
                        'feedback': ''
                    })
            
            # Ensure we have at least one question
            if not questions:
                questions = [{
                    'question': 'What are the main points discussed in the document?',
                    'ideal_answer': 'The document covers several key points...',
                    'user_answer': '',
                    'feedback': ''
                }]
            
            # Store the generated questions for later evaluation
            self.generated_questions = questions
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}", exc_info=True)
            # Return a default question if generation fails
            return [{
                'question': 'What are the main points discussed in the document?',
                'ideal_answer': 'The document covers several key points...',
                'user_answer': '',
                'feedback': ''
            }]
    
    def evaluate_answer(self, question_idx: int, user_answer: str) -> Dict[str, Any]:
        """
        Evaluate a user's answer to a generated question using the LLM.
        
        Args:
            question_idx: Index of the question in the generated_questions list
            user_answer: The user's answer to evaluate
            
        Returns:
            Dict containing evaluation results with keys:
            - is_correct (bool): Whether the answer is correct
            - feedback (str): Detailed feedback on the answer
            - ideal_answer (str): The expected ideal answer
        """
        if not self.generated_questions or question_idx >= len(self.generated_questions):
            return {
                'is_correct': False,
                'feedback': 'Invalid question index. Please try again.',
                'ideal_answer': ''
            }
        
        question_data = self.generated_questions[question_idx]
        question_data['user_answer'] = user_answer
        
        try:
            prompt = f"""[INST] <<SYS>>
You are a helpful teaching assistant evaluating a student's answer to a question.

Guidelines for evaluation:
1. Be fair and constructive in your feedback.
2. Consider the meaning and key points, not just exact wording.
3. If the answer is partially correct, explain what's right and what's missing.
4. If the answer is incorrect, provide a clear explanation of the correct answer.
5. Keep feedback concise but helpful.

Format your response as:

EVALUATION: [CORRECT/PARTIALLY CORRECT/INCORRECT]

FEEDBACK: [Your detailed feedback here]

EXPLANATION: [A brief explanation of the correct answer]
<</SYS>>

QUESTION: {question_data['question']}

IDEAL ANSWER: {question_data['ideal_answer']}

STUDENT'S ANSWER: {user_answer}

Please evaluate the student's answer and provide feedback.
[/INST]"""
            
            evaluation = self.llm.generate(
                prompt=prompt,
                max_tokens=512,
                temperature=0.3,  # Lower temperature for more consistent evaluations
                top_p=0.9,
                do_sample=True
            )
            
            # Parse the evaluation
            evaluation = evaluation.strip()
            is_correct = 'CORRECT' in evaluation.upper()
            
            # Extract feedback and explanation
            feedback = ""
            explanation = ""
            
            feedback_match = re.search(r'FEEDBACK[:\s]*(.*?)(?:\n\n|$)', evaluation, re.DOTALL | re.IGNORECASE)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
            
            explanation_match = re.search(r'EXPLANATION[:\s]*(.*?)(?:\n\n|$)', evaluation, re.DOTALL | re.IGNORECASE)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
            
            # If we couldn't parse the feedback, use the whole evaluation
            if not feedback and not explanation:
                feedback = evaluation
            
            # Store the feedback in the question data
            question_data['feedback'] = feedback
            
            return {
                'is_correct': is_correct,
                'feedback': feedback,
                'explanation': explanation,
                'ideal_answer': question_data['ideal_answer']
            }
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}", exc_info=True)
            return {
                'is_correct': False,
                'feedback': 'An error occurred while evaluating your answer. Please try again.',
                'explanation': '',
                'ideal_answer': question_data['ideal_answer']
            }
