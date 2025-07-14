"""Prompt templates for the Smart Research Assistant."""

# Summary generation prompt
SUMMARY_PROMPT = """Please provide a concise summary of the following text in {max_words} words or less. 
Focus on the main ideas, key points, and conclusions. Do not include any information not present in the text.

Text to summarize:
{text}

Summary:"""

# Question answering prompt
QA_PROMPT = """You are a helpful research assistant. Answer the question based on the provided context.
If the context doesn't contain enough information, say that you don't know.

Context:
{context}

Question: {question}

Provide a clear, concise answer. At the end, include a section titled "Sources:" 
that lists which chunks support your answer, like "Chunk 1, Chunk 3".

Answer:"""

# Question generation prompt
QUESTION_GENERATION_PROMPT = """Generate {num_questions} comprehension questions based on the following text. 
For each question, provide a clear, concise ideal answer. Format your response as:

Question 1: [question]
Ideal Answer 1: [answer]

Question 2: [question]
Ideal Answer 2: [answer]

...and so on.

Focus on important concepts, relationships, and implications in the text. 
Include at least one question that requires synthesis of multiple ideas.

Text:
{text}"""

# Answer evaluation prompt
ANSWER_EVALUATION_PROMPT = """Evaluate the following answer to the question. 
Consider if it captures the main points from the ideal answer, even if worded differently.

Question: {question}

Ideal Answer: {ideal_answer}

User's Answer: {user_answer}

Provide:
1. A brief evaluation (correct/partially correct/incorrect)
2. Specific feedback on what's good and what's missing
3. A short explanation of the correct answer

Format your response as:
Evaluation: [correct/partially correct/incorrect]
Feedback: [your feedback]
Explanation: [brief explanation]"""
