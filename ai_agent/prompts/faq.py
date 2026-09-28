"""Prompt templates for FAQ and follow-up question generation.

Single file for all FAQ-related prompts (initial FAQs + conversation follow-ups),
matching the convention of one prompt file per workflow module.
"""

FAQ_SYSTEM_PROMPT = (
    "You generate FAQ questions that a user would likely ask an AI agent.\n"
    "Rules:\n"
    "- Generate exactly {count} questions.\n"
    "- Each question should be concise (under 15 words).\n"
    "- Questions should showcase the agent's capabilities.\n"
    "- Questions must be diverse and cover different capabilities.\n"
    "- Return ONLY a JSON array of strings, no other text.\n"
    '- Example: ["How do I create a report?", "What data can you access?"]\n'
)

FAQ_USER_PROMPT = "Generate {count} FAQ questions for this AI agent:\n\n{context}"

FAQ_FROM_HISTORY_SYSTEM_PROMPT = (
    "You generate FAQ questions based on the most common topics users ask an AI agent about.\n"
    "Rules:\n"
    "- Generate exactly {count} questions.\n"
    "- Each question should be concise (under 15 words).\n"
    "- Base questions on the actual user conversation samples provided.\n"
    "- Return ONLY a JSON array of strings, no other text.\n"
    '- Example: ["Show me today\'s sales report", "How many active users?"]\n'
)

FAQ_FROM_HISTORY_USER_PROMPT = (
    "Agent: {agent_name}\n"
    "Recent user questions (sample):\n{user_questions}{prompt_snippet}\n\n"
    "Generate {count} FAQ questions that represent the most common topics."
)

FOLLOWUP_SYSTEM_PROMPT = (
    "You generate natural follow-up questions that a user might ask after their previous question.\n"
    "Rules:\n"
    "- Generate exactly {count} follow-up questions.\n"
    "- Each question must be concise (under 15 words).\n"
    "- Questions must be conversational, relevant, and helpful follow-ups to the user's last question.\n"
    "- Consider the recent conversation history to provide contextual follow-ups.\n"
    "- Return ONLY a JSON array of strings, no other text.\n"
    '- Example: ["Can you summarize that in a table?", "What are the next steps?"]\n'
)

FOLLOWUP_USER_PROMPT = (
    "Agent: {agent_name}\nRecent Conversation:\n{conversation_history}\n\n"
    "User's previous question: {main_question}{prompt_snippet}\n\n"
    "Generate {count} follow-up questions."
)
