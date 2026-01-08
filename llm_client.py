from typing import Dict, List
from openai import OpenAI

# NASA Expert System Prompt
SYSTEM_PROMPT = """You are a NASA mission expert and historian with deep knowledge of space exploration,
particularly the Apollo program and Space Shuttle missions. You have extensive expertise in:

- Apollo 11: The first Moon landing mission (July 1969)
- Apollo 13: The famous "successful failure" mission with the oxygen tank explosion (April 1970)
- Space Shuttle Challenger: The tragic disaster during launch (January 1986)

When answering questions:
1. Base your responses primarily on the provided context from NASA documents
2. Cite specific sources when available (e.g., "According to the technical transcript...")
3. If the context doesn't contain enough information, acknowledge this clearly
4. Provide accurate technical details while making them accessible
5. When discussing tragedies, maintain a respectful and factual tone
6. If you're unsure about something, say so rather than making up information

You have access to mission transcripts, technical documents, and official NASA records.
Always prioritize accuracy and cite your sources from the provided context."""


def generate_response(openai_key: str, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """
    Generate response using OpenAI with context

    Args:
        openai_key: OpenAI API key
        user_message: The user's question
        context: Retrieved context from NASA documents
        conversation_history: Previous conversation messages
        model: OpenAI model to use

    Returns:
        Generated response string
    """

    # Create OpenAI Client
    client = OpenAI(api_key=openai_key)

    # Build messages list starting with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (limit to last 10 exchanges to manage context length)
    history_limit = 20  # 10 exchanges = 20 messages (user + assistant)
    recent_history = conversation_history[-history_limit:] if len(conversation_history) > history_limit else conversation_history

    for msg in recent_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Create the user message with context
    if context:
        user_content = f"""Based on the following NASA mission documents, please answer my question.

{context}

Question: {user_message}

Please provide a detailed answer based on the context above. If the context doesn't contain relevant information, say so clearly."""
    else:
        user_content = f"""Question: {user_message}

Note: No specific NASA documents were retrieved for this query. Please answer based on your general knowledge, but clearly indicate when you're not referencing specific mission documents."""

    messages.append({"role": "user", "content": user_content})

    # Send request to OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )

        # Return response
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {str(e)}"
