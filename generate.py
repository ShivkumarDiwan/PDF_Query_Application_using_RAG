import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key) if api_key else None

def generate_response(retrieved_texts, query, max_tokens=150):
    """
    Generates a response based on the retrieved texts and query.

    Args:
    retrieved_texts (list): List of retrieved text strings.
    query (str): Query string.
    max_tokens (int): Maximum number of tokens for the response.

    Returns:
    str: Generated response.
    """
    if not client:
        return "Error: OPENAI_API_KEY not found. Please set your OPENAI_API_KEY in the .env file."

    context = "\n".join(retrieved_texts)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.5,
    )
    return response.choices[0].message.content

