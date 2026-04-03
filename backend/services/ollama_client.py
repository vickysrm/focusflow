import ollama
from typing import Optional
import os

# Default to local qwen2:1.5b unless requested otherwise
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:1.5b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    if GROQ_API_KEY:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    else:
        groq_client = None
except ImportError:
    groq_client = None


def _call_llm(prompt: str) -> str:
    """Intelligently route to Cloud Groq API if key exists, otherwise local Ollama."""
    if groq_client:
        # Use cloud Groq for blazing fast deployment inference
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    else:
        # Fallback to local Ollama daemon
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']


def summarize_segment(transcript_segment: str) -> str:
    """Generate a 3-5 bullet plain-language summary of a transcript chunk."""
    prompt = f"""You are a meeting assistant for a neurodiverse professional with ADHD or dyslexia.
Summarize the following meeting segment in 3-5 short bullet points.
Use plain, simple English. No jargon. No filler. Only substance.
Start each bullet with a dash (-).

Transcript segment:
{transcript_segment}"""
    
    return _call_llm(prompt)


def generate_digest(full_transcript: str, action_items: list, decisions: list) -> str:
    """Generate a post-meeting digest."""
    items_str = "\n".join(f"- {i}" for i in action_items) or "None identified"
    decisions_str = "\n".join(f"- {d}" for d in decisions) or "None identified"
    
    prompt = f"""You are a meeting assistant. Generate a clean post-meeting digest.

Action items identified:
{items_str}

Decisions made:
{decisions_str}

Full transcript:
{full_transcript}

Generate a digest with these sections:
1. TL;DR (2-3 sentences max)
2. Key decisions
3. Action items with owners if mentioned
4. Open questions to follow up on

Use plain, simple English suitable for someone with dyslexia or ADHD."""
    
    return _call_llm(prompt)


def answer_question(question: str, context_chunks: list[str]) -> str:
    """Answer a user question using retrieved transcript context."""
    context = "\n---\n".join(context_chunks)
    
    prompt = f"""You are answering a question from a meeting participant.
Answer ONLY based on what has been said in this meeting so far.
If the answer is not clearly present, say: "I don't have enough context from the meeting to answer that yet."
Never fabricate or infer beyond what was explicitly said.
Keep your answer short and plain.

Meeting context:
{context}

Question: {question}"""
    
    return _call_llm(prompt)
