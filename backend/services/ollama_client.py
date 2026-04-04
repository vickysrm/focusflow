import os
import httpx

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:1.5b")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"


def _call_llm(prompt: str) -> str:
    """Call Groq API directly via HTTP. Falls back to local Ollama if no key."""
    api_key = os.getenv("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")

    if api_key:
        # Direct HTTP call to Groq — no package needed, always works
        response = httpx.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    else:
        # Fallback to local Ollama
        import ollama
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
    
    prompt = f"""You are an intelligent meeting assistant.
First, try to answer the user's question using the provided meeting context.
If the context does not contain the answer, you may answer the question using your general knowledge, but briefly mention that it wasn't explicitly discussed in the meeting yet.
Always organize your answers cleanly using Markdown bullet points or numbered lists if there are multiple parts.
Keep your answer clear, helpful, and plain.

Meeting context:
{context}

Question: {question}"""
    
    return _call_llm(prompt)
