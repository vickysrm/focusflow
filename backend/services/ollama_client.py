import os
from typing import Iterable, List, Optional

try:
    import groq
except ImportError:
    groq = None

try:
    import ollama
except ImportError:
    ollama = None

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama-3.1-8b-instant')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

SYSTEM_PROMPT = (
    'You are a helpful meeting assistant. Use plain language, keep your response concise, and answer using only the provided meeting content.'
)


def _normalize_response(response: object) -> str:
    if response is None:
        return ''

    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        message = response.get('message')
        if isinstance(message, dict):
            return str(message.get('content', '') or '').strip()
        if isinstance(message, str):
            return message.strip()
        if 'choices' in response:
            choices = response.get('choices')
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    return str(first.get('message', {}).get('content', '') or '').strip()
        return str(response).strip()

    if hasattr(response, 'get'):
        try:
            message = response.get('message')
            if isinstance(message, dict):
                return str(message.get('content', '') or '').strip()
        except Exception:
            pass

    if hasattr(response, 'message'):
        message = getattr(response, 'message')
        if isinstance(message, dict):
            return str(message.get('content', '') or '').strip()
        if isinstance(message, str):
            return message.strip()

    return str(response).strip()


def _truncate_text(text: str, max_chars: int = 3000) -> str:
    return text if len(text) <= max_chars else text[-max_chars:]


def _get_groq_client():
    if not GROQ_API_KEY or groq is None:
        return None
    return groq.Client(api_key=GROQ_API_KEY)


def _call_groq(prompt: str, documents: Optional[List[str]] = None) -> str:
    client = _get_groq_client()
    if client is None:
        raise RuntimeError('GROQ client is unavailable')

    if documents:
        context = '\n\n'.join(documents)
        prompt = f"{prompt}\n\nMeeting context:\n{context}"

    response = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        model=GROQ_MODEL,
        temperature=0.2,
        max_completion_tokens=512,
    )
    return _normalize_response(response)


def _call_ollama(prompt: str, documents: Optional[List[str]] = None) -> str:
    if ollama is None:
        raise RuntimeError('Ollama client is unavailable')

    if documents:
        context = '\n\n'.join(documents)
        prompt = f"{prompt}\n\nMeeting context:\n{context}"

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        stream=False,
    )
    return _normalize_response(response)


def _call_llm(prompt: str, documents: Optional[List[str]] = None) -> str:
    if GROQ_API_KEY and groq is not None:
        try:
            return _call_groq(prompt, documents=documents)
        except Exception:
            pass

    if ollama is not None:
        return _call_ollama(prompt, documents=documents)

    raise RuntimeError('No available LLM backend. Install groq or ollama and configure the environment.')


def summarize_segment(text: str) -> str:
    if not text:
        return ''
    text = _truncate_text(text, max_chars=3000)
    prompt = (
        'Summarize the following meeting transcript segment into a concise, plain-language summary. '
        'Focus on the key ideas, decisions, and action items where possible. Do not invent information.\n\n'
        f'Transcript segment:\n{text}'
    )
    return _call_llm(prompt)


def answer_question(question: str, context_chunks: Iterable[str]) -> str:
    if not question:
        return 'No question provided.'
    context_text = '\n'.join(context_chunks) if context_chunks else ''
    if len(context_text) > 3000:
        context_text = _truncate_text(context_text, max_chars=3000)

    prompt = (
        'You are a meeting assistant. Answer the user question using only the provided meeting context. '
        'If the answer is not contained in the context, say that you do not know. Keep the answer short and factual.\n\n'
        f'Meeting context:\n{context_text}\n\nQuestion: {question}\nAnswer:'
    )
    return _call_llm(prompt)


def generate_digest(full_text: str, action_items: Optional[List[str]] = None, decisions: Optional[List[str]] = None) -> str:
    if not full_text:
        return 'No transcript available.'

    action_items = action_items or []
    decisions = decisions or []
    transcript = _truncate_text(full_text, max_chars=4000)

    prompt = (
        'Create a concise meeting digest with a short summary, key takeaways, and a brief list of action items and decisions. '
        'Keep the tone professional and easy to read.\n\n'
        f'Transcript:\n{transcript}\n\n'
        f'Action items:\n{chr(10).join(action_items) if action_items else "None"}\n\n'
        f'Decisions:\n{chr(10).join(decisions) if decisions else "None"}\n\n'
        'Digest:'
    )
    return _call_llm(prompt)
