import requests
import os
import random
import tiktoken
from datetime import datetime
from config import OPENROUTER_API_KEY, OPENROUTER_API_KEY_2, MODEL, SUM_MODEL, RATE_MODEL

API_URL = "https://openrouter.ai/api/v1/chat/completions"

def count_tokens(text, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text or ""))

def query_openrouter(prompt=None, model=MODEL, context_messages=None, system_prompt=None, t="bot"):
    def make_request(api_key):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context_messages:
            messages.extend(context_messages)
        if prompt:
            messages.append({"role": "user", "content": prompt})

        prompt_tokens = count_tokens(prompt, model=model)
        if t == "rate":
            max_output = 2560
        elif t == "sum":
            max_output = (prompt_tokens + 1) // 2
        else:
            # random_offset = random.randint(-200, 200)
            # max_output = int(max(1, min(abs(prompt_tokens + random_offset), 4096)))
            max_output = 2560
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_output,
        }

        response = requests.post(API_URL, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        return response.json()

    try:
        result = make_request(OPENROUTER_API_KEY)
    except Exception as e:
        print(f"[Ошибка API] {e}")
        try:
            result = make_request(OPENROUTER_API_KEY_2)
        except Exception as e2:
            raise Exception(f"Оба ключа не сработали: {e} | {e2}")
        
    if 'error' in result:
        error_msg = result['error'].get('message', 'Unknown API error')
        raise Exception(f"OpenRouter error: {error_msg}")

    if 'choices' not in result:
        raise Exception(f"Ответ OpenRouter не содержит 'choices': {result}")

    return result['choices'][0]['message']['content']


def summarize_message(message, model=SUM_MODEL):
    prompt = f"""Ты — профессиональный компрессор текста. Сожми сообщение до 1-2 предложений, сохраняя ключевые факты и эмоциональную насыщенность.
Без вводных фраз и пояснений. От первого лица.

Сообщение:
{message}

Сжатое сообщение:"""

    result = query_openrouter(
        prompt=prompt,
        model=model,
        t="sum"
    )
    return result


def compress_to_long_term(message, date, model=SUM_MODEL):
    is_important, _ = is_important_fact(message, date)
    if not is_important:
        return None
    
    prompt = f"""Ты — эксперт по сжатию информации для долгосрочной памяти. Извлеки из сообщения ТОЛЬКО ОДИН САМЫЙ ВАЖНЫЙ ФАКТ по следующим правилам:
1. Исключи все обращения к собеседнику
2. Удали любые реакции на слова собеседника
3. Оставь только ключевой факт
4. Формат: краткое утверждение (1 предложение) без пояснений

Сообщение: {message}

Извлеченный факт:"""

    compressed_fact = query_openrouter(
        prompt=prompt,
        model=model,
        t="sum"
    ).strip()
    
    if compressed_fact.startswith(('"', "'")) and compressed_fact.endswith(('"', "'")):
        compressed_fact = compressed_fact[1:-1]
    
    return compressed_fact


def is_important_fact(fact, date):
    prompt = f"""You are an expert in assessing information importance. Answer ONLY with a whole number from 0 to 10. No explanation.

# Time-Based Importance Scale
[ETERNALLY RELEVANT]
10 = Lifetime goals, dreams, values
9 = Key events with a lifelong effect

[TEMPORARILY RELEVANT]
8 = Important, but time-sensitive
7 = Current tasks
6 = Contextual facts

[INSIGNIFICANT]
0-5 = Routine, one-off mentions, outdated data

# Assessment Criteria
1. Initial Importance: Determine a base score based on content
2. Time-Based Importance:
• For categories 6-8: For categories 6-8: Reduce by 1 point for every 5 days since the fact, unless there is a specific validity period.
• Today: {datetime.utcnow().strftime("%d.%m %H:%M")}
• Fact from: {date}
3. For 9-10 points, time is NOT taken into account.

Fact: "{fact}"

Answer (only number 0-10):"""

    response = query_openrouter(
        prompt=prompt,
        model=RATE_MODEL,
        t="rate"
    ).strip()

    score = None
    for token in response.split():
        if token.isdigit():
            val = int(token)
            if 0 <= val <= 10:
                score = val
                break
    
    
    if score is not None:
        offset = random.randint(-1, 1)
        final_score = max(0, min(10, score + offset))
        return final_score >= 6, final_score

    return False
