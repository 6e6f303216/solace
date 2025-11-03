import sqlite3
from config import HEADER, DB_PATH, MODEL
from db import (
    add_to_context, 
    search_context, 
    clear_context, 
    delete_from_long_term, 
    save_to_long_term, 
    get_full_context, 
    get_long_term_memory_prune,
    search_memories
)
from ai_client import query_openrouter, summarize_message, is_important_fact, compress_to_long_term

from datetime import datetime

user_chats = set()

def handle_message_as_bot(bot, chat_id, text):
    user_id = chat_id
    user_chats.add(user_id)

    try:
        now = datetime.utcnow()
        readable_stamp = now.strftime("%d.%m %H:%M")

        formatted_context = []

        long_term_results = search_memories(user_id, text)
        if long_term_results:
            memory_lines = []
            for role, summary in long_term_results:
                speaker = "Пользователь" if role == "user" else "Вы"
                memory_lines.append(f"- {speaker}: {summary}")
            memory_block = "\n".join(memory_lines)
            formatted_context.append({
                "role": "user",
                "content": f"Релевантные воспоминания из прошлых разговоров (используйте только если уместно):\n{memory_block}"
            })

        context_results = search_context(user_id, text)
        if context_results:
            context_results.sort(key=lambda x: datetime.fromisoformat(x[2]))
            context_lines = []
            for role, summary, _ in context_results:
                speaker = "Пользователь" if role == "user" else "Вы"
                context_lines.append(f"- {speaker}: {summary}")
            context_block = "\n".join(context_lines)
            formatted_context.append({
                "role": "user",
                "content": f"Релевантные реплики из текущего разговора:\n{context_block}"
            })

        prompt = f"Пользователь написал [{readable_stamp}]: {text}\nВы отвечаете без указания времени:"

        reply = query_openrouter(
            prompt=prompt,
            context_messages=formatted_context,
            system_prompt=HEADER
        )

        if len(text.split()) < 12:
            summarized = text
        else:
            summarized = summarize_message(text)
        add_to_context(user_id, "user", text, summarized)

        bot_summary = summarize_message(reply)
        add_to_context(user_id, "assistant", reply, bot_summary)
        bot.send_message(chat_id, reply)

    except Exception as e:
        bot.send_message(chat_id, "Произошла ошибка при обработке ответа.")
        print(f"[Ошибка OpenRouter]: {e}")

def offload_context_to_long_term():
    for user_id in user_chats:
        rows = get_full_context(user_id)
        for role, summary, content, timestamp in rows:
            try:
                dt = datetime.fromisoformat(timestamp)
                date_str = dt.date().isoformat()
            except Exception:
                date_str = datetime.utcnow().date().isoformat()
            is_important, rate = is_important_fact(summary, date_str)
            if is_important:
                compressed = compress_to_long_term(summary, date_str)
                save_to_long_term(user_id, role, content, compressed, rate)
        clear_context(user_id)
        prune_long_term_memory(user_id)

def prune_long_term_memory(user_id):
    for summary, date, rate in get_long_term_memory_prune(user_id):
        if rate >= 9:
            continue

        is_important, new_rate = is_important_fact(summary, date)
        if not is_important:
            delete_from_long_term(user_id, summary)
        else:
            if new_rate != rate:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute(
                    "UPDATE long_term_memory SET rate = ? WHERE user_id = ? AND summary = ?",
                    (new_rate, user_id, summary)
                )
                conn.commit()
                conn.close()

def get_user_chats():
    return user_chats