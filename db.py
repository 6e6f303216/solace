import sqlite3
from datetime import datetime
import numpy as np
from config import DB_PATH
from vector_store import vector_store
from embeddings import EmbeddingModel
from contextlib import contextmanager

embedding_model = EmbeddingModel()

@contextmanager
def db_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def init_db():
    """Инициализация базы SQLite (только для авторизации и метаданных)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS auth (
            user_id INTEGER PRIMARY KEY,
            authorized INTEGER DEFAULT 0
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT,
            content TEXT,
            summary TEXT,
            date TEXT,
            rate INTEGER DEFAULT 0
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS context_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT,
            content TEXT,
            summary TEXT,
            timestamp TEXT,
            timestamp_iso TEXT
        )
    ''')

    conn.commit()
    conn.close()

def is_authorized(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT authorized FROM auth WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result and result[0] == 1

def authorize_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO auth (user_id, authorized) VALUES (?, 1)", (user_id,))
    conn.commit()
    conn.close()
def add_to_context(user_id, role, content, summary):
    """Добавление сообщения в текущий контекст + вектор в индекс"""
    now = datetime.utcnow()
    timestamp_iso = now.isoformat()
    readable_stamp = now.strftime("%d.%m %H:%M")
    summary_with_time = f"[{readable_stamp}]{summary}"

    embedding = embedding_model.blob_to_numpy(
        embedding_model.get_embedding(summary)
    )

    vector_store.add(
        embedding,
        {
            "user_id": user_id,
            "role": role,
            "content": content,
            "summary": summary_with_time,
            "timestamp": timestamp_iso
        }
    )

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO context_memory (user_id, role, content, summary, timestamp, timestamp_iso) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, role, content, summary_with_time, readable_stamp, timestamp_iso)
    )
    conn.commit()
    conn.close()

def search_context(user_id, query, threshold=0.3, top_k=10):
    """Поиск релевантных сообщений в контексте по векторному индексу"""
    if not query.strip():
        return []

    query_vec = embedding_model.blob_to_numpy(
        embedding_model.get_embedding(query, is_query=True)
    )

    results = vector_store.search(query_vec, top_k=top_k, threshold=threshold)
    return [
        (meta["role"], meta["summary"], meta["timestamp"])
        for _, meta in results if meta["user_id"] == user_id
    ]

def get_recent_context(user_id, limit=24):
    """Возвращает последние N сообщений из контекста (без поиска по эмбеддингам)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT role, content, summary FROM context_memory 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (user_id, limit))
    rows = c.fetchall()
    conn.close()

    rows = rows[::-1]
    if not rows:
        return []

    for i in reversed(range(len(rows))):
        if rows[i][0] == 'user':
            role, content, summary = rows[i]
            rows[i] = (role, content, content)
            break

    return [
        (role, content if role == 'user' and i == len(rows) - 1 else summary)
        for i, (role, content, summary) in enumerate(rows)
    ]

def clear_context(user_id):
    """Удаляет весь текущий контекст пользователя и очищает векторное хранилище"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM context_memory WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

    # Очистка FAISS по user_id
    vector_store.delete(lambda meta: meta["user_id"] == user_id)

def get_full_context(user_id):
    """Возвращает все сообщения контекста пользователя"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT role, summary, content, timestamp 
        FROM context_memory 
        WHERE user_id = ?
    """, (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def save_to_long_term(user_id, role, content, summary, rate):
    """Сохраняет сообщение в долговременную память и векторное хранилище"""
    emb = embedding_model.blob_to_numpy(
        embedding_model.get_embedding(summary)
    )

    vector_store.add(
        emb,
        {
            "user_id": user_id,
            "role": role,
            "content": content,
            "summary": summary,
            "date": datetime.utcnow().date().isoformat(),
            "rate": rate
        }
    )

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO long_term_memory (user_id, role, content, summary, date, rate) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, role, content, summary, datetime.utcnow().date().isoformat(), rate)
    )
    conn.commit()
    conn.close()

def delete_from_long_term(user_id, summary):
    """Удаляет запись из долговременной памяти"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM long_term_memory WHERE user_id = ? AND summary = ?", (user_id, summary))
    conn.commit()
    conn.close()

    vector_store.delete(lambda meta: meta["user_id"] == user_id and meta["summary"] == summary)

def get_long_term_memory(user_id):
    """Возвращает список всех долговременных воспоминаний"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, summary, date FROM long_term_memory WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_long_term_memory_prune(user_id):
    """Возвращает список воспоминаний для чистки"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT summary, date, rate FROM long_term_memory WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def search_memories(user_id, query, threshold=0.3, top_k=10):
    """Поиск по долговременной памяти с использованием векторного поиска"""
    if not query.strip():
        return []

    query_vec = embedding_model.blob_to_numpy(
        embedding_model.get_embedding(query, is_query=True)
    )

    results = vector_store.search(query_vec, top_k=top_k, threshold=threshold)
    return [
        (meta["role"], meta["summary"])
        for _, meta in results if meta["user_id"] == user_id
    ]
