import json
import sqlite3
from datetime import datetime
from ai_client import query_openrouter
from config import DEEP_MODEL

def init_tests_db():
    conn = sqlite3.connect('chat_sessions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            test_name TEXT NOT NULL,
            current_question INTEGER DEFAULT 0,
            answers TEXT DEFAULT '[]',
            is_completed BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            test_name TEXT NOT NULL,
            answers TEXT NOT NULL,
            analysis TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

init_tests_db()

TESTS = {
    "emotional_intelligence": {
        "name": "Тест на эмоциональный интеллект",
        "description": "Помогает понять, насколько хорошо вы распознаете и управляете эмоциями",
        "questions": [
            {
                "question": "Когда кто-то рассказывает о своих проблемах, я обычно:",
                "options": [
                    "Сразу предлагаю решения",
                    "Сначала стараюсь понять их чувства",
                    "Слушаю, но не всегда знаю, что сказать",
                    "Чувствую дискомфорт и стараюсь сменить тему"
                ]
            },
            {
                "question": "В стрессовой ситуации я:",
                "options": [
                    "Сохраняю спокойствие и ищу решение",
                    "Чувствую тревогу, но стараюсь взять себя в руки",
                    "Легко выхожу из себя",
                    "Стараюсь избегать таких ситуаций"
                ]
            },
            {
                "question": "Когда я вижу, что коллега расстроен:",
                "options": [
                    "Спрашиваю, что случилось и предлагаю помощь",
                    "Замечаю, но не вмешиваюсь",
                    "Чувствую себя неловко и избегаю контакта",
                    "Не всегда замечаю изменения в настроении других"
                ]
            },
            {
                "question": "Мои эмоции обычно:",
                "options": [
                    "Я их хорошо понимаю и могу объяснить",
                    "Иногда бывают запутанными",
                    "Часто берут верх над разумом",
                    "Стараюсь не показывать их окружающим"
                ]
            },
            {
                "question": "При конфликте я склонен(на):",
                "options": [
                    "Искать компромисс",
                    "Анализировать чувства всех участников",
                    "Защищать свою позицию",
                    "Избегать прямого конфликта"
                ]
            }
        ]
    },
    "self_awareness": {
        "name": "Тест на самопознание",
        "description": "Помогает лучше понять свои ценности, сильные и слабые стороны",
        "questions": [
            {
                "question": "Я хорошо понимаю, что для меня действительно важно в жизни:",
                "options": [
                    "Да, у меня четкая система ценностей",
                    "В основном да, но иногда сомневаюсь",
                    "Периодически задумываюсь об этом",
                    "Часто чувствую неопределенность в этом вопросе"
                ]
            },
            {
                "question": "Когда я принимаю важные решения:",
                "options": [
                    "Руководствуюсь своими внутренними принципами",
                    "Советуюсь с другими, но окончательное решение принимаю сам(а)",
                    "Часто полагаюсь на мнение окружающих",
                    "Испытываю трудности с принятием решений"
                ]
            },
            {
                "question": "Мои сильные стороны:",
                "options": [
                    "Я их хорошо знаю и использую",
                    "Знаю, но не всегда применяю эффективно",
                    "Иногда сомневаюсь в своих способностях",
                    "Затрудняюсь назвать свои сильные стороны"
                ]
            },
            {
                "question": "О своих слабостях я:",
                "options": [
                    "Знаю и работаю над ними",
                    "Признаю, но не всегда стараюсь исправить",
                    "Стараюсь не думать о них",
                    "Не люблю говорить о своих слабостях"
                ]
            },
            {
                "question": "Рефлексия и самоанализ для меня:",
                "options": [
                    "Регулярная и важная практика",
                    "Периодическое занятие",
                    "Случается редко, обычно в кризисные моменты",
                    "Дается с трудом, избегаю этого"
                ]
            }
        ]
    },
    "stress_resistance": {
        "name": "Тест на стрессоустойчивость", 
        "description": "Оценивает вашу способность справляться со стрессовыми ситуациями",
        "questions": [
            {
                "question": "В непредвиденных ситуациях я обычно:",
                "options": [
                    "Быстро адаптируюсь и действую",
                    "Немного волнуюсь, но справляюсь",
                    "Сильно нервничаю",
                    "Теряюсь и не знаю что делать"
                ]
            },
            {
                "question": "Когда на меня оказывают давление:",
                "options": [
                    "Сохраняю самообладание",
                    "Могу работать, но чувствую напряжение", 
                    "Становлюсь раздражительным(ой)",
                    "Чувствую себя подавленным(ой)"
                ]
            },
        ]
    }
}

class TestManager:
    def __init__(self):
        self.active_sessions = {}
    
    def start_test(self, chat_id, test_name):
        """Начинает новый тест"""
        if test_name not in TESTS:
            return None
        
        conn = sqlite3.connect('chat_sessions.db')
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE test_sessions SET is_completed = TRUE WHERE chat_id = ? AND is_completed = FALSE',
            (chat_id,)
        )

        cursor.execute(
            'INSERT INTO test_sessions (chat_id, test_name) VALUES (?, ?)',
            (chat_id, test_name)
        )
        session_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def get_current_question(self, chat_id):
        """Получает текущий вопрос для пользователя"""
        conn = sqlite3.connect('chat_sessions.db')
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT ts.id, ts.test_name, ts.current_question, ts.answers, ts.is_completed 
               FROM test_sessions ts 
               WHERE ts.chat_id = ? AND ts.is_completed = FALSE 
               ORDER BY ts.id DESC LIMIT 1''',
            (chat_id,)
        )
        
        session = cursor.fetchone()
        conn.close()
        
        if not session:
            return None
        
        session_id, test_name, current_question, answers_json, is_completed = session
        test = TESTS[test_name]
        questions = test['questions']
        
        if current_question >= len(questions):
            return None
        
        current_q_data = questions[current_question]
        return {
            'session_id': session_id,
            'test_name': test_name,
            'test_title': test['name'],
            'question_number': current_question + 1,
            'total_questions': len(questions),
            'question': current_q_data['question'],
            'options': current_q_data['options']
        }
    
    def save_answer(self, session_id, answer_index):
        """Сохраняет ответ и переходит к следующему вопросу"""
        conn = sqlite3.connect('chat_sessions.db')
        cursor = conn.cursor()
        
        # Получаем текущее состояние
        cursor.execute(
            'SELECT chat_id, test_name, current_question, answers FROM test_sessions WHERE id = ?',
            (session_id,)
        )
        session = cursor.fetchone()
        
        if not session:
            conn.close()
            return False
        
        chat_id, test_name, current_question, answers_json = session
        answers = json.loads(answers_json)
        
        answers.append({
            'question_index': current_question,
            'answer_index': answer_index,
            'timestamp': datetime.now().isoformat()
        })
        
        test_questions = TESTS[test_name]['questions']
        next_question = current_question + 1
        is_completed = next_question >= len(test_questions)
        
        cursor.execute(
            '''UPDATE test_sessions 
               SET current_question = ?, answers = ?, is_completed = ?
               WHERE id = ?''',
            (next_question, json.dumps(answers), is_completed, session_id)
        )
        
        conn.commit()
        conn.close()
        
        if is_completed:
            self._complete_test(session_id, chat_id, test_name, answers)
        
        return True, is_completed
    
    def _complete_test(self, session_id, chat_id, test_name, answers):
        """Завершает тест и запускает анализ"""
        test = TESTS[test_name]
        questions = test['questions']
        
        analysis_data = {
            'test_name': test['name'],
            'questions': [],
            'answers': []
        }
        
        for answer in answers:
            q_index = answer['question_index']
            a_index = answer['answer_index']
            
            analysis_data['questions'].append(questions[q_index]['question'])
            analysis_data['answers'].append(questions[q_index]['options'][a_index])
        
        analysis = self._analyze_with_ai(analysis_data)
        
        conn = sqlite3.connect('chat_sessions.db')
        cursor = conn.cursor()
        
        cursor.execute(
            '''INSERT INTO test_results (chat_id, test_name, answers, analysis)
               VALUES (?, ?, ?, ?)''',
            (chat_id, test_name, json.dumps(analysis_data), analysis)
        )
        
        conn.commit()
        conn.close()
        
        return analysis
    
    def _analyze_with_ai(self, analysis_data):
        """Анализирует ответы с помощью DeepSeek R1"""
        prompt = f"""
        Проанализируй ответы пользователя на тест "{analysis_data['test_name']}" и дай развернутую обратную связь.
        
        Вопросы и ответы:
        {json.dumps(analysis_data, ensure_ascii=False, indent=2)}
        
        Проанализируй:
        1. Общие паттерны в ответах
        2. Сильные стороны, которые проявляются в ответах
        3. Возможные зоны роста
        4. Рекомендации для дальнейшего развития
        5. Инсайты о личности пользователя
        
        Будь внимательным, эмпатичным и поддерживающим психологом. Давай глубокий, но понятный анализ.
        """
        
        try:
            response = query_openrouter(
                prompt=prompt,
                model=DEEP_MODEL,
                system_prompt="Ты - опытный психолог, который помогает людям лучше понять себя через психологические тесты. Ты анализируешь ответы и даешь глубокую, поддерживающую обратную связь."
            )
            return response
        except Exception as e:
            return f"Произошла ошибка при анализе: {str(e)}"
    
    def get_test_result(self, chat_id, test_name):
        """Получает результат последнего теста"""
        conn = sqlite3.connect('chat_sessions.db')
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT analysis, created_at FROM test_results 
               WHERE chat_id = ? AND test_name = ? 
               ORDER BY created_at DESC LIMIT 1''',
            (chat_id, test_name)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'analysis': result[0],
                'created_at': result[1]
            }
        return None

test_manager = TestManager()