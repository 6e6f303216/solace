import telebot
import random
import os
import time
import whisper
import torch
import tempfile
import threading
import traceback
from config import TELEGRAM_TOKEN
from db import init_db
from messages import handle_message_as_bot, get_user_chats
from tests import TESTS, test_manager

init_db()

bot = telebot.TeleBot(TELEGRAM_TOKEN)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Инициализация Whisper model на {device}...")
torch.set_num_threads(os.cpu_count())
WHISPER_MODEL = whisper.load_model("base", device=device)
print("Whisper model загружена")

def transcribe_audio(audio_path):
    """Транскрибируем аудио в текст с помощью Whisper"""
    try:
        fp16 = (device == "cuda")
        result = WHISPER_MODEL.transcribe(
            audio_path,
            fp16=fp16,
            language='ru'
        )

        
        if "text" in result and isinstance(result["text"], str):
            text = result["text"].strip()
            return text
        else:
            print("Ошибка: результат транскрипции не содержит текст")
            return None
    except Exception as e:
        print(f"Ошибка транскрибации: {e}")
        traceback.print_exc()
        return None

@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    try:
        user_id = message.from_user.id
        chat_id = message.chat.id
            
        get_user_chats().add(chat_id)

        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio:
            temp_audio.write(downloaded_file)
            temp_path = temp_audio.name
        
        text = transcribe_audio(temp_path)
        os.unlink(temp_path)
        
        if not text or not isinstance(text, str) or not text.strip():
            bot.reply_to(message, "Не удалось распознать речь или сообщение пустое")
            return

        handle_message_as_bot(bot, chat_id, text)
        
    except Exception as e:
        print(f"Ошибка обработки голоса: {e}")
        traceback.print_exc()
        try:
            bot.reply_to(message, "Произошла ошибка при обработке голосового сообщения")
        except:
            pass

@bot.message_handler(commands=['tests'])
def show_tests(message):
    """Показывает доступные тесты"""
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    
    for test_key, test_data in TESTS.items():
        keyboard.add(telebot.types.KeyboardButton(f"🧠 {test_data['name']}"))
    
    keyboard.add(telebot.types.KeyboardButton("❌ Отмена"))
    
    bot.send_message(
        message.chat.id,
        "Выберите тест для прохождения:\n\n" +
        "\n".join([f"• {test['name']}: {test['description']}" for test in TESTS.values()]),
        reply_markup=keyboard
    )

@bot.message_handler(func=lambda message: message.text.startswith('🧠'))
def start_test(message):
    """Начинает выбранный тест"""
    test_name = None
    for test_key, test_data in TESTS.items():
        if message.text == f"🧠 {test_data['name']}":
            test_name = test_key
            break
    
    if not test_name:
        bot.send_message(message.chat.id, "Тест не найден")
        return
    
    # Начинаем тест
    test_manager.start_test(message.chat.id, test_name)
    
    # Показываем первый вопрос
    show_next_question(message.chat.id)

def show_next_question(chat_id):
    """Показывает следующий вопрос теста"""
    question_data = test_manager.get_current_question(chat_id)
    
    if not question_data:
        bot.send_message(chat_id, "Тест завершен или не найден")
        return
    
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    
    for i, option in enumerate(question_data['options']):
        keyboard.add(telebot.types.KeyboardButton(f"{i+1}. {option}"))
    
    keyboard.add(telebot.types.KeyboardButton("❌ Прервать тест"))
    
    question_text = (
        f"📊 {question_data['test_title']}\n"
        f"Вопрос {question_data['question_number']}/{question_data['total_questions']}\n\n"
        f"{question_data['question']}"
    )
    
    bot.send_message(chat_id, question_text, reply_markup=keyboard)

@bot.message_handler(func=lambda message: message.text.replace('.', '').isdigit() and 1 <= int(message.text.replace('.', '')) <= 10)
def handle_test_answer(message):
    """Обрабатывает ответ на вопрос теста"""
    chat_id = message.chat.id
    
    # Проверяем, есть ли активный тест
    question_data = test_manager.get_current_question(chat_id)
    if not question_data:
        return
    
    try:
        answer_index = int(message.text.split('.')[0]) - 1
        
        # Сохраняем ответ
        success, is_completed = test_manager.save_answer(question_data['session_id'], answer_index)
        
        if not success:
            bot.send_message(chat_id, "Ошибка при сохранении ответа")
            return
        
        if is_completed:
            # Тест завершен, показываем результаты
            show_test_results(chat_id, question_data['test_name'])
        else:
            # Показываем следующий вопрос
            show_next_question(chat_id)
            
    except (ValueError, IndexError):
        bot.send_message(chat_id, "Пожалуйста, выберите вариант ответа из предложенных")

@bot.message_handler(func=lambda message: message.text == "❌ Прервать тест")
def cancel_test(message):
    """Прерывает текущий тест"""
    # Просто показываем главное меню
    show_main_menu(message.chat.id)
    bot.send_message(message.chat.id, "Тест прерван")

@bot.message_handler(func=lambda message: message.text == "❌ Отмена")
def cancel_action(message):
    """Отменяет текущее действие"""
    show_main_menu(message.chat.id)

def show_test_results(chat_id, test_name):
    """Показывает результаты теста"""
    result = test_manager.get_test_result(chat_id, test_name)
    
    if not result:
        bot.send_message(chat_id, "Результаты теста не найдены")
        return
    
    # Показываем основной анализ
    bot.send_message(chat_id, "📊 **Результаты теста**\n\n" + result['analysis'])
    
    # Предлагаем обсудить результаты
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(telebot.types.KeyboardButton("💬 Обсудить результаты"))
    keyboard.add(telebot.types.KeyboardButton("📋 Выбрать другой тест"))
    keyboard.add(telebot.types.KeyboardButton("🏠 Главное меню"))
    
    bot.send_message(
        chat_id,
        "Хотите обсудить результаты подробнее или пройти другой тест?",
        reply_markup=keyboard
    )

@bot.message_handler(func=lambda message: message.text == "💬 Обсудить результаты")
def discuss_results(message):
    """Начинает обсуждение результатов теста"""
    bot.send_message(
        message.chat.id,
        "Расскажите, что вас больше всего заинтересовало в результатах? "
        "Какие выводы вы сделали? Задавайте любые вопросы о тесте!",
        reply_markup=create_main_menu_keyboard()
    )

def show_main_menu(chat_id):
    """Показывает главное меню"""
    bot.send_message(
        chat_id,
        "Выберите действие:",
        reply_markup=create_main_menu_keyboard()
    )

def create_main_menu_keyboard():
    """Создает клавиатуру главного меню"""
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(telebot.types.KeyboardButton("📊 Пройти тест"))
    keyboard.add(telebot.types.KeyboardButton("💬 Поговорить с ботом"))
    return keyboard

# Обновим основной обработчик сообщений чтобы игнорировать сообщения во время тестов
@bot.message_handler(func=lambda m: True)
def echo_handler(message):
    # Проверяем, не находится ли пользователь в процессе теста
    question_data = test_manager.get_current_question(message.chat.id)
    if question_data:
        # Если есть активный тест, игнорируем обычные сообщения
        return
        
    user_id = message.from_user.id
    chat_id = message.chat.id
    
    if message.content_type != 'text':
        return
        
    text = message.text.strip()
    get_user_chats().add(chat_id)

    bot.send_chat_action(message.chat.id, 'typing')
    handle_message_as_bot(bot, chat_id, text)

if __name__ == '__main__':
    while True:
        try:
            bot.infinity_polling()
        except Exception as e:
            print(f"[Polling crash] {e}")
            traceback.print_exc()
            time.sleep(5)
