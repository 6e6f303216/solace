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

@bot.message_handler(func=lambda m: True)
def echo_handler(message):
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
