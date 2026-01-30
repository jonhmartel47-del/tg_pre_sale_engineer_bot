# bot.py
# Requirements:
#   pip install python-telegram-bot==21.6 openai==1.55.3
#
# Env vars required:
#   TELEGRAM_BOT_TOKEN=<token from BotFather>
#   OPENAI_API_KEY=<sk-...>
#   OPENAI_VECTOR_STORE_ID=<vs_...>
#
# Optional (recommended) access control:
#   ALLOWED_USER_IDS=12345,67890         # Telegram numeric user IDs (comma-separated)
#   ALLOWED_CHAT_IDS=-100111222333,-999  # Allowed group chat IDs (comma-separated)
#
# How to get your user_id:
#   Start the bot and send /whoami

import os
import logging
from typing import Set, Optional

from telegram import Update
from telegram.constants import ChatType
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("tg-pre-sales-engineer-bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()

ALLOWED_USER_IDS_RAW = os.getenv("ALLOWED_USER_IDS", "").strip()
ALLOWED_CHAT_IDS_RAW = os.getenv("ALLOWED_CHAT_IDS", "").strip()

def _parse_int_set(csv: str) -> Set[int]:
    out: Set[int] = set()
    if not csv:
        return out
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            pass
    return out

ALLOWED_USER_IDS: Set[int] = _parse_int_set(ALLOWED_USER_IDS_RAW)
ALLOWED_CHAT_IDS: Set[int] = _parse_int_set(ALLOWED_CHAT_IDS_RAW)

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_RULES = """
Ты внутренний пресейл-инженер/техподдержка для менеджеров компании.

КРИТИЧЕСКИ ВАЖНО:
1) Отвечай ТОЛЬКО на основе фрагментов, найденных через инструмент file_search (внутренняя база знаний).
2) Запрещено додумывать, предполагать, "возможно", "скорее всего" и т.п.
3) Если точного ответа в найденных фрагментах нет — ответь ровно так:
   "Не нашёл в базе знаний. Уточни вводные или эскалируй инженеру."

ФОРМАТ:
- Коротко и по делу.
- Если уместно: шаги 1–2–3.
- В конце всегда добавляй строку:
  "Источники: <названия документов или 'нет'>"
  Если источников нет — напиши "Источники: нет".
"""

def _is_allowed(update: Update) -> bool:
    """Allow by chat_id (group) OR by user_id (private/group). If no lists set -> allow all (pilot mode)."""
    if not (ALLOWED_USER_IDS or ALLOWED_CHAT_IDS):
        return True  # pilot mode: open access

    chat_id = update.effective_chat.id if update.effective_chat else None
    user_id = update.effective_user.id if update.effective_user else None

    if chat_id is not None and chat_id in ALLOWED_CHAT_IDS:
        return True
    if user_id is not None and user_id in ALLOWED_USER_IDS:
        return True
    return False

def _ensure_config() -> Optional[str]:
    if not TELEGRAM_BOT_TOKEN:
        return "Missing TELEGRAM_BOT_TOKEN"
    if not OPENAI_API_KEY:
        return "Missing OPENAI_API_KEY"
    if not VECTOR_STORE_ID:
        return "Missing OPENAI_VECTOR_STORE_ID"
    return None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return

    txt = (
        "Привет! Я бот-помощник для менеджеров.\n\n"
        "Задай вопрос по продукту/архитектуре/ошибкам — я отвечу ТОЛЬКО по внутренней базе знаний.\n"
        "Если ответа нет в документах, я так и скажу.\n\n"
        "Команды:\n"
        "/help — подсказка\n"
        "/whoami — узнать твой user_id\n"
    )
    await update.message.reply_text(txt)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return

    await update.message.reply_text(
        "Примеры вопросов:\n"
        "• Какие порты/протоколы используются?\n"
        "• Как выглядит типовая схема подключения?\n"
        "• Что означает ошибка <код/текст>?\n"
        "• Какие требования к сети/PoE/железу?\n\n"
        "Важно: я отвечаю только по базе знаний. Если не нахожу — прошу уточнить или эскалировать."
    )

async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /whoami полезен даже если нет в whitelist — но лучше тоже ограничить
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return

    user = update.effective_user
    chat = update.effective_chat
    await update.message.reply_text(
        f"user_id: {user.id}\n"
        f"username: @{user.username}" if user and user.username else f"user_id: {user.id}\nusername: (нет)\n"
        + f"chat_id: {chat.id}\nchat_type: {chat.type}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return

    question = update.message.text.strip()
    if len(question) < 2:
        return

    # Небольшая защита от “романов”
    if len(question) > 4000:
        await update.message.reply_text("Слишком длинный запрос. Сократи, пожалуйста, до 1–2 абзацев.")
        return

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_RULES},
                {"role": "user", "content": question},
            ],
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [VECTOR_STORE_ID],
                }
            ],
        )

        answer = (getattr(resp, "output_text", "") or "").strip()
        if not answer:
            answer = "Не нашёл в базе знаний. Уточни вводные или эскалируй инженеру.\nИсточники: нет"

        # Telegram ограничение ~4096 символов
        if len(answer) > 3900:
            answer = answer[:3900] + "\n\n(сообщение обрезано)"

        await update.message.reply_text(answer)

    except Exception as e:
        logger.exception("Error while handling message")
        await update.message.reply_text(f"Ошибка при обработке запроса: {e}")

def main():
    cfg_err = _ensure_config()
    if cfg_err:
        raise RuntimeError(cfg_err)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is starting (polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
