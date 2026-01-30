# bot.py
# Requirements:
#   pip install python-telegram-bot==21.6 openai>=1.55.0
#
# Env vars required:
#   TELEGRAM_BOT_TOKEN=<token from BotFather>
#   OPENAI_API_KEY=<sk-...>
#   OPENAI_VECTOR_STORE_ID=<vs_...>
#
# Optional access control:
#   ALLOWED_USER_IDS=12345,67890
#   ALLOWED_CHAT_IDS=-100111222333
#
# Optional quality/behavior knobs:
#   FILE_SEARCH_MAX_RESULTS=30          # 1..50
#   FILE_SEARCH_SCORE_THRESHOLD=0.15    # 0..1
#   MODEL_EXTRACT=gpt-4.1-mini
#   MODEL_FINAL=gpt-4.1-mini
#
# Optional admin notifications when bot can't find answer:
#   ADMIN_CHAT_ID=-1001234567890        # where to forward "not found" questions

import os
import json
import logging
from typing import Set, Optional, Dict, Any

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tg-pre-sale-engineer-bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()

ALLOWED_USER_IDS_RAW = os.getenv("ALLOWED_USER_IDS", "").strip()
ALLOWED_CHAT_IDS_RAW = os.getenv("ALLOWED_CHAT_IDS", "").strip()

MODEL_EXTRACT = os.getenv("MODEL_EXTRACT", "gpt-4.1-mini").strip()
MODEL_FINAL = os.getenv("MODEL_FINAL", "gpt-4.1-mini").strip()

ADMIN_CHAT_ID_RAW = os.getenv("ADMIN_CHAT_ID", "").strip()

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

ADMIN_CHAT_ID: Optional[int] = None
if ADMIN_CHAT_ID_RAW:
    try:
        ADMIN_CHAT_ID = int(ADMIN_CHAT_ID_RAW)
    except ValueError:
        ADMIN_CHAT_ID = None

def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

FILE_SEARCH_MAX_RESULTS = max(1, min(50, _get_int_env("FILE_SEARCH_MAX_RESULTS", 30)))
FILE_SEARCH_SCORE_THRESHOLD = max(0.0, min(1.0, _get_float_env("FILE_SEARCH_SCORE_THRESHOLD", 0.15)))

client = OpenAI(api_key=OPENAI_API_KEY)

def _is_allowed(update: Update) -> bool:
    # If no whitelist configured -> allow all (pilot)
    if not (ALLOWED_USER_IDS or ALLOWED_CHAT_IDS):
        return True
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

# ---------------------------
# Modes (manager-friendly)
# ---------------------------
DEFAULT_MODE = "presale"  # best for managers

MODE_DESCRIPTIONS = {
    "presale": "пресейл-ответ: архитектура/ограничения/что уточнить",
    "client": "ответ для клиента (без лишней кухни)",
    "diag": "диагностика: ошибки/проверки/шаги",
    "short": "коротко, 3-7 строк",
}

def _get_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    mode = context.user_data.get("mode", DEFAULT_MODE)
    return mode if mode in MODE_DESCRIPTIONS else DEFAULT_MODE

def _set_mode(context: ContextTypes.DEFAULT_TYPE, mode: str) -> None:
    if mode in MODE_DESCRIPTIONS:
        context.user_data["mode"] = mode

# ---------------------------
# Prompting
# ---------------------------
EXTRACTOR_SYSTEM = f"""
Ты внутренний пресейл-инженер/техподдержка для менеджеров.

ТВОЯ ЗАДАЧА: извлечь максимально релевантные ФАКТЫ из внутренних документов через tool file_search,
при необходимости из нескольких файлов (чтобы потом собрать единый ответ).

ЖЁСТКИЕ ПРАВИЛА:
- Используй ТОЛЬКО информацию из найденных фрагментов file_search.
- НЕЛЬЗЯ додумывать или предполагать.
- Если фактов недостаточно — верни статус NOT_FOUND или PARTIAL и сформулируй, чего не хватает.

Верни результат СТРОГО в JSON (без текста вокруг) в таком виде:
{{
  "status": "OK" | "PARTIAL" | "NOT_FOUND",
  "facts": [
    {{
      "fact": "краткий факт",
      "why_relevant": "зачем это важно менеджеру",
      "source_hint": "название документа или краткая ссылка на него"
    }}
  ],
  "assumptions": [
    "только если явно написано в документах (иначе пусто)"
  ],
  "questions_to_ask": [
    "2-5 уточняющих вопросов, если нужно"
  ],
  "sources": [
    "список документов (уникальные названия)"
  ]
}}

Язык: русский. Термины (SIP, RTP, PoE, ONVIF и т.п.) сохраняй как в документах.
"""

def _final_system(mode: str) -> str:
    base = f"""
Ты — внутренний пресейл-инженер/техподдержка для менеджеров.
Используй ТОЛЬКО предоставленные факты (facts). Нельзя добавлять новые знания.
Если факты пустые или статус NOT_FOUND — верни ровно:
"Не нашёл в базе знаний. Уточни вводные или эскалируй инженеру."

Форматируй ответ удобно для менеджера.
Всегда заканчивай строкой: "Источники: <...>" (или "Источники: нет").
Язык: русский.
"""
    if mode == "presale":
        return base + """
РЕЖИМ: ПРЕСЕЙЛ.
Структура:
1) Вывод (1-2 строки)
2) Как это работает / архитектура (пункты)
3) Требования/ограничения (пункты)
4) Типовая схема (если есть факты)
5) Что уточнить у клиента (вопросы)
6) Источники
"""
    if mode == "client":
        return base + """
РЕЖИМ: ДЛЯ КЛИЕНТА.
- Пиши проще, без внутренней кухни.
- Без лишних деталей, но без потери точности.
Структура:
1) Короткий ответ
2) Условия/ограничения
3) Какие данные нужны для точного ответа (если нужно)
4) Источники
"""
    if mode == "diag":
        return base + """
РЕЖИМ: ДИАГНОСТИКА.
Структура:
1) Возможная причина (ТОЛЬКО если есть факт)
2) Что проверить (шаги)
3) Логи/параметры/порты (если есть факты)
4) Когда эскалировать инженеру
5) Источники
"""
    # short
    return base + """
РЕЖИМ: КОРОТКО.
- 3–7 строк максимум.
- Только самое важное.
- Источники в конце.
"""

def _tool_spec() -> list:
    # В Responses API настройки задаются на верхнем уровне tool-объекта.
    # Самый совместимый параметр — max_num_results.
    return [{
        "type": "file_search",
        "vector_store_ids": [VECTOR_STORE_ID],
        "max_num_results": FILE_SEARCH_MAX_RESULTS,
        # ranking_options иногда не принимается в Responses API/SDK → оставляем выключенным.
        # "ranking_options": {"ranker": "auto", "score_threshold": FILE_SEARCH_SCORE_THRESHOLD},
    }]


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None

def extract_facts(question: str) -> Dict[str, Any]:
    resp = client.responses.create(
        model=MODEL_EXTRACT,
        input=[
            {"role": "system", "content": EXTRACTOR_SYSTEM},
            {"role": "user", "content": question},
        ],
        tools=_tool_spec(),
        # If you don't want OpenAI to store responses, you can set store=False (optional).
        # store=False,
    )
    text = (getattr(resp, "output_text", "") or "").strip()
    data = _safe_json_load(text)
    if not data:
        # Fallback: treat as not found (keeps safety)
        return {"status": "NOT_FOUND", "facts": [], "assumptions": [], "questions_to_ask": [], "sources": []}
    # Normalize
    data.setdefault("status", "NOT_FOUND")
    data.setdefault("facts", [])
    data.setdefault("assumptions", [])
    data.setdefault("questions_to_ask", [])
    data.setdefault("sources", [])
    return data

def compose_answer(facts_blob: Dict[str, Any], mode: str) -> str:
    status = facts_blob.get("status", "NOT_FOUND")
    facts = facts_blob.get("facts", []) or []
    sources = facts_blob.get("sources", []) or []
    questions = facts_blob.get("questions_to_ask", []) or []

    if status == "NOT_FOUND" or not facts:
        return "Не нашёл в базе знаний. Уточни вводные или эскалируй инженеру.\nИсточники: нет"

    payload = {
        "status": status,
        "facts": facts,
        "questions_to_ask": questions,
        "sources": sources,
    }

    resp = client.responses.create(
        model=MODEL_FINAL,
        input=[
            {"role": "system", "content": _final_system(mode)},
            {"role": "user", "content": f"Вопрос менеджера: {facts_blob.get('original_question','') or ''}\n\nВот извлечённые факты (JSON):\n{json.dumps(payload, ensure_ascii=False)}"},
        ],
        # no tools here — prevents adding new facts
    )
    answer = (getattr(resp, "output_text", "") or "").strip()
    if not answer:
        answer = "Не нашёл в базе знаний. Уточни вводные или эскалируй инженеру.\nИсточники: нет"
    return answer

async def notify_admin_if_not_found(context: ContextTypes.DEFAULT_TYPE, question: str, facts_blob: Dict[str, Any]):
    if ADMIN_CHAT_ID is None:
        return
    try:
        status = facts_blob.get("status", "NOT_FOUND")
        if status == "NOT_FOUND" or not facts_blob.get("facts"):
            msg = "⚠️ Бот НЕ нашёл ответ в базе.\n\nВопрос:\n" + question
            await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=msg)
    except Exception:
        logger.exception("Failed to notify admin")

# ---------------------------
# Telegram handlers
# ---------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return
    await update.message.reply_text(
        "Привет! Я бот-помощник для менеджеров.\n"
        "Отвечаю строго по внутренним документам (могу собирать ответ из нескольких файлов).\n\n"
        "Команды:\n"
        "/mode — показать режим\n"
        "/mode presale|client|diag|short — сменить режим\n"
        "/whoami — узнать твой user_id\n"
    )

async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return
    user = update.effective_user
    chat = update.effective_chat
    uname = f"@{user.username}" if user and user.username else "(нет)"
    await update.message.reply_text(f"user_id: {user.id}\nusername: {uname}\nchat_id: {chat.id}\nchat_type: {chat.type}")

async def mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return

    parts = (update.message.text or "").split()
    if len(parts) == 1:
        mode = _get_mode(context)
        await update.message.reply_text(f"Текущий режим: {mode} — {MODE_DESCRIPTIONS[mode]}")
        return

    new_mode = parts[1].strip().lower()
    if new_mode not in MODE_DESCRIPTIONS:
        await update.message.reply_text("Неизвестный режим. Доступно: presale, client, diag, short")
        return
    _set_mode(context, new_mode)
    await update.message.reply_text(f"Ок. Режим: {new_mode} — {MODE_DESCRIPTIONS[new_mode]}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return

    question = update.message.text.strip()
    if len(question) < 2:
        return
    if len(question) > 4000:
        await update.message.reply_text("Слишком длинный запрос. Сократи до 1–2 абзацев.")
        return

    try:
        mode = _get_mode(context)

        # Pass 1: extract facts using file_search
        facts_blob = extract_facts(question)
        facts_blob["original_question"] = question

        # Optional: notify admin on NOT_FOUND
        await notify_admin_if_not_found(context, question, facts_blob)

        # Pass 2: compose final answer using only extracted facts
        answer = compose_answer(facts_blob, mode)

        if len(answer) > 3900:
            answer = answer[:3900] + "\n\n(сообщение обрезано)"

        await update.message.reply_text(answer)

    except Exception as e:
        logger.exception("Error while handling message")
        await update.message.reply_text(f"Ошибка при обработке запроса: {e}")

def main():
    err = _ensure_config()
    if err:
        raise RuntimeError(err)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("mode", mode_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is starting (polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
