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
import math
import re
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

def _in_calculator(context: ContextTypes.DEFAULT_TYPE) -> bool:
    return bool(context.user_data.get("calculator_active", False))

def _set_calculator(context: ContextTypes.DEFAULT_TYPE, active: bool) -> None:
    context.user_data["calculator_active"] = active
    
def allowed_speaker_types(room_type: str) -> Set[str]:
    rt = room_type.lower()
    if "офис" in rt or "переговор" in rt:
        return {"потолочный", "настенный"}
    if "корид" in rt or "холл" in rt:
        return {"потолочный", "настенный"}
    if "склад" in rt:
        return {"настенный", "колонный", "рупорный"}
    if "цех" in rt:
        return {"колонный", "рупорный"}
    if "улиц" in rt:
        return {"уличный", "рупорный", "проекторный"}
    # fallback
    return {"потолочный", "настенный", "колонный", "рупорный"}

def parse_room_params(text: str) -> Dict[str, Any]:
    """
    Accepts multiline like:
    S=240
    H=3.2
    тип=офис
    шум=55
    контент=только речь
    препятствия=перегородки
    монтаж=потолок
    этаж=1
    """
    t = text.lower().strip()

    def grab_num(key: str) -> Optional[float]:
        m = re.search(rf"{key}\s*=\s*([0-9]+(?:[.,][0-9]+)?)", t)
        if not m:
            return None
        return float(m.group(1).replace(",", "."))

    def grab_str(key: str) -> Optional[str]:
        m = re.search(rf"{key}\s*=\s*([^\n\r]+)", t)
        if not m:
            return None
        return m.group(1).strip()

    S = grab_num("s")
    H = grab_num("h")
    room_type = grab_str("тип") or grab_str("тип_помещения")
    noise = grab_num("шум") or grab_num("уровень_шума")
    content = grab_str("контент")
    obstacles = grab_str("препятствия")
    mount = grab_str("монтаж") or grab_str("разрешённый_монтаж")
    floor = grab_str("этаж") or "1"

    if not S or not H or not room_type or not content or not obstacles or not mount:
        missing = []
        if not S: missing.append("S")
        if not H: missing.append("H")
        if not room_type: missing.append("тип")
        if not content: missing.append("контент")
        if not obstacles: missing.append("препятствия")
        if not mount: missing.append("монтаж")
        raise ValueError("Не хватает параметров: " + ", ".join(missing))

    return {
        "S": float(S),
        "H": float(H),
        "room_type": room_type,
        "noise": float(noise) if noise is not None else None,
        "content": content,
        "obstacles": obstacles,
        "mount": mount,
        "floor": floor
    }

def default_noise_by_type(room_type: str) -> float:
    rt = room_type.lower()
    if "офис" in rt or "переговор" in rt:
        return 50.0
    if "корид" in rt or "холл" in rt:
        return 55.0
    if "склад" in rt:
        return 65.0
    if "улиц" in rt:
        return 70.0
    if "цех" in rt:
        # если цех без уточнений — берём умеренный
        return 75.0
    return 60.0


def rmax_effective(room_type: str, rmax: float) -> float:
    rt = room_type.lower()
    if "офис" in rt or "переговор" in rt:
        return min(rmax, 6.0)
    if "склад" in rt:
        # в ТЗ 8–10 м: возьмём 9 как середину
        return min(rmax, 9.0)
    if "цех" in rt:
        # 6–8 м: возьмём 7
        return min(rmax, 7.0)
    return rmax

def calc_for_model(S: float, L_target: float, room_type: str, maxSPL_1m: float) -> Dict[str, Any]:
    # Rmax = 10 ^ ((maxSPL_1m - L_target) / 20)
    rmax = 10 ** ((maxSPL_1m - L_target) / 20.0)
    r_eff = rmax_effective(room_type, rmax)

    k_overlap = 0.55
    s_one = math.pi * (r_eff ** 2) * k_overlap
    N = math.ceil(S / s_one) if s_one > 0 else 0

    return {
        "Rmax": rmax,
        "Rmax_effective": r_eff,
        "S_one": s_one,
        "N": N,
        "step": round(r_eff * 1.0, 2)  # условный шаг = Rmax_effective (можно уточнять)
    }


CATALOG_EXTRACT_SYSTEM = """
Ты извлекаешь каталог IP-громкоговорителей из внутренних документов.

Нужно вернуть СТРОГО JSON без текста вокруг:
{
  "items": [
    {
      "model": "...",
      "type": "потолочный|настенный|колонный|рупорный|уличный|проекторный",
      "maxSPL_1m": 120,
      "P_poe": 7.5,
      "poe_standard": "802.3af|802.3at|802.3bt|unknown",
      "price": 0
    }
  ]
}

Правила:
- maxSPL_1m распознавай по синонимам: "Максимальный уровень громкости", "Max SPL", "Maximum SPL" и т.п.
- Если maxSPL_1m нет — НЕ добавляй модель в items.
- DC-питание игнорируй; poe_standard извлекай если есть.
- P_poe если нет — оставь null.
- price если нет — оставь null.
- Собирай из нескольких файлов, если нужно.
"""

def get_catalog_from_vector_store() -> Dict[str, Any]:
    # Просим собрать каталог из базы знаний.
    # Для больших баз лучше отдельный catalog-файл, но и из паспортов тоже сможет вытащить.
    resp = client.responses.create(
        model=MODEL_EXTRACT,
        input=[
            {"role": "system", "content": CATALOG_EXTRACT_SYSTEM},
            {"role": "user", "content": "Собери список моделей и их параметры для расчёта (catalog)."},
        ],
        tools=[{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
            "max_num_results": 50
        }],
    )
    text = (getattr(resp, "output_text", "") or "").strip()
    data = _safe_json_load(text)
    if not data or "items" not in data:
        return {"items": []}
    return data
    
async def handle_calculator_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    try:
        params = parse_room_params(text)
    except Exception as e:
        await update.message.reply_text(f"Не понял ввод. {e}\nНапиши /calc_help для примера.")
        return

    S = params["S"]
    room_type = params["room_type"]
    noise = params["noise"] if params["noise"] is not None else default_noise_by_type(room_type)

    # Фиксированное правило +15 дБ
    L_target = noise + 15.0

    # Получаем каталог моделей из базы знаний
    catalog = get_catalog_from_vector_store()
    items = catalog.get("items", [])

    if not items:
        await update.message.reply_text(
            "Не смог собрать каталог моделей из базы знаний.\n"
            "Совет: добавь отдельный файл catalog_speakers (таблица с model/type/maxSPL_1m/P_poe/poe_standard/price) и попробуй снова."
        )
        return

    allowed_types = allowed_speaker_types(room_type)
    mount = params["mount"].lower()

    # Фильтр по применимости типа + монтажу
    # Маппинг монтаж -> допустимые типы (упрощенно)
    mount_allowed = {
        "потолок": {"потолочный"},
        "стена": {"настенный", "рупорный"},
        "колонна": {"колонный"},
        "любой": {"потолочный", "настенный", "колонный", "рупорный", "уличный", "проекторный"},
    }
    mount_set = mount_allowed.get(mount, mount_allowed["любой"])

    filtered = []
    for it in items:
        t = (it.get("type") or "").lower().strip()
        if not t:
            continue
        if t not in allowed_types:
            continue
        if t not in mount_set and mount != "любой":
            continue
        maxspl = it.get("maxSPL_1m")
        if maxspl is None:
            continue
        filtered.append(it)

    if not filtered:
        await update.message.reply_text(
            f"Нет подходящих моделей под тип помещения='{room_type}' и монтаж='{mount}'.\n"
            f"Допустимые типы для помещения: {', '.join(sorted(allowed_types))}."
        )
        return

    # Группируем по типам и считаем
    results_by_type: Dict[str, list] = {}
    for it in filtered:
        t = it["type"].lower()
        calc = calc_for_model(S, L_target, room_type, float(it["maxSPL_1m"]))
        entry = {**it, **calc}
        results_by_type.setdefault(t, []).append(entry)

    # Для каждого типа показываем 1–3 лучших (по N, затем по цене если есть)
    lines = []
    lines.append("🧮 Расчёт IP-громкоговорителей (SPL)\n")
    lines.append(f"Параметры: S={S} м², тип={room_type}, шум={noise} дБ → L_target={L_target} дБ (+15), монтаж={mount}, этаж={params['floor']}\n")

    for t, entries in results_by_type.items():
        # сортировка: меньше N — лучше; если цена есть — дешевле лучше
        def key_fn(e):
            price = e.get("price")
            price_val = float(price) if price is not None else 1e18
            return (e.get("N", 10**9), price_val)
        entries_sorted = sorted(entries, key=key_fn)

        lines.append(f"### Вариант: {t}")
        for e in entries_sorted[:3]:
            poe = e.get("P_poe")
            poe_str = f"{poe} Вт" if poe is not None else "нет данных"
            poe_std = e.get("poe_standard") or "unknown"
            price = e.get("price")
            price_str = f"{price}" if price is not None else "нет"

            lines.append(
                f"- Модель: {e.get('model')}\n"
                f"  maxSPL_1m: {e.get('maxSPL_1m')} дБ\n"
                f"  Rmax_effective: {round(e.get('Rmax_effective', 0),2)} м\n"
                f"  Рекоменд. шаг: ~{e.get('step')} м\n"
                f"  Кол-во: {e.get('N')} шт\n"
                f"  PoE: {poe_str} / стандарт: {poe_std}\n"
                f"  Цена: {price_str}\n"
            )

    # PoE итог (если есть P_poe)
    total_poe_known = True
    P_total = 0.0
    for t, entries in results_by_type.items():
        best = sorted(entries, key=lambda e: (e.get("N", 10**9), float(e.get("price") or 1e18)))[0]
        if best.get("P_poe") is None:
            total_poe_known = False
            continue
        P_total += float(best["N"]) * float(best["P_poe"])

    if total_poe_known and P_total > 0:
        P_required = P_total * 1.2
        lines.append(f"\n🔌 PoE (оценка по лучшим вариантам каждого типа): P_total={round(P_total,2)} Вт, P_required(×1.2)={round(P_required,2)} Вт")
    else:
        lines.append("\n🔌 PoE: у некоторых моделей нет P_poe — расчёт коммутаторов будет оценочным.")

    # Telegram limit
    out = "\n".join(lines)
    if len(out) > 3900:
        out = out[:3900] + "\n\n(сообщение обрезано)"
    await update.message.reply_text(out)


get_catalog_from_vector_store()

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
        # If Calculator mode enabled, route to calculator logic
    if _in_calculator(context):
        await handle_calculator_message(update, context)
        return

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

    # Calculator commands
    app.add_handler(CommandHandler("calculator", calculator_cmd))
    app.add_handler(CommandHandler("calc", calculator_cmd))
    app.add_handler(CommandHandler("calc_help", calc_help_cmd))
    app.add_handler(CommandHandler("calc_stop", calc_stop_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is starting (polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

