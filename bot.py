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
#   MODEL_EXTRACT=gpt-4o-mini          # recommended for structured outputs
#   MODEL_FINAL=gpt-4.1-mini
#
# Optional admin notifications when bot can't find answer:
#   ADMIN_CHAT_ID=-1001234567890        # where to forward "not found" questions

import os
import sys
import json
import logging
import math
import re
from typing import Set, Optional, Dict, Any, List, Tuple

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from openai import OpenAI


# ---------------------------
# Logging (force stderr so Railway shows it reliably)
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("tg-pre-sale-engineer-bot")


# ---------------------------
# Environment
# ---------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()

ALLOWED_USER_IDS_RAW = os.getenv("ALLOWED_USER_IDS", "").strip()
ALLOWED_CHAT_IDS_RAW = os.getenv("ALLOWED_CHAT_IDS", "").strip()

MODEL_EXTRACT = os.getenv("MODEL_EXTRACT", "gpt-4o-mini").strip()
MODEL_FINAL = os.getenv("MODEL_FINAL", "gpt-4.1-mini").strip()

ADMIN_CHAT_ID_RAW = os.getenv("ADMIN_CHAT_ID", "").strip()

FILE_SEARCH_MAX_RESULTS = 30
try:
    FILE_SEARCH_MAX_RESULTS = int(os.getenv("FILE_SEARCH_MAX_RESULTS", "30").strip() or "30")
except ValueError:
    FILE_SEARCH_MAX_RESULTS = 30
FILE_SEARCH_MAX_RESULTS = max(1, min(50, FILE_SEARCH_MAX_RESULTS))

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------
# Helpers
# ---------------------------
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


def _ensure_config() -> Optional[str]:
    if not TELEGRAM_BOT_TOKEN:
        return "Missing TELEGRAM_BOT_TOKEN"
    if not OPENAI_API_KEY:
        return "Missing OPENAI_API_KEY"
    if not VECTOR_STORE_ID:
        return "Missing OPENAI_VECTOR_STORE_ID"
    return None


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


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _tool_spec() -> list:
    # Responses API: file_search tool with max_num_results at top-level.
    return [{
        "type": "file_search",
        "vector_store_ids": [VECTOR_STORE_ID],
        "max_num_results": FILE_SEARCH_MAX_RESULTS,
    }]


# ---------------------------
# Modes (Q&A)
# ---------------------------
DEFAULT_MODE = "presale"

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
# Calculator state
# ---------------------------
def _in_calculator(context: ContextTypes.DEFAULT_TYPE) -> bool:
    return bool(context.user_data.get("calculator_active", False))


def _set_calculator(context: ContextTypes.DEFAULT_TYPE, active: bool) -> None:
    context.user_data["calculator_active"] = active


def _looks_like_calc_input(text: str) -> bool:
    """
    Auto-detect calculator input so it still works after Railway restarts
    even if user didn't run /calc.
    """
    t = (text or "").strip()
    return bool(re.search(r"(?im)^s\s*=", t) and re.search(r"(?im)^h\s*=", t) and re.search(r"(?im)^тип", t))


# ---------------------------
# Calculator: parsing & math (NO "этаж")
# ---------------------------
def parse_room_params(text: str) -> Dict[str, Any]:
    """
    Accepts multiline like:
    S=2400
    H=3.5
    тип=склад
    шум=85                (optional; if missing/empty -> auto)
    контент=только речь
    препятствия=перегородки
    монтаж=стена           (optional; used only as preference, not hard filter)
    (этаж=... is ignored if present)
    """
    t = text.strip()

    def grab_num(key: str) -> Optional[float]:
        # allow "шум=" empty -> None
        m_empty = re.search(rf"(?im)^{re.escape(key)}\s*=\s*$", t)
        if m_empty:
            return None

        m = re.search(rf"(?im)^{re.escape(key)}\s*=\s*([0-9]+(?:[.,][0-9]+)?)\s*$", t)
        if not m:
            return None
        return float(m.group(1).replace(",", "."))

    def grab_str(*keys: str) -> Optional[str]:
        for key in keys:
            m = re.search(rf"(?im)^{re.escape(key)}\s*=\s*(.+?)\s*$", t)
            if m:
                val = m.group(1).strip()
                if val:
                    return val
        return None

    S = grab_num("S")
    H = grab_num("H")
    room_type = grab_str("тип", "тип_помещения")
    noise = grab_num("шум") or grab_num("уровень_шума")
    content = grab_str("контент")
    obstacles = grab_str("препятствия")
    mount = grab_str("монтаж", "разрешённый_монтаж")  # optional

    missing = []
    if not S:
        missing.append("S")
    if not H:
        missing.append("H")
    if not room_type:
        missing.append("тип")
    if not content:
        missing.append("контент")
    if not obstacles:
        missing.append("препятствия")
    if missing:
        raise ValueError("Не хватает параметров: " + ", ".join(missing))

    return {
        "S": float(S),
        "H": float(H),
        "room_type": room_type.strip(),
        "noise": float(noise) if noise is not None else None,
        "content": content.strip(),
        "obstacles": obstacles.strip(),
        "mount": (mount or "").strip().lower(),  # may be ""
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
        return 75.0
    return 60.0


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
    return {"потолочный", "настенный", "колонный", "рупорный"}


def rmax_effective(room_type: str, rmax: float) -> float:
    rt = room_type.lower()
    if "офис" in rt or "переговор" in rt:
        return min(rmax, 6.0)
    if "склад" in rt:
        return min(rmax, 9.0)  # 8–10m midpoint
    if "цех" in rt:
        return min(rmax, 7.0)  # 6–8m midpoint
    return rmax


def calc_for_model(S: float, L_target: float, room_type: str, maxSPL_1m: float) -> Dict[str, Any]:
    # Rmax = 10 ^ ((maxSPL_1m - L_target) / 20)
    rmax = 10 ** ((maxSPL_1m - L_target) / 20.0)
    r_eff = rmax_effective(room_type, rmax)

    k_overlap = 0.55
    s_one = math.pi * (r_eff ** 2) * k_overlap
    N = math.ceil(S / s_one) if s_one > 0 else 0

    return {
        "Rmax_effective": r_eff,
        "N": N,
        "step": round(r_eff, 2),
    }


# ---------------------------
# Catalog extraction (Vector Store) - stable JSON schema
# ---------------------------
CATALOG_EXTRACT_SYSTEM = """
Ты извлекаешь КАТАЛОГ IP-громкоговорителей из внутренних документов.

ВАЖНО:
- Если найден файл catalog_speakers.json, используй его как ОСНОВНОЙ источник каталога.
- Паспортами дополняй только если в catalog_speakers.json нет строки по модели.

Нужно вернуть СТРОГО JSON без текста вокруг:
{
  "items": [
    {
      "model": "строка",
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
- Если maxSPL_1m не найден — НЕ добавляй модель в items.
- DC-питание игнорируй; poe_standard извлекай если есть (если нет — "unknown").
- P_poe если нет — ставь null.
- price если нет — ставь null.
"""


def get_catalog_from_vector_store() -> Dict[str, Any]:
    """
    Returns: {"items": [ {model,type,maxSPL_1m,P_poe,poe_standard,price}, ... ]}
    Guaranteed JSON via Structured Outputs (json_schema).
    """
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "type": {"type": "string"},
                        "maxSPL_1m": {"type": "number"},
                        "P_poe": {"type": ["number", "null"]},
                        "poe_standard": {"type": "string"},
                        "price": {"type": ["number", "null"]},
                    },
                    "required": ["model", "type", "maxSPL_1m", "P_poe", "poe_standard", "price"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    try:
resp = client.responses.create(
    model=MODEL_EXTRACT,
    tool_choice="required",
    input=[
        {"role": "system", "content": CATALOG_EXTRACT_SYSTEM},
        {"role": "user", "content": "Найди catalog_speakers.json и верни полный каталог items."},
    ],
    tools=[{
        "type": "file_search",
        "vector_store_ids": [VECTOR_STORE_ID],
        "max_num_results": 50,
    }],
    text={
        "format": {
            "type": "json_schema",
            "name": "speaker_catalog",
            "strict": True,
            "schema": schema,
        }
    },
)
    except Exception:
        logger.exception("Catalog extraction failed")
        return {"items": []}

    text_out = (getattr(resp, "output_text", "") or "").strip()
    data = _safe_json_load(text_out)
    if not data:
        return {"items": []}

    items = data.get("items", []) or []

    # Normalize / sanitize
    norm_items: List[Dict[str, Any]] = []
    for it in items:
        try:
            model = (it.get("model") or "").strip()
            stype = (it.get("type") or "").strip().lower()
            maxspl = it.get("maxSPL_1m", None)
            poe = it.get("P_poe", None)
            poe_std = (it.get("poe_standard") or "unknown").strip()
            price = it.get("price", None)

            if not model or not stype or maxspl is None:
                continue

            norm_items.append({
                "model": model,
                "type": stype,
                "maxSPL_1m": float(maxspl),
                "P_poe": float(poe) if poe is not None else None,
                "poe_standard": poe_std,
                "price": float(price) if price is not None else None,
            })
        except Exception:
            continue

    return {"items": norm_items}


# ---------------------------
# Q&A (two-pass)
# ---------------------------
EXTRACTOR_SYSTEM = """
Ты внутренний пресейл-инженер/техподдержка для менеджеров.

ТВОЯ ЗАДАЧА: извлечь максимально релевантные ФАКТЫ из внутренних документов через tool file_search,
при необходимости из нескольких файлов (чтобы потом собрать единый ответ).

ЖЁСТКИЕ ПРАВИЛА:
- Используй ТОЛЬКО информацию из найденных фрагментов file_search.
- НЕЛЬЗЯ додумывать или предполагать.
- Если фактов недостаточно — верни статус NOT_FOUND или PARTIAL и сформулируй, чего не хватает.

Верни результат СТРОГО в JSON (без текста вокруг) в таком виде:
{
  "status": "OK" | "PARTIAL" | "NOT_FOUND",
  "facts": [
    {
      "fact": "краткий факт",
      "why_relevant": "зачем это важно менеджеру",
      "source_hint": "название документа или краткая ссылка на него"
    }
  ],
  "questions_to_ask": [
    "2-5 уточняющих вопросов, если нужно"
  ],
  "sources": [
    "список документов (уникальные названия)"
  ]
}

Язык: русский. Термины (SIP, RTP, PoE, ONVIF и т.п.) сохраняй как в документах.
"""


def _final_system(mode: str) -> str:
    base = """
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
2) Архитектура/как это работает (пункты)
3) Требования/ограничения (пункты)
4) Типовая схема (если есть факты)
5) Что уточнить у клиента (вопросы)
6) Источники
"""
    if mode == "client":
        return base + """
РЕЖИМ: ДЛЯ КЛИЕНТА.
- Пиши проще, без внутренней кухни.
Структура:
1) Короткий ответ
2) Условия/ограничения
3) Что уточнить (если нужно)
4) Источники
"""
    if mode == "diag":
        return base + """
РЕЖИМ: ДИАГНОСТИКА.
Структура:
1) Возможная причина (ТОЛЬКО если есть факт)
2) Что проверить (шаги)
3) Параметры/порты/логи (если есть факты)
4) Когда эскалировать
5) Источники
"""
    return base + """
РЕЖИМ: КОРОТКО.
- 3–7 строк максимум.
- Источники в конце.
"""


def extract_facts(question: str) -> Dict[str, Any]:
    resp = client.responses.create(
        model=MODEL_EXTRACT,
        input=[
            {"role": "system", "content": EXTRACTOR_SYSTEM},
            {"role": "user", "content": question},
        ],
        tools=_tool_spec(),
    )
    text = (getattr(resp, "output_text", "") or "").strip()
    data = _safe_json_load(text)
    if not data:
        return {"status": "NOT_FOUND", "facts": [], "questions_to_ask": [], "sources": []}

    data.setdefault("status", "NOT_FOUND")
    data.setdefault("facts", [])
    data.setdefault("questions_to_ask", [])
    data.setdefault("sources", [])
    return data


def compose_answer(facts_blob: Dict[str, Any], mode: str, original_question: str) -> str:
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
            {"role": "user", "content": f"Вопрос менеджера: {original_question}\n\nФакты (JSON):\n{json.dumps(payload, ensure_ascii=False)}"},
        ],
    )
    answer = (getattr(resp, "output_text", "") or "").strip()
    if not answer:
        answer = "Не нашёл в базе знаний. Уточни вводные или эскалируй инженеру.\nИсточники: нет"
    return answer


async def notify_admin_if_not_found(context: ContextTypes.DEFAULT_TYPE, question: str, facts_blob: Dict[str, Any]) -> None:
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
# Calculator output formatting (per your required format)
# ---------------------------
def _calc_sort_key(entry: Dict[str, Any]) -> Tuple[int, float]:
    n = int(entry.get("N", 10**9))
    price = entry.get("price")
    price_val = float(price) if price is not None else 1e18
    return (n, price_val)


def _mount_preference_rank(mount: str, speaker_type: str) -> int:
    """
    Lower is better.
    We do NOT hard-filter by mount because user wants multiple variants.
    We only sort preferred types first.
    """
    m = (mount or "").lower()
    t = (speaker_type or "").lower()

    if not m:
        return 10
    if m in ("любой", "any"):
        return 10

    # common synonyms
    if m in ("настенный", "настенная", "стена", "wall"):
        return 0 if t in ("настенный", "рупорный") else 5
    if m in ("потолок", "потолочный", "ceiling"):
        return 0 if t == "потолочный" else 5
    if m in ("колонна", "колонный", "column"):
        return 0 if t == "колонный" else 5

    return 10


async def handle_calculator_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("\n===== CALC REQUEST START =====")

    if not update.message or not update.message.text:
        print("No message or no text")
        return

    text = update.message.text.strip()
    print("Incoming text:", text)

    try:
        params = parse_room_params(text)
        print("Parsed params:", params)
    except Exception as e:
        print("Parse error:", e)
        await update.message.reply_text(f"Не понял ввод. {e}\nНапиши /calc_help для примера.")
        return

    S = params["S"]
    room_type = params["room_type"]
    noise = params["noise"] if params["noise"] is not None else default_noise_by_type(room_type)

    print("Room area S:", S)
    print("Room type:", room_type)
    print("Noise level:", noise)

    # Fixed +15 dB rule
    L_target = noise + 15.0
    print("Target SPL L_target:", L_target)

    # Get catalog
    print("Calling get_catalog_from_vector_store()...")
    catalog = get_catalog_from_vector_store()

    print("Catalog raw response type:", type(catalog))
    print("Catalog raw response:", catalog)

    if catalog is None:
        print("Catalog is None")

    items = catalog.get("items", []) if isinstance(catalog, dict) else []

    print("Items count:", len(items))

    if items:
        print("First item example:", items[0])
    else:
        print("No items found in catalog")

    if not items:
        print("ERROR: catalog empty")
        await update.message.reply_text(
            "Не смог собрать каталог моделей из базы знаний.\n"
            "Проверь, что в Vector Store загружен файл catalog_speakers.json и он проиндексирован.\n"
            "И что в нём есть поля: model, type, maxSPL_1m, P_poe, poe_standard, price."
        )
        return

    allowed_types = allowed_speaker_types(room_type)
    print("Allowed speaker types:", allowed_types)

    # Filter only by room applicability and required fields
    filtered: List[Dict[str, Any]] = []

    for i, it in enumerate(items):
        print(f"\nChecking item {i}: {it}")

        stype = (it.get("type") or "").lower().strip()
        print("Speaker type:", stype)

        if not stype:
            print("Rejected: empty type")
            continue

        if stype not in allowed_types:
            print("Rejected: type not allowed for this room")
            continue

        if it.get("maxSPL_1m") is None:
            print("Rejected: missing maxSPL_1m")
            continue

        print("Accepted")
        filtered.append(it)

    print("\nFiltered items count:", len(filtered))

    if filtered:
        print("First filtered item:", filtered[0])
    else:
        print("No filtered items")

    if not filtered:
        print("ERROR: no suitable models after filtering")
        await update.message.reply_text(
            f"Нет подходящих моделей под тип помещения='{room_type}'.\n"
            f"Допустимые типы для помещения: {', '.join(sorted(allowed_types))}."
        )
        return

    print("SUCCESS: catalog loaded and filtered correctly")
    print("===== CALC REQUEST END =====\n")

    # Calculate per model, group by type
    results_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for it in filtered:
        stype = (it.get("type") or "").lower().strip()
        calc = calc_for_model(S, L_target, room_type, float(it["maxSPL_1m"]))
        entry = {**it, **calc}
        results_by_type.setdefault(stype, []).append(entry)

    # Choose best models per type (1-3) and sort types by mount preference
    mount = params.get("mount", "")

    type_order = sorted(
        results_by_type.keys(),
        key=lambda t: (_mount_preference_rank(mount, t), t)
    )

    lines: List[str] = []
    lines.append("🧮 Calculator — подбор IP-громкоговорителей (SPL)\n")
    lines.append(f"Вводные: S={S} м², H={params['H']} м, тип={room_type}, шум={noise} дБ → L_target={L_target} дБ (+15)")
    if mount:
        lines.append(f"Монтаж (предпочтение): {mount}")
    lines.append("")

    for stype in type_order:
        entries = results_by_type.get(stype, [])
        if not entries:
            continue
        entries_sorted = sorted(entries, key=_calc_sort_key)

        lines.append(f"Тип громкоговорителя: {stype}")

        # Show up to 3 model options within the type
        for e in entries_sorted[:3]:
            model = e.get("model")
            maxspl = e.get("maxSPL_1m")
            n = int(e.get("N", 0))
            step = e.get("step")
            poe_1 = e.get("P_poe")

            if poe_1 is None:
                poe_str = "нет данных"
            else:
                poe_total = float(poe_1) * n
                poe_str = f"{poe_1} Вт / {round(poe_total, 2)} Вт"

            lines.append(f"- Модель: {model}")
            lines.append(f"  maxSPL_1m: {maxspl} дБ")
            lines.append(f"  Количество громкоговорителей: {n} шт")
            lines.append(f"  Рекомендованный шаг установки: ~{step} м")
            lines.append(f"  PoE-потребление (1 шт / всего): {poe_str}")
        lines.append("")

    out = "\n".join(lines).strip()
    if len(out) > 3900:
        out = out[:3900] + "\n\n(сообщение обрезано)"
    await update.message.reply_text(out)


# ---------------------------
# Telegram handlers
# ---------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return

    await update.message.reply_text(
        "Привет! Я бот-помощник для менеджеров.\n"
        "Отвечаю строго по внутренним документам.\n\n"
        "Команды:\n"
        "/mode — показать режим\n"
        "/mode presale|client|diag|short — сменить режим\n"
        "/whoami — узнать твой user_id\n"
        "/calc — включить калькулятор (можно и просто отправить параметры S/H/тип)\n"
        "/calc_help — пример ввода\n"
        "/calc_stop — выключить калькулятор\n"
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return

    user = update.effective_user
    chat = update.effective_chat
    uname = f"@{user.username}" if user and user.username else "(нет)"
    await update.message.reply_text(
        f"user_id: {user.id}\nusername: {uname}\nchat_id: {chat.id}\nchat_type: {chat.type}"
    )


async def mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def calculator_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return
    _set_calculator(context, True)
    await update.message.reply_text(
        "🧮 Calculator включён.\n"
        "Пришли параметры одним сообщением, пример:\n"
        "S=2400\nH=3.5\nтип=склад\nшум=85\nконтент=только речь\nпрепятствия=перегородки\nмонтаж=стена\n\n"
        "Команды: /calc_help, /calc_stop"
    )


async def calc_help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return
    await update.message.reply_text(
        "🧮 Формат ввода (этаж не нужен):\n"
        "S=площадь_м2\nH=высота_м\nтип=офис|коридор|склад|цех|улица\nшум=дБ (опционально)\n"
        "контент=только речь|музыка\nпрепятствия=открытое пространство|перегородки|стеллажи|оборудование\n"
        "монтаж=потолок|стена|колонна|любой (опционально, как предпочтение)\n\n"
        "Пример:\nS=20000\nH=3.5\nтип=склад\nшум=85\nконтент=только речь\nпрепятствия=перегородки\nмонтаж=стена"
    )


async def calc_stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return
    _set_calculator(context, False)
    await update.message.reply_text("Calculator выключён. Можешь задавать обычные вопросы.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    if not _is_allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return

    text = update.message.text.strip()

    # Auto-detect calculator input (stable even if mode lost after restart)
    if _looks_like_calc_input(text) or _in_calculator(context):
        await handle_calculator_message(update, context)
        return

    # Normal Q&A flow
    question = text
    if len(question) < 2:
        return
    if len(question) > 4000:
        await update.message.reply_text("Слишком длинный запрос. Сократи до 1–2 абзацев.")
        return

    try:
        mode = _get_mode(context)
        facts_blob = extract_facts(question)
        await notify_admin_if_not_found(context, question, facts_blob)
        answer = compose_answer(facts_blob, mode, question)

        if len(answer) > 3900:
            answer = answer[:3900] + "\n\n(сообщение обрезано)"

        await update.message.reply_text(answer)
    except Exception as e:
        logger.exception("Error while handling message")
        await update.message.reply_text(f"Ошибка при обработке запроса: {e}")


# ---------------------------
# Main
# ---------------------------
def main() -> None:
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

    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is starting (polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
