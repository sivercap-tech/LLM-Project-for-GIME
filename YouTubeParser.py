"""
LLM-фильтр видео о мигрантах в России — Gemini API (улучшенная версия)
Вход:  ~/migrants_research/to_filter.json
Выход: ~/migrants_research/data/index.json
       ~/migrants_research/data/videos_YYYY.json
       ~/migrants_research/accepted_videos.json
       ~/migrants_research/uncertain_videos.json

Новое по сравнению с предыдущей версией:
- Свободные фреймы (модель сама определяет)
- Тип канала (гос, оппозиция, диаспора, блогер...)
- География (город/регион если упомянут)
- Ключевые слова из заголовка/описания
- Субъект видео (про кого)
- Нормализация tone на русский
"""

import json
import time
import logging
import os
from pathlib import Path

import google.generativeai as genai

# ── Настройки ─────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyB9yXLrsRs86qr9pFrmaBPPw4NgKO1yrzo")
genai.configure(api_key=GEMINI_API_KEY)

BASE_DIR   = Path.home() / "migrants_research"
OUTPUT_DIR = BASE_DIR / "data"
INPUT_FILE = BASE_DIR / "to_filter.json"

BASE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gemini-2.5-flash-lite"
API_DELAY  = 7.0    # 10 RPM → минимум 6 сек, берём 7 для надёжности
SAVE_EVERY = 10     # сохранять каждые N видео

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

model = genai.GenerativeModel(MODEL_NAME)

JSON_CONFIG = genai.types.GenerationConfig(
    temperature=0.1,
    max_output_tokens=300,
    response_mime_type="application/json",
)


# ── Промпт ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
Ты помогаешь отбирать YouTube-видео для академического исследования дискурса \
о трудовых мигрантах в России (2015–2025).

Видео РЕЛЕВАНТНО если оно про:
- трудовую миграцию в Россию (из Средней Азии, СНГ)
- положение мигрантов в российском обществе
- отношение россиян к мигрантам
- криминальные новости с участием мигрантов
- политику в отношении мигрантов в РФ

Видео НЕ РЕЛЕВАНТНО если:
- про эмиграцию россиян за рубеж
- про беженцев в Европе или других странах
- тема мигрантов упомянута вскользь, основная тема другая
- художественный фильм, музыкальный клип, игра

Оцени видео:
Заголовок: {title}
Описание: {description}
Теги: {tags}
Канал: {channel_title}

Верни JSON со всеми полями:
{{
  "relevant": true или false,
  "certainty": "high" или "medium" или "low",

  "frame": "короткая фраза на русском — главная тема видео своими словами",

  "tone": "негативный" или "нейтральный" или "позитивный",

  "subject": "мигранты как группа" или "конкретная национальность" или
             "российские власти" или "работодатели" или "общество" или "иное",

  "channel_type": "государственный" или "оппозиционный" или "диаспорный" или
                  "новостной" или "блогер" или "неизвестно",

  "geography": "город или регион если упомянут в заголовке или описании, иначе null",

  "keywords": ["3-5 ключевых слов из заголовка и описания на русском"],

  "reason": "одно предложение на русском почему релевантно или нет"
}}\
"""


# ── Нормализация ──────────────────────────────────────────────────────────────

TONE_MAP = {
    "neutral":   "нейтральный",
    "negative":  "негативный",
    "positive":  "позитивный",
    "нейтральный": "нейтральный",
    "негативный":  "негативный",
    "позитивный":  "позитивный",
}

def normalize_tone(tone: str | None) -> str | None:
    if tone is None:
        return None
    return TONE_MAP.get(tone.lower().strip(), tone)


# ── JSON-хранилище ────────────────────────────────────────────────────────────

def load_index() -> dict[str, dict]:
    p = OUTPUT_DIR / "index.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return {v["video_id"]: v for v in json.load(f)}
    return {}


def save_index(index: dict[str, dict]):
    p = OUTPUT_DIR / "index.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(list(index.values()), f, ensure_ascii=False, indent=2)


def update_year_files(updated_videos: list[dict]):
    from collections import defaultdict
    by_year: dict[int, list] = defaultdict(list)
    for v in updated_videos:
        if v.get("year"):
            by_year[v["year"]].append(v)

    for year, videos in by_year.items():
        p = OUTPUT_DIR / f"videos_{year}.json"
        existing: dict[str, dict] = {}
        if p.exists():
            with open(p, encoding="utf-8") as f:
                existing = {v["video_id"]: v for v in json.load(f)}
        for v in videos:
            existing[v["video_id"]] = v
        with open(p, "w", encoding="utf-8") as f:
            json.dump(list(existing.values()), f, ensure_ascii=False, indent=2)


# ── Gemini-классификация ──────────────────────────────────────────────────────

def classify_one(video: dict) -> dict:
    """Классифицирует одно видео. Возвращает dict с llm-полями."""
    prompt = PROMPT_TEMPLATE.format(
        title=video.get("title", ""),
        description=video.get("description", "")[:300],
        tags=", ".join(video.get("tags", [])[:10]),
        channel_title=video.get("channel_title", ""),
    )

    try:
        response = model.generate_content(prompt, generation_config=JSON_CONFIG)
        result = json.loads(response.text)
        return {
            "relevant":     result.get("relevant"),
            "certainty":    result.get("certainty"),
            "frame":        result.get("frame"),
            "tone":         normalize_tone(result.get("tone")),
            "subject":      result.get("subject"),
            "channel_type": result.get("channel_type"),
            "geography":    result.get("geography"),
            "keywords":     result.get("keywords", []),
            "reason":       result.get("reason"),
        }

    except json.JSONDecodeError:
        log.warning(f"JSON parse error для {video.get('video_id')}: {response.text!r:.80}")
        return _empty_result("parse error")

    except Exception as e:
        log.error(f"API error для {video.get('video_id')}: {e}")
        return _empty_result(f"api error: {e}")


def _empty_result(reason: str) -> dict:
    return {
        "relevant":     None,
        "certainty":    "low",
        "frame":        None,
        "tone":         None,
        "subject":      None,
        "channel_type": None,
        "geography":    None,
        "keywords":     [],
        "reason":       reason,
    }


# ── Тестовый режим ────────────────────────────────────────────────────────────

def run_test(n: int = 3):
    """
    Прогоняет N видео через Gemini и печатает результат.
    Ничего не сохраняет. При лимите 20 RPD рекомендуем n=3.
    """
    if not INPUT_FILE.exists():
        log.error(f"Файл не найден: {INPUT_FILE}")
        return

    with open(INPUT_FILE, encoding="utf-8") as f:
        sample = json.load(f)[:n]

    log.info(f"Тест на {len(sample)} видео (модель: {MODEL_NAME})\n")

    for i, v in enumerate(sample, 1):
        print(f"{'─' * 60}")
        print(f"[{i}/{n}] {v.get('title', '')[:70]}")
        print(f"Год: {v.get('year')}  Канал: {v.get('channel_title', '')[:40]}")

        result = classify_one(v)

        print(f"relevant={result.get('relevant')}  "
              f"certainty={result.get('certainty')}")
        print(f"frame:        {result.get('frame')}")
        print(f"tone:         {result.get('tone')}")
        print(f"subject:      {result.get('subject')}")
        print(f"channel_type: {result.get('channel_type')}")
        print(f"geography:    {result.get('geography')}")
        print(f"keywords:     {result.get('keywords')}")
        print(f"reason:       {result.get('reason')}")

        if i < n:
            print(f"  (пауза {API_DELAY} сек...)")
            time.sleep(API_DELAY)

    print(f"\n{'─' * 60}")
    print(f"Тест завершён. Использовано {n} из 20 дневных запросов.")
    print("Если результаты корректны — запускай run().")


# ── Главный пайплайн ──────────────────────────────────────────────────────────

def run(input_file: Path = INPUT_FILE):
    if not input_file.exists():
        log.error(f"Файл не найден: {input_file}")
        return

    with open(input_file, encoding="utf-8") as f:
        to_filter: list[dict] = json.load(f)

    log.info(f"Загружено {len(to_filter)} видео")
    log.info(f"При 20 RPD это займёт ~{len(to_filter) // 20} дней")

    index       = load_index()
    processed   = 0
    updated_buf = []

    for v in to_filter:
        llm = classify_one(v)
        vid = v["video_id"]

        if vid in index:
            index[vid]["llm_relevant"]     = llm["relevant"]
            index[vid]["llm_certainty"]    = llm["certainty"]
            index[vid]["llm_frame"]        = llm["frame"]
            index[vid]["llm_tone"]         = llm["tone"]
            index[vid]["llm_subject"]      = llm["subject"]
            index[vid]["llm_channel_type"] = llm["channel_type"]
            index[vid]["llm_geography"]    = llm["geography"]
            index[vid]["llm_keywords"]     = llm["keywords"]
            index[vid]["llm_reason"]       = llm["reason"]
            updated_buf.append(index[vid])

        processed += 1
        log.info(
            f"  [{processed}/{len(to_filter)}] "
            f"relevant={llm['relevant']}  "
            f"frame={llm['frame']}  "
            f"{v.get('title', '')[:45]}"
        )

        if processed % SAVE_EVERY == 0:
            save_index(index)
            update_year_files(updated_buf)
            updated_buf = []
            log.info(f"  Сохранено ({processed} видео)")

        time.sleep(API_DELAY)

    save_index(index)
    update_year_files(updated_buf)
    log.info("\nГотово!")
    print_results(index)
    export_accepted(index)
    export_uncertain(index)


# ── Статистика ────────────────────────────────────────────────────────────────

def print_results(index: dict[str, dict]):
    from collections import defaultdict, Counter

    by_year: dict[int, list] = defaultdict(list)
    for v in index.values():
        if v.get("year"):
            by_year[v["year"]].append(v)

    print(f"\n{'Год':<6} {'Всего':>6} {'Принято':>8} {'Отклонено':>10} {'Неясно':>8}")
    print("-" * 44)
    total_accepted = 0
    for yr in sorted(by_year):
        vids     = by_year[yr]
        accepted = sum(1 for v in vids if v.get("llm_relevant") is True
                       and v.get("llm_certainty") == "high")
        rejected = sum(1 for v in vids if v.get("llm_relevant") is False
                       and v.get("llm_certainty") == "high")
        unclear  = len(vids) - accepted - rejected
        total_accepted += accepted
        print(f"{yr:<6} {len(vids):>6} {accepted:>8} {rejected:>10} {unclear:>8}")

    print(f"\nИтого принято (high certainty): {total_accepted}")

    # Топ фреймов
    frames = Counter(
        v.get("llm_frame")
        for v in index.values()
        if v.get("llm_relevant") is True and v.get("llm_frame")
    )
    if frames:
        print("\nТоп фреймов:")
        for frame, cnt in frames.most_common(10):
            print(f"  {frame:<30} {cnt}")

    # Типы каналов
    channel_types = Counter(
        v.get("llm_channel_type")
        for v in index.values()
        if v.get("llm_relevant") is True and v.get("llm_channel_type")
    )
    if channel_types:
        print("\nТипы каналов:")
        for ct, cnt in channel_types.most_common():
            print(f"  {ct:<20} {cnt}")

    # Топ географий
    geos = Counter(
        v.get("llm_geography")
        for v in index.values()
        if v.get("llm_relevant") is True
        and v.get("llm_geography")
        and v.get("llm_geography") != "null"
    )
    if geos:
        print("\nТоп географий:")
        for geo, cnt in geos.most_common(10):
            print(f"  {geo:<20} {cnt}")


# ── Экспорт ───────────────────────────────────────────────────────────────────

def export_accepted(index: dict[str, dict]):
    """Финальный датасет — релевантные с высокой уверенностью."""
    accepted = [
        v for v in index.values()
        if v.get("llm_relevant") is True
        and v.get("llm_certainty") == "high"
    ]
    out_path = BASE_DIR / "accepted_videos.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(accepted, f, ensure_ascii=False, indent=2)
    log.info(f"Принятые ({len(accepted)}) → {out_path}")


def export_uncertain(index: dict[str, dict]):
    """Неоднозначные — для ручной проверки."""
    uncertain = [
        v for v in index.values()
        if v.get("llm_certainty") in ("medium", "low")
        or v.get("llm_relevant") is None
    ]
    out_path = BASE_DIR / "uncertain_videos.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(uncertain, f, ensure_ascii=False, indent=2)
    log.info(f"Неоднозначные ({len(uncertain)}) → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()  "frame": "короткая фраза на русском — главная тема видео своими словами",

  "tone": "негативный" или "нейтральный" или "позитивный",

  "subject": "мигранты как группа" или "конкретная национальность" или
             "российские власти" или "работодатели" или "общество" или "иное",

  "channel_type": "государственный" или "оппозиционный" или "диаспорный" или
                  "новостной" или "блогер" или "неизвестно",

  "geography": "город или регион если упомянут в заголовке или описании, иначе null",

  "keywords": ["3-5 ключевых слов из заголовка и описания на русском"],

  "reason": "одно предложение на русском почему релевантно или нет"
}}\
"""


# ── Нормализация ──────────────────────────────────────────────────────────────

TONE_MAP = {
    "neutral":   "нейтральный",
    "negative":  "негативный",
    "positive":  "позитивный",
    "нейтральный": "нейтральный",
    "негативный":  "негативный",
    "позитивный":  "позитивный",
}

def normalize_tone(tone: str | None) -> str | None:
    if tone is None:
        return None
    return TONE_MAP.get(tone.lower().strip(), tone)


# ── JSON-хранилище ────────────────────────────────────────────────────────────

def load_index() -> dict[str, dict]:
    p = OUTPUT_DIR / "index.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return {v["video_id"]: v for v in json.load(f)}
    return {}


def save_index(index: dict[str, dict]):
    p = OUTPUT_DIR / "index.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(list(index.values()), f, ensure_ascii=False, indent=2)


def update_year_files(updated_videos: list[dict]):
    from collections import defaultdict
    by_year: dict[int, list] = defaultdict(list)
    for v in updated_videos:
        if v.get("year"):
            by_year[v["year"]].append(v)

    for year, videos in by_year.items():
        p = OUTPUT_DIR / f"videos_{year}.json"
        existing: dict[str, dict] = {}
        if p.exists():
            with open(p, encoding="utf-8") as f:
                existing = {v["video_id"]: v for v in json.load(f)}
        for v in videos:
            existing[v["video_id"]] = v
        with open(p, "w", encoding="utf-8") as f:
            json.dump(list(existing.values()), f, ensure_ascii=False, indent=2)


# ── Gemini-классификация ──────────────────────────────────────────────────────

def classify_one(video: dict) -> dict:
    """Классифицирует одно видео. Возвращает dict с llm-полями."""
    prompt = PROMPT_TEMPLATE.format(
        title=video.get("title", ""),
        description=video.get("description", "")[:300],
        tags=", ".join(video.get("tags", [])[:10]),
        channel_title=video.get("channel_title", ""),
    )

    try:
        response = model.generate_content(prompt, generation_config=JSON_CONFIG)
        result = json.loads(response.text)
        return {
            "relevant":     result.get("relevant"),
            "certainty":    result.get("certainty"),
            "frame":        result.get("frame"),
            "tone":         normalize_tone(result.get("tone")),
            "subject":      result.get("subject"),
            "channel_type": result.get("channel_type"),
            "geography":    result.get("geography"),
            "keywords":     result.get("keywords", []),
            "reason":       result.get("reason"),
        }

    except json.JSONDecodeError:
        log.warning(f"JSON parse error для {video.get('video_id')}: {response.text!r:.80}")
        return _empty_result("parse error")

    except Exception as e:
        log.error(f"API error для {video.get('video_id')}: {e}")
        return _empty_result(f"api error: {e}")


def _empty_result(reason: str) -> dict:
    return {
        "relevant":     None,
        "certainty":    "low",
        "frame":        None,
        "tone":         None,
        "subject":      None,
        "channel_type": None,
        "geography":    None,
        "keywords":     [],
        "reason":       reason,
    }


# ── Тестовый режим ────────────────────────────────────────────────────────────

def run_test(n: int = 3):
    """
    Прогоняет N видео через Gemini и печатает результат.
    Ничего не сохраняет. При лимите 20 RPD рекомендуем n=3.
    """
    if not INPUT_FILE.exists():
        log.error(f"Файл не найден: {INPUT_FILE}")
        return

    with open(INPUT_FILE, encoding="utf-8") as f:
        sample = json.load(f)[:n]

    log.info(f"Тест на {len(sample)} видео (модель: {MODEL_NAME})\n")

    for i, v in enumerate(sample, 1):
        print(f"{'─' * 60}")
        print(f"[{i}/{n}] {v.get('title', '')[:70]}")
        print(f"Год: {v.get('year')}  Канал: {v.get('channel_title', '')[:40]}")

        result = classify_one(v)

        print(f"relevant={result.get('relevant')}  "
              f"certainty={result.get('certainty')}")
        print(f"frame:        {result.get('frame')}")
        print(f"tone:         {result.get('tone')}")
        print(f"subject:      {result.get('subject')}")
        print(f"channel_type: {result.get('channel_type')}")
        print(f"geography:    {result.get('geography')}")
        print(f"keywords:     {result.get('keywords')}")
        print(f"reason:       {result.get('reason')}")

        if i < n:
            print(f"  (пауза {API_DELAY} сек...)")
            time.sleep(API_DELAY)

    print(f"\n{'─' * 60}")
    print(f"Тест завершён. Использовано {n} из 20 дневных запросов.")
    print("Если результаты корректны — запускай run().")


# ── Главный пайплайн ──────────────────────────────────────────────────────────

def run(input_file: Path = INPUT_FILE):
    if not input_file.exists():
        log.error(f"Файл не найден: {input_file}")
        return

    with open(input_file, encoding="utf-8") as f:
        to_filter: list[dict] = json.load(f)

    log.info(f"Загружено {len(to_filter)} видео")
    log.info(f"При 20 RPD это займёт ~{len(to_filter) // 20} дней")

    index       = load_index()
    processed   = 0
    updated_buf = []

    for v in to_filter:
        llm = classify_one(v)
        vid = v["video_id"]

        if vid in index:
            index[vid]["llm_relevant"]     = llm["relevant"]
            index[vid]["llm_certainty"]    = llm["certainty"]
            index[vid]["llm_frame"]        = llm["frame"]
            index[vid]["llm_tone"]         = llm["tone"]
            index[vid]["llm_subject"]      = llm["subject"]
            index[vid]["llm_channel_type"] = llm["channel_type"]
            index[vid]["llm_geography"]    = llm["geography"]
            index[vid]["llm_keywords"]     = llm["keywords"]
            index[vid]["llm_reason"]       = llm["reason"]
            updated_buf.append(index[vid])

        processed += 1
        log.info(
            f"  [{processed}/{len(to_filter)}] "
            f"relevant={llm['relevant']}  "
            f"frame={llm['frame']}  "
            f"{v.get('title', '')[:45]}"
        )

        if processed % SAVE_EVERY == 0:
            save_index(index)
            update_year_files(updated_buf)
            updated_buf = []
            log.info(f"  Сохранено ({processed} видео)")

        time.sleep(API_DELAY)

    save_index(index)
    update_year_files(updated_buf)
    log.info("\nГотово!")
    print_results(index)
    export_accepted(index)
    export_uncertain(index)


# ── Статистика ────────────────────────────────────────────────────────────────

def print_results(index: dict[str, dict]):
    from collections import defaultdict, Counter

    by_year: dict[int, list] = defaultdict(list)
    for v in index.values():
        if v.get("year"):
            by_year[v["year"]].append(v)

    print(f"\n{'Год':<6} {'Всего':>6} {'Принято':>8} {'Отклонено':>10} {'Неясно':>8}")
    print("-" * 44)
    total_accepted = 0
    for yr in sorted(by_year):
        vids     = by_year[yr]
        accepted = sum(1 for v in vids if v.get("llm_relevant") is True
                       and v.get("llm_certainty") == "high")
        rejected = sum(1 for v in vids if v.get("llm_relevant") is False
                       and v.get("llm_certainty") == "high")
        unclear  = len(vids) - accepted - rejected
        total_accepted += accepted
        print(f"{yr:<6} {len(vids):>6} {accepted:>8} {rejected:>10} {unclear:>8}")

    print(f"\nИтого принято (high certainty): {total_accepted}")

    # Топ фреймов
    frames = Counter(
        v.get("llm_frame")
        for v in index.values()
        if v.get("llm_relevant") is True and v.get("llm_frame")
    )
    if frames:
        print("\nТоп фреймов:")
        for frame, cnt in frames.most_common(10):
            print(f"  {frame:<30} {cnt}")

    # Типы каналов
    channel_types = Counter(
        v.get("llm_channel_type")
        for v in index.values()
        if v.get("llm_relevant") is True and v.get("llm_channel_type")
    )
    if channel_types:
        print("\nТипы каналов:")
        for ct, cnt in channel_types.most_common():
            print(f"  {ct:<20} {cnt}")

    # Топ географий
    geos = Counter(
        v.get("llm_geography")
        for v in index.values()
        if v.get("llm_relevant") is True
        and v.get("llm_geography")
        and v.get("llm_geography") != "null"
    )
    if geos:
        print("\nТоп географий:")
        for geo, cnt in geos.most_common(10):
            print(f"  {geo:<20} {cnt}")


# ── Экспорт ───────────────────────────────────────────────────────────────────

def export_accepted(index: dict[str, dict]):
    """Финальный датасет — релевантные с высокой уверенностью."""
    accepted = [
        v for v in index.values()
        if v.get("llm_relevant") is True
        and v.get("llm_certainty") == "high"
    ]
    out_path = BASE_DIR / "accepted_videos.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(accepted, f, ensure_ascii=False, indent=2)
    log.info(f"Принятые ({len(accepted)}) → {out_path}")


def export_uncertain(index: dict[str, dict]):
    """Неоднозначные — для ручной проверки."""
    uncertain = [
        v for v in index.values()
        if v.get("llm_certainty") in ("medium", "low")
        or v.get("llm_relevant") is None
    ]
    out_path = BASE_DIR / "uncertain_videos.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(uncertain, f, ensure_ascii=False, indent=2)
    log.info(f"Неоднозначные ({len(uncertain)}) → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()    ("2016-01-01T00:00:00Z", "2016-12-31T23:59:59Z"),
    ("2017-01-01T00:00:00Z", "2017-12-31T23:59:59Z"),
    ("2018-01-01T00:00:00Z", "2018-12-31T23:59:59Z"),
    ("2019-01-01T00:00:00Z", "2019-12-31T23:59:59Z"),
    ("2020-01-01T00:00:00Z", "2020-12-31T23:59:59Z"),
    ("2021-01-01T00:00:00Z", "2021-12-31T23:59:59Z"),
    ("2022-01-01T00:00:00Z", "2022-12-31T23:59:59Z"),
    ("2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z"),
    ("2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z"),
    ("2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z"),
]

MAX_RESULTS_PER_QUERY = 50
TRANSCRIPT_LANGS = ["ru", "ru-RU"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

def empty_video(video_id: str) -> dict:
    return {
        "video_id":        video_id,
        "title":           "",
        "description":     "",
        "tags":            [],
        "channel_id":      "",
        "channel_title":   "",
        "published_at":    "",
        "year":            None,
        "view_count":      None,
        "like_count":      None,
        "comment_count":   None,
        "duration":        "",
        "search_query":    "",
        "transcript":      None,       
        "transcript_type": None,       
        # LLM-разметка
        "llm_relevant":    None,
        "llm_certainty":   None,       
        "llm_frame":       None,      
        "llm_tone":        None,       
        "llm_reason":      None,
    }

def load_index() -> dict[str, dict]:
    """Единый индекс всех видео — для дедупликации между запросами."""
    p = OUTPUT_DIR / "index.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return {v["video_id"]: v for v in json.load(f)}
    return {}


def save_index(index: dict[str, dict]):
    p = OUTPUT_DIR / "index.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(list(index.values()), f, ensure_ascii=False, indent=2)


def upsert_year_file(year: int, new_videos: list[dict]):
    """
    Дописывает новые видео в срез по году.
    Срезы по годам — основной вход для эмбеддинг-модели.
    """
    p = OUTPUT_DIR / f"videos_{year}.json"
    existing: dict[str, dict] = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            existing = {v["video_id"]: v for v in json.load(f)}

    for v in new_videos:
        existing[v["video_id"]] = v

    with open(p, "w", encoding="utf-8") as f:
        json.dump(list(existing.values()), f, ensure_ascii=False, indent=2)

    return len(existing)

def build_youtube():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def search_videos(youtube, query: str, published_after: str, published_before: str) -> list[dict]:
    """
    Поиск видео + запрос статистики.
    Квота: 100 (search.list) + 1 (videos.list) = ~101 единица.
    """
    try:
        resp = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            publishedAfter=published_after,
            publishedBefore=published_before,
            maxResults=MAX_RESULTS_PER_QUERY,
            relevanceLanguage="ru",
            regionCode="RU",
            order="relevance",
        ).execute()
    except HttpError as e:
        log.error(f"search.list error '{query}': {e}")
        return []

    items = resp.get("items", [])
    if not items:
        return []

    video_ids = [it["id"]["videoId"] for it in items]
    stats = _fetch_statistics(youtube, video_ids)

    videos = []
    for it in items:
        vid = it["id"]["videoId"]
        sn  = it["snippet"]
        st  = stats.get(vid, {})

        v = empty_video(vid)
        v.update({
            "title":         sn.get("title", ""),
            "description":   sn.get("description", ""),
            "tags":          st.get("tags", []),
            "channel_id":    sn.get("channelId", ""),
            "channel_title": sn.get("channelTitle", ""),
            "published_at":  sn.get("publishedAt", ""),
            "year":          int(sn.get("publishedAt", "0000")[:4]),
            "view_count":    _to_int(st.get("viewCount")),
            "like_count":    _to_int(st.get("likeCount")),
            "comment_count": _to_int(st.get("commentCount")),
            "duration":      st.get("duration", ""),
            "search_query":  query,
        })
        videos.append(v)

    return videos


def _fetch_statistics(youtube, video_ids: list[str]) -> dict[str, dict]:
    try:
        resp = youtube.videos().list(
            id=",".join(video_ids),
            part="statistics,contentDetails,snippet",
        ).execute()
    except HttpError as e:
        log.error(f"videos.list error: {e}")
        return {}

    out = {}
    for it in resp.get("items", []):
        st = it.get("statistics", {})
        sn = it.get("snippet", {})
        out[it["id"]] = {
            "viewCount":    st.get("viewCount"),
            "likeCount":    st.get("likeCount"),
            "commentCount": st.get("commentCount"),
            "duration":     it.get("contentDetails", {}).get("duration"),
            "tags":         sn.get("tags", []),
        }
    return out


def _to_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

def fetch_transcript(video_id: str) -> tuple[str | None, str | None]:
    """
    Приоритет:
      1. Ручные русские субтитры  → тип "manual"
      2. Авто-генерированные рус. → тип "auto"
      3. Любые доступные          → тип "auto_XX"
    Возвращает (текст, тип) или (None, None).
    """
    try:
        tlist = YouTubeTranscriptApi.list_transcripts(video_id)

        for fetch_fn, label in [
            (lambda: tlist.find_manually_created_transcript(TRANSCRIPT_LANGS), "manual"),
            (lambda: tlist.find_generated_transcript(TRANSCRIPT_LANGS),        "auto"),
        ]:
            try:
                t    = fetch_fn()
                text = _segments_to_text(t.fetch())
                return text, label
            except Exception:
                continue

      t    = next(iter(tlist))
        text = _segments_to_text(t.fetch())
        return text, f"auto_{t.language_code}"

    except (NoTranscriptFound, TranscriptsDisabled):
        return None, None
    except Exception as e:
        log.debug(f"Transcript {video_id}: {e}")
        return None, None


def _segments_to_text(segments: list[dict]) -> str:
    return " ".join(
        seg.get("text", "").replace("\n", " ").strip()
        for seg in segments
        if seg.get("text", "").strip()
    )

def run(
    queries: list[str]       = SEARCH_QUERIES,
    year_ranges: list[tuple] = YEAR_RANGES,
    fetch_transcripts: bool  = True,
    api_delay: float         = 0.5,
):
    index   = load_index()
    youtube = build_youtube()
    total_new = 0

    for year_from, year_to in year_ranges:
        year     = int(year_from[:4])
        year_new = []
        log.info(f"── {year} ────────────────────────────")

        for query in queries:
            videos = search_videos(youtube, query, year_from, year_to)

            for v in videos:
                if v["video_id"] in index:
                    continue                    # дедупликация

                if fetch_transcripts:
                    text, ttype = fetch_transcript(v["video_id"])
                    v["transcript"]      = text
                    v["transcript_type"] = ttype
                    time.sleep(0.2)

                index[v["video_id"]] = v
                year_new.append(v)
                total_new += 1

            log.info(
                f"  [{year}] '{query[:42]}' "
                f"→ {len(videos)} найдено, {sum(1 for v in year_new if v['transcript'])} с субтитрами"
            )
            time.sleep(api_delay)

        if year_new:
            n = upsert_year_file(year, year_new)
            log.info(f"  [{year}] videos_{year}.json — всего в файле: {n}")

    save_index(index)
    log.info(f"\nГотово. Новых видео: {total_new}")
    _print_stats(index)


def _print_stats(index: dict[str, dict]):
    from collections import defaultdict
    by_year: dict[int, list] = defaultdict(list)
    for v in index.values():
        by_year[v.get("year")].append(v)

    print(f"\n{'Год':<6} {'Видео':>6} {'Субтитры':>10} {'Ручные':>8} {'Без LLM':>9}")
    print("-" * 44)
    for yr in sorted(by_year):
        vids  = by_year[yr]
        tr    = sum(1 for v in vids if v.get("transcript"))
        manual= sum(1 for v in vids if v.get("transcript_type") == "manual")
        nollm = sum(1 for v in vids if v.get("llm_relevant") is None)
        print(f"{yr:<6} {len(vids):>6} {tr:>10} {manual:>8} {nollm:>9}")

    print(f"\nВсего: {len(index)}")
    print(f"\nФайлы:")
    print(f"  data/index.json          — полный индекс всех видео")
    print(f"  data/videos_YYYY.json    — срезы по годам (→ эмбеддинги)")

def export_for_llm_filter():
    out_path = BASE_DIR / "to_filter.json"
    index = load_index()
    batch = [
        {
            "video_id":    v["video_id"],
            "title":       v["title"],
            "description": v["description"][:300],
            "tags":        v["tags"][:10],
            "year":        v["year"],
        }
        for v in index.values()
        if v.get("llm_relevant") is None
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
    log.info(f"Экспортировано {len(batch)} видео → {out_path}")


if __name__ == "__main__":
    run()
    export_for_llm_filter()
