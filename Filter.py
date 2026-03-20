"""
LLM-фильтр видео о мигрантах в России — Gemini API
Вход:  ~/migrants_research/to_filter.json
Выход: ~/migrants_research/data/index.json
       ~/migrants_research/data/videos_YYYY.json
       ~/migrants_research/accepted_videos.json
       ~/migrants_research/uncertain_videos.json
"""

import json
import time
import logging
import os
from pathlib import Path

import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "XXX")
genai.configure(api_key=GEMINI_API_KEY)

BASE_DIR   = Path.home() / "migrants_research"
OUTPUT_DIR = BASE_DIR / "data"
INPUT_FILE = BASE_DIR / "to_filter.json"

BASE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "gemini-2.5-flash-lite"
API_DELAY   = 7.0   # 10 RPM → минимум 6 сек, берём 7 для надёжности
SAVE_EVERY  = 10    # сохранять каждые N видео (лимит маленький, сохраняем чаще)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

model = genai.GenerativeModel(MODEL_NAME)

# Конфиг с JSON-режимом — Gemini не оборачивает ответ в ```json```
JSON_CONFIG = genai.types.GenerationConfig(
    temperature=0.1,
    max_output_tokens=200,
    response_mime_type="application/json",
)

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

Верни JSON со всеми полями:
{{
  "relevant": true или false,
  "certainty": "high" или "medium" или "low",
  "frame": "одно слово или короткая фраза на русском",
  "tone": "негативный" или "нейтральный" или "позитивный",
  "reason": "одно предложение на русском почему"
}}\
"""

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

TONE_MAP = {
    "neutral":  "нейтральный",
    "negative": "негативный",
    "positive": "позитивный",
}

def normalize_tone(tone: str | None) -> str | None:
    if tone is None:
        return None
    return TONE_MAP.get(tone.lower(), tone)

def classify_one(video: dict) -> dict:
    """Классифицирует одно видео. Возвращает dict с llm-полями."""
    prompt = PROMPT_TEMPLATE.format(
        title=video.get("title", ""),
        description=video.get("description", "")[:300],
        tags=", ".join(video.get("tags", [])[:10]),
    )

    try:
        response = model.generate_content(prompt, generation_config=JSON_CONFIG)
        result = json.loads(response.text)
        return {
            "relevant":  result.get("relevant"),
            "certainty": result.get("certainty"),
            "frame":     result.get("frame"),
            "tone":      normalize_tone(result.get("tone")),
            "reason":    result.get("reason"),
        }

    except json.JSONDecodeError:
        log.warning(f"JSON parse error для {video['video_id']}: {response.text!r:.80}")
        return {"relevant": None, "certainty": "low", "frame": None, "tone": None, "reason": "parse error"}

    except Exception as e:
        log.error(f"API error для {video['video_id']}: {e}")
        return {"relevant": None, "certainty": "low", "frame": None, "tone": None, "reason": f"api error: {e}"}

def run_test(n: int = 3):
    """
    Прогоняет N видео через Gemini и печатает результат.
    Ничего не сохраняет — только для проверки.
    При лимите 20 RPD рекомендуем n=3.
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
        print(f"Год: {v.get('year')}  ID: {v.get('video_id')}")

        result = classify_one(v)

        print(f"relevant={result.get('relevant')}  "
              f"certainty={result.get('certainty')}  "
              f"frame={result.get('frame')}  "
              f"tone={result.get('tone')}")
        print(f"reason: {result.get('reason')}")

        if i < n:
            print(f"  (пауза {API_DELAY} сек...)")
            time.sleep(API_DELAY)

    print(f"\n{'─' * 60}")
    print(f"Тест завершён. Использовано {n} из 20 дневных запросов.")
    print("Если результаты корректны — запускай run().")


# ── Главный пайплайн ──────────────────────────────────────────────────────────

def run(input_file: Path = INPUT_FILE):
    """
    Фильтрует все видео из input_file.
    При лимите 20 RPD: ~20 видео в день, полный датасет за ~174 дня.
    Рекомендуем сначала сменить модель на gemini-1.5-flash (1500 RPD).
    """
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
            index[vid]["llm_relevant"]  = llm["relevant"]
            index[vid]["llm_certainty"] = llm["certainty"]
            index[vid]["llm_frame"]     = llm["frame"]
            index[vid]["llm_tone"]      = llm["tone"]
            index[vid]["llm_reason"]    = llm["reason"]
            updated_buf.append(index[vid])

        processed += 1
        log.info(
            f"  [{processed}/{len(to_filter)}] "
            f"relevant={llm['relevant']}  certainty={llm['certainty']}  "
            f"{v.get('title', '')[:50]}"
        )

        # Периодическое сохранение
        if processed % SAVE_EVERY == 0:
            save_index(index)
            update_year_files(updated_buf)
            updated_buf = []
            log.info(f"  Сохранено ({processed} видео)")

        time.sleep(API_DELAY)

    # Финальное сохранение
    save_index(index)
    update_year_files(updated_buf)

    log.info("\nГотово!")
    print_results(index)
    export_accepted(index)
    export_uncertain(index)

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

    frames = Counter(
        v.get("llm_frame")
        for v in index.values()
        if v.get("llm_relevant") is True and v.get("llm_frame")
    )
    if frames:
        print("\nФреймы (принятые видео):")
        for frame, cnt in frames.most_common():
            print(f"  {frame:<14} {cnt}")


def export_accepted(index: dict[str, dict]):
    """Финальный датасет — только релевантные с высокой уверенностью."""
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

    run()
