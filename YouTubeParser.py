pip install google-api-python-client youtube-transcript-api

import os
import json
import time
import logging
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "API_KEY")

BASE_DIR = Path.home() / "migrants_research"
BASE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("data")
OUTPUT_DIR = Path.home() / "migrants_research" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_QUERIES = [
    "мигранты в России",
    "трудовые мигранты Россия",
    "гастарбайтеры Россия",
    "таджики в России",
    "узбеки в России",
    "мигранты преступность Россия",
    "депортация мигрантов Россия",
    "нелегальные мигранты Россия",
    "мигранты проблема Россия",
    "приезжие Россия",
]

YEAR_RANGES = [
    ("2015-01-01T00:00:00Z", "2015-12-31T23:59:59Z"),
    ("2016-01-01T00:00:00Z", "2016-12-31T23:59:59Z"),
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
