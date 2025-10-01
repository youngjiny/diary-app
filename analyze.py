# diary_now.py  (Happiness/Sadness/Love/Anger 전용)
from __future__ import annotations
from pathlib import Path
import csv

# ===== 경로 =====
MODEL_DIR = Path("models/kobert-ml")
BOOKS_CSV = Path("data/items_books.csv")
MOVIES_CSV = Path("data/items_movies.csv")
CUSTOM_LEXICON_CSV = Path("data/custom_lexicon.csv")  # columns: stem,emotion

# ===== 고정 감정셋 =====
TARGET = ["Happiness", "Sadness", "Love", "Anger"]

# 과거/소문자/동의 표현 → 표준 감정으로 정규화
EMO_MAP = {
    # 표준형
    "happiness": "Happiness",
    "sadness": "Sadness",
    "love": "Love",
    "anger": "Anger",
    # 과거/동의 레이블
    "joy": "Happiness",
    "happy": "Happiness",
    "sad": "Sadness",
    "mad": "Anger",
    "angry": "Anger",
    # 한글 레이블이 섞여도 대비
    "기쁨": "Happiness",
    "슬픔": "Sadness",
    "사랑": "Love",
    "분노": "Anger",
}
def norm_emo(name: str | None) -> str | None:
    if not name: return None
    return EMO_MAP.get(str(name).strip().lower())

# ===== 최소 내장 사전(씨앗) → custom_lexicon.csv와 병합 사용 =====
LEXICON_BASE = {
    "Happiness": {"행복", "기쁘", "즐겁", "설렘", "유쾌", "환희", "흥겹", "흥겨"},
    "Sadness": {"슬픔", "우울", "허전", "침울", "눈물"},
    "Love": {"사랑", "애정", "좋아", "다정", "연모", "설레"},
    "Anger": {"화가", "분노", "짜증", "억울", "성났", "격분"},
}
NEG = {"안", "못", "아니", "별로", "덜", "부정"}  # 부정 트리거

# ===== 전처리/토크나이즈 =====
def ko_tokenize(s: str) -> list[str]:
    try:
        from konlpy.tag import Okt
        okt = Okt()
        return okt.morphs(s, stem=True)
    except Exception:
        # 매우 단순 백업 토크나이저
        import re
        s = re.sub(r"[\t\r\n.,!?;:\-\(\)\[\]『』“”\"'`·…]", " ", s)
        return [t for t in s.split() if t]

# ===== 커스텀 사전 로딩 (stem,emotion) =====
def load_custom_lexicon(path: Path = CUSTOM_LEXICON_CSV) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    out: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            stem = str(row.get("stem", "")).strip()
            emo_std = norm_emo(row.get("emotion"))
            if stem and emo_std in TARGET:
                out.setdefault(emo_std, set()).add(stem)
    return out

# ===== 사전 기반 스코어 =====
def score_fallback(text: str) -> list[tuple[str, float]]:
    # 최신 커스텀 사전 병합
    merged = {k: set(v) for k, v in LEXICON_BASE.items()}
    custom = load_custom_lexicon()
    for emo, stems in custom.items():
        merged.setdefault(emo, set()).update(stems)

    tokens = ko_tokenize(text)

    # 부정어 직후 window(2) 토큰에 반전/감쇄 적용
    neg_idx: set[int] = set()
    for i, t in enumerate(tokens):
        if any(n in t for n in NEG):
            for j in range(i + 1, min(len(tokens), i + 3)):
                neg_idx.add(j)

    scores = {e: 0.0 for e in TARGET}
    for i, t in enumerate(tokens):
        for emo, stems in merged.items():
            if any(st in t for st in stems):
                scores[emo] += (-0.7 if i in neg_idx else 1.0)

    # 길이 보정
    L = max(len(tokens), 1)
    for k in scores:
        scores[k] = round(scores[k] / (L ** 0.5), 4)

    # 1) 양수 스코어 우선 반환
    positives = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    positives = [(e, s) for e, s in positives[:2] if s > 0]
    if positives:
        return positives

    # 2) 모두 음수/0이면, 가장 큰 '절대값'의 음수를 매핑해서 양수로 환산
    #    - 부정된 Happiness/Love -> Sadness 로 해석(보수적)
    NEG_MAP = {
        "Happiness": "Sadness",
        "Love": "Sadness",
        "Anger": "Anger",     # "안 화났어"는 사실 중립에 가깝지만 4라벨이라 보수적으로 유지
        "Sadness": "Sadness",
    }
    negatives = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
    picked: list[tuple[str, float]] = []
    for emo, val in negatives:
        if val < 0:
            mapped = NEG_MAP.get(emo, emo)
            picked.append((mapped, round(abs(val), 4)))
        if len(picked) == 2:
            break
    return picked

# ===== KoBERT 예측 (있으면 사용) =====
def predict_kobert(text: str, threshold=0.5, topk=3):
    if not MODEL_DIR.exists():
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval()

        t = " ".join(ko_tokenize(text))
        inputs = tok(t, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = mdl(**inputs).logits
            probs = torch.sigmoid(logits).flatten().tolist()

        id2label = getattr(mdl.config, "id2label", None)
        names_raw = [id2label[i] if id2label else f"label_{i}" for i in range(len(probs))]

        # 모델 라벨을 표준 감정으로 정규화 + 중복 id 집계
        agg: dict[str, float] = {}
        for n, p in zip(names_raw, probs):
            std = norm_emo(n)
            if std in TARGET:
                agg[std] = max(agg.get(std, 0.0), float(p))  # max(또는 sum) 선택 가능

        picked = [(e, agg[e]) for e in agg if agg[e] >= threshold]
        if not picked and agg:
            picked = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:topk]
        return sorted(picked, key=lambda x: x[1], reverse=True)[:2] if picked else None
    except Exception:
        return None

# ===== 카탈로그 로드 및 추천 =====
def load_catalog(path: Path):
    if not path.exists(): return None
    items = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # emotions 컬럼 우선, 없으면 tags도 감정 대용으로 시도
            emos_raw = [t.strip() for t in str(row.get("emotions", "")).split(";") if t.strip()]
            if not emos_raw:
                emos_raw = [t.strip() for t in str(row.get("tags", "")).split(";") if t.strip()]
            emos_std = []
            for e in emos_raw:
                std = norm_emo(e)
                if std in TARGET:
                    emos_std.append(std)
            row["emotions_std"] = emos_std
            items.append(row)
    return items

def recommend(picked: list[tuple[str, float]], text: str):
    books_cat = load_catalog(BOOKS_CSV)
    movies_cat = load_catalog(MOVIES_CSV)

    if books_cat and movies_cat:
        w = {e: float(p) for e, p in picked}
        toks = set(ko_tokenize(text))

        def score_item(it: dict) -> float:
            emo_score = sum(w.get(e, 0.0) for e in it.get("emotions_std", []))
            tags = [t.strip() for t in str(it.get("tags", "")).split(";") if t.strip()]
            tag_bonus = sum(1 for t in tags if t in toks) * 0.1
            return emo_score + tag_bonus

        books_sorted = sorted(books_cat, key=score_item, reverse=True)[:5]
        movies_sorted = sorted(movies_cat, key=score_item, reverse=True)[:5]
        return books_sorted, movies_sorted, "catalog"

    # 카탈로그 없으면 내장 폴백
    FALLBACK_BOOKS = {
        "Happiness": ["아주 작은 습관의 힘", "어떤 하루", "걷는 사람, 하정우"],
        "Sadness": ["아몬드", "죽음에 관하여", "보통의 존재"],
        "Love": ["사랑의 기술", "연애의 문장들", "달러구트 꿈 백화점"],
        "Anger": ["감정 어휘", "멈추면, 비로소 보이는 것들", "철학은 어떻게 삶의 무기가 되는가"],
    }
    FALLBACK_MOVIES = {
        "Happiness": ["리틀 포레스트", "라라랜드", "월터의 상상은 현실이 된다"],
        "Sadness": ["맨체스터 바이 더 씨", "문라이트", "이터널 선샤인"],
        "Love": ["비긴 어게인", "비포 선라이즈", "노팅 힐"],
        "Anger": ["위플래쉬", "조커", "베테랑"],
    }

    books, movies = [], []
    for emo, _ in picked:
        books += FALLBACK_BOOKS.get(emo, [])[:3]
        movies += FALLBACK_MOVIES.get(emo, [])[:3]

    def uniq_keep_order(lst: list[str]) -> list[str]:
        seen, out = set(), []
        for x in lst:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return [{"title": b} for b in uniq_keep_order(books)[:5]], [{"title": m} for m in uniq_keep_order(movies)[:5]], "fallback"

# ===== 메인 =====
if __name__ == "__main__":
    print("📝 여러 줄로 일기를 쓰세요. 끝내려면 빈 줄에서 엔터.")
    lines: list[str] = []
    while True:
        line = input()
        if not line.strip(): break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        print("입력 없음. 종료."); raise SystemExit

    picked = predict_kobert(text, threshold=0.5, topk=3)
    method = "KoBERT" if picked else "lexicon"
    if not picked:
        picked = score_fallback(text)

    print(f"\n🔎 분석방법: {method}")
    print("🎯 상위 감정:", [f"{e}:{round(float(s),3)}" for e, s in picked] if picked else "없음")

    books, movies, src = recommend(picked, text)
    print(f"\n📚 책 추천 Top5 ({'카탈로그' if src=='catalog' else '내장 목록'}):")
    for i, b in enumerate(books, 1):
        title = b.get("title", str(b))
        extra = []
        if "author" in b and b["author"]: extra.append(b["author"])
        if "year" in b and b["year"]: extra.append(str(b["year"]))
        print(f" {i}. {title}" + (f" — {', '.join(extra)}" if extra else ""))

    print(f"\n🎬 영화 추천 Top5 ({'카탈로그' if src=='catalog' else '내장 목록'}):")
    for i, m in enumerate(movies, 1):
        title = m.get("title", str(m))
        year = m.get("year", "")
        print(f" {i}. {title}" + (f" ({year})" if year else ""))

    print("")
