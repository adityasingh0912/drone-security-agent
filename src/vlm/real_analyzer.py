"""
vlm/real_analyzer.py
Real VLM analyzer using BLIP (Salesforce/blip-image-captioning-base).
Runs on CPU. Features:
- Frame-level caching: saves BLIP+VQA results to frames_cache.json
- On re-run with same video: loads cache instantly, skips BLIP inference
- Dual-pass captioning for anti-hallucination
- VQA safety checks per frame
"""

import os
import json
import hashlib
from PIL import Image

CACHE_FILE = "frames_cache.json"

# ── Load BLIP ─────────────────────────────────────────────────────────────────

print("[VLM] Loading BLIP model...")
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    MODEL_NAME = "Salesforce/blip-image-captioning-base"
    processor  = BlipProcessor.from_pretrained(MODEL_NAME)
    model      = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    model.eval()
    BLIP_AVAILABLE = True
    print("[VLM] BLIP loaded ✅")
except Exception as e:
    BLIP_AVAILABLE = False
    print(f"[VLM] BLIP unavailable ({e}) — using mock captions")


# ── Object keyword maps ───────────────────────────────────────────────────────

OBJECT_MAP = [
    ("person",      ["person", "people", "man", "woman", "pedestrian",
                     "crowd", "individual", "someone", "walker", "commuter"]),
    ("bicycle",     ["bicycle", "bike", "cycling", "cyclist", "cycle"]),
    ("motorcycle",  ["motorcycle", "motorbike", "scooter", "moped", "tuk"]),
    ("rickshaw",    ["rickshaw", "auto", "tuk-tuk", "three-wheel", "three wheel"]),
    ("car",         ["car", "sedan", "hatchback", "suv", "automobile", "taxi"]),
    ("bus",         ["bus", "minibus", "coach", "double-decker"]),
    ("truck",       ["truck", "lorry", "pickup", "van"]),
    ("traffic",     ["traffic", "road", "street", "highway", "lane"]),
    ("crossing",    ["crossing", "crosswalk", "zebra", "intersection"]),
]

ACTION_KEYWORDS = [
    "walking", "running", "standing", "sitting", "waiting",
    "crossing", "driving", "riding", "loitering", "gathering",
    "entering", "exiting", "parked", "moving", "stopping", "lying",
]

VQA_QUESTIONS = {
    "people_crossing":  "Are there people crossing the street?",
    "vehicle_on_road":  "Is there a car or vehicle on the road?",
    "crowd_present":    "Is there a large crowd of people?",
    "person_on_ground": "Is anyone lying on the ground?",
    "motorcycle":       "Is there a motorcycle or scooter in the scene?",
    "road_blocked":     "Is there a vehicle stopped or blocking the road?",
    "tuk_tuk":          "Is there a tuk-tuk or rickshaw in the scene?",
}

NO_ANSWERS = [
    "no ", "not ", "none", "nobody", "nothing", "empty",
    "cannot", "can not", "without", "absent", "unclear", "unknown"
]

VQA_POSITIVE_KEYWORDS = {
    "people_crossing":  ["crossing", "crosswalk", "people", "pedestrian", "walking"],
    "vehicle_on_road":  ["car", "vehicle", "bus", "truck", "taxi", "road", "driving"],
    "crowd_present":    ["crowd", "people", "group", "many", "several", "pedestrian"],
    "person_on_ground": ["lying", "ground", "fallen", "floor", "sitting on"],
    "motorcycle":       ["motorcycle", "motorbike", "scooter", "moped", "bike"],
    "road_blocked":     ["blocked", "blocking", "stopped", "stationary", "parked"],
    "tuk_tuk":          ["tuk", "rickshaw", "auto", "three-wheel", "small vehicle"],
}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _video_hash(video_path: str) -> str:
    """Generate a hash from video file size + modified time (fast, no full read)."""
    try:
        stat = os.stat(video_path)
        key  = f"{video_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


def load_cache(video_path: str) -> list:
    """Load cached analysis if it exists and matches current video."""
    if not os.path.exists(CACHE_FILE):
        return []
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("video_hash") == _video_hash(video_path):
            analyses = cache.get("analyses", [])
            print(f"[CACHE] Loaded {len(analyses)} cached frame analyses ✅")
            print(f"[CACHE] Skipping BLIP inference — using saved results")
            return analyses
        else:
            print("[CACHE] Video changed — reprocessing with BLIP...")
            return []
    except Exception as e:
        print(f"[CACHE] Cache load failed ({e}) — reprocessing...")
        return []


def save_cache(video_path: str, analyses: list):
    """Save analysis results to cache file."""
    try:
        cache = {
            "video_hash": _video_hash(video_path),
            "video_path": video_path,
            "frame_count": len(analyses),
            "analyses": analyses,
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        print(f"[CACHE] Saved {len(analyses)} frame analyses to {CACHE_FILE} ✅")
    except Exception as e:
        print(f"[CACHE] Could not save cache: {e}")


# ── VQA ───────────────────────────────────────────────────────────────────────

def vqa_frame(image_path: str, question: str) -> str:
    if not BLIP_AVAILABLE:
        return "unknown"
    try:
        image  = Image.open(image_path).convert("RGB")
        inputs = processor(image, question, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        return processor.decode(out[0], skip_special_tokens=True).strip().lower()
    except Exception as e:
        return f"error: {e}"


def _vqa_is_yes(answer: str, check_key: str = None) -> bool:
    ans = answer.lower().strip()
    if not ans or ans.startswith("error") or ans == "unknown":
        return False
    if any(neg in ans for neg in NO_ANSWERS):
        return False
    if check_key and check_key in VQA_POSITIVE_KEYWORDS:
        return any(kw in ans for kw in VQA_POSITIVE_KEYWORDS[check_key])
    return len(ans) > 3


def run_vqa_checks(image_path: str) -> dict:
    return {key: vqa_frame(image_path, q) for key, q in VQA_QUESTIONS.items()}


# ── Caption ───────────────────────────────────────────────────────────────────

def caption_image(image_path: str) -> str:
    if not BLIP_AVAILABLE:
        return f"Security camera frame: {os.path.basename(image_path)}"
    try:
        image = Image.open(image_path).convert("RGB")

        inputs1 = processor(image, return_tensors="pt")
        with torch.no_grad():
            out1 = model.generate(**inputs1, max_new_tokens=60, num_beams=5)
        caption1 = processor.decode(out1[0], skip_special_tokens=True)

        inputs2 = processor(image, text="a security camera shows", return_tensors="pt")
        with torch.no_grad():
            out2 = model.generate(**inputs2, max_new_tokens=60, num_beams=5)
        caption2 = processor.decode(out2[0], skip_special_tokens=True)

        return (caption1 if _keyword_score(caption1) >= _keyword_score(caption2) else caption2).strip()
    except Exception as e:
        return f"Frame analysis error: {str(e)}"


def _keyword_score(caption: str) -> int:
    cap, score = caption.lower(), 0
    for _, kws in OBJECT_MAP:
        if any(kw in cap for kw in kws):
            score += 1
    for kw in ACTION_KEYWORDS:
        if kw in cap:
            score += 1
    return score


# ── Object + action extraction ────────────────────────────────────────────────

def _extract_objects(caption: str, vqa: dict = None) -> list:
    cap, objects = caption.lower(), []
    for label, keywords in OBJECT_MAP:
        if any(kw in cap for kw in keywords) and label not in objects:
            objects.append(label)
    if vqa:
        if _vqa_is_yes(vqa.get("motorcycle", ""), "motorcycle") and "motorcycle" not in objects:
            objects.append("motorcycle")
        if _vqa_is_yes(vqa.get("tuk_tuk", ""), "tuk_tuk") and "rickshaw" not in objects:
            objects.append("rickshaw")
        if _vqa_is_yes(vqa.get("vehicle_on_road", ""), "vehicle_on_road") and "car" not in objects:
            objects.append("car")
        if _vqa_is_yes(vqa.get("people_crossing", ""), "people_crossing") and "person" not in objects:
            objects.append("person")
    return objects


def _extract_actions(caption: str) -> list:
    cap = caption.lower()
    return [kw for kw in ACTION_KEYWORDS if kw in cap]


def _is_low_confidence(caption: str, objects: list) -> bool:
    return not objects and not any(kw in caption.lower() for kw in ACTION_KEYWORDS)


def _compute_risk(objects: list, actions: list, time: str, vqa: dict = None) -> int:
    score = 0
    try:
        hour = int(time.split(":")[0])
        if hour >= 22 or hour < 5:
            score += 4
    except Exception:
        pass
    if "person" in objects:
        score += 1
    score += len({"loitering", "running", "gathering", "crossing", "lying"} & set(actions))
    if len(objects) >= 3:
        score += 1
    if vqa:
        if _vqa_is_yes(vqa.get("person_on_ground", ""), "person_on_ground"): score += 4
        if _vqa_is_yes(vqa.get("road_blocked", ""), "road_blocked"):          score += 2
        if _vqa_is_yes(vqa.get("crowd_present", ""), "crowd_present"):        score += 1
    return min(score, 10)


def _build_vqa_flags(vqa: dict) -> list:
    flags = []
    checks = [
        ("person_on_ground", "PERSON_ON_GROUND"),
        ("road_blocked",     "ROAD_BLOCKED"),
        ("crowd_present",    "CROWD_DETECTED"),
        ("people_crossing",  "PEDESTRIANS_CROSSING"),
        ("motorcycle",       "MOTORCYCLE_DETECTED"),
        ("tuk_tuk",          "TUK_TUK_DETECTED"),
        ("vehicle_on_road",  "VEHICLE_ON_ROAD"),
    ]
    for key, flag in checks:
        if _vqa_is_yes(vqa.get(key, ""), key):
            flags.append(flag)
    return flags


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze_real_frame(frame_info: dict) -> dict:
    filepath = frame_info["filepath"]
    frame_id = frame_info["frame_id"]
    time     = frame_info["time"]
    location = frame_info.get("location", "Street / Main Gate")

    print(f"  [BLIP] Frame {frame_id:02d} ({frame_info['filename']})...")
    caption  = caption_image(filepath)
    vqa      = run_vqa_checks(filepath)
    objects  = _extract_objects(caption, vqa)
    actions  = _extract_actions(caption)
    low_conf = _is_low_confidence(caption, objects)
    risk     = _compute_risk(objects, actions, time, vqa)
    flags    = _build_vqa_flags(vqa)

    tag = " [LOW CONFIDENCE]" if low_conf else ""
    print(f"         Caption : {caption}{tag}")
    print(f"         Objects : {objects} | VQA flags: {flags} | Risk: {risk}/10")

    return {
        "frame_id":         frame_id,
        "time":             time,
        "location":         location,
        "raw_description":  caption,
        "detected_objects": objects,
        "detected_actions": actions,
        "caption":          caption,
        "risk_score":       risk,
        "low_confidence":   low_conf,
        "vqa_results":      vqa,
        "vqa_flags":        flags,
        "source":           "real_video",
        "image_path":       filepath,
    }


def analyze_all_frames(frames_dir: str = "frames",
                       video_path: str = "security_footage.mp4") -> list:
    """
    Analyze all frames. Uses cache if video hasn't changed.
    Pass video_path for cache validation.
    """
    import glob

    # Try loading from cache first
    cached = load_cache(video_path)
    if cached:
        return cached

    # No cache — run BLIP on all frames
    image_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not image_files:
        print(f"[ERROR] No frames in '{frames_dir}/' — run extract_frames.py first")
        return []

    print(f"\n[VLM] Analyzing {len(image_files)} frames with BLIP + VQA...")
    analyses = []
    for i, filepath in enumerate(image_files):
        frame_info = {
            "frame_id": i + 1,
            "time":     f"{i // 60:02d}:{i % 60:02d}",
            "filename": os.path.basename(filepath),
            "filepath": filepath,
            "location": "Street / Main Gate",
        }
        analyses.append(analyze_real_frame(frame_info))

    # Save to cache
    save_cache(video_path, analyses)
    return analyses


def generate_video_summary(analyses: list) -> str:
    if not analyses:
        return "No frames analyzed."
    all_objects = []
    all_flags   = []
    for a in analyses:
        all_objects.extend(a["detected_objects"])
        all_flags.extend(a.get("vqa_flags", []))
    unique  = list(set(all_objects))
    flags   = list(set(all_flags))
    high    = [a for a in analyses if a["risk_score"] >= 6]
    lowconf = [a for a in analyses if a.get("low_confidence")]
    return (
        f"Security scan of {len(analyses)} real video frames; "
        f"detected: {', '.join(unique) if unique else 'no significant objects'}; "
        f"VQA flags: {', '.join(flags) if flags else 'none'}; "
        f"{len(high)} elevated-risk frame(s); "
        f"{len(lowconf)} low-confidence frame(s) flagged for manual review."
    )