"""
vlm/analyzer.py
Mock VLM (Vision Language Model) analyzer.
In production: swap analyze_frame() with BLIP2 / LLaVA inference.
For this prototype: uses regex + keyword extraction on text frame descriptions
to simulate what a real VLM would produce.
"""

import re
from typing import Optional

# Object categories the VLM "detects"
VEHICLE_KEYWORDS = [
    "ford f150", "truck", "van", "sedan", "suv", "car", "vehicle",
    "ups", "delivery", "white sedan", "blue ford", "brown van"
]

PERSON_KEYWORDS = [
    "person", "individual", "man", "woman", "people", "driver",
    "someone", "figure", "pedestrian", "suspect"
]

ANIMAL_KEYWORDS = ["dog", "cat", "animal", "bird"]

ACTION_KEYWORDS = [
    "loitering", "standing", "running", "attempting to open",
    "walking", "exiting", "entering", "parked", "waiting",
    "suspicious", "looking around", "not moving"
]


def analyze_frame(frame_id: int, time: str, location: str,
                  description: str) -> dict:
    """
    Simulates VLM output for a text-based frame.
    Returns structured analysis with detected objects, actions, and risk score.

    In production: replace description with actual image bytes and call:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(...)
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
    """
    desc_lower = description.lower()

    detected_objects = []
    detected_actions = []

    # Detect vehicles
    for kw in VEHICLE_KEYWORDS:
        if kw in desc_lower:
            label = _normalize_vehicle(kw)
            if label not in detected_objects:
                detected_objects.append(label)

    # Detect persons
    for kw in PERSON_KEYWORDS:
        if kw in desc_lower:
            if "person" not in detected_objects:
                detected_objects.append("person")
            break

    # Detect animals
    for kw in ANIMAL_KEYWORDS:
        if kw in desc_lower:
            detected_objects.append(kw)

    # Detect actions
    for kw in ACTION_KEYWORDS:
        if kw in desc_lower:
            detected_actions.append(kw)

    # Risk score: 0-10
    risk_score = _compute_risk(detected_objects, detected_actions, time)

    # Generate a short VLM-style caption
    caption = _generate_caption(detected_objects, detected_actions, location)

    return {
        "frame_id":         frame_id,
        "time":             time,
        "location":         location,
        "raw_description":  description,
        "detected_objects": detected_objects,
        "detected_actions": detected_actions,
        "caption":          caption,
        "risk_score":       risk_score,
    }


def _normalize_vehicle(kw: str) -> str:
    if "f150" in kw or "ford" in kw:
        return "Blue Ford F150"
    if "ups" in kw or "delivery" in kw or "van" in kw or "brown" in kw:
        return "delivery van"
    if "sedan" in kw:
        return "sedan"
    if "truck" in kw:
        return "truck"
    return "vehicle"


def _compute_risk(objects: list, actions: list, time: str) -> int:
    score = 0
    # Midnight window
    hour = _parse_hour(time)
    if hour >= 23 or hour < 5:
        score += 4

    # Person presence
    if "person" in objects:
        score += 2

    # Suspicious actions
    risky_actions = {"loitering", "attempting to open", "suspicious",
                     "looking around", "not moving"}
    score += len(risky_actions & set(actions)) * 2

    # Cap at 10
    return min(score, 10)


def _parse_hour(time_str: str) -> int:
    try:
        return int(time_str.split(":")[0])
    except Exception:
        return 0


def _generate_caption(objects: list, actions: list, location: str) -> str:
    if not objects:
        return f"No significant objects detected at {location}."
    obj_str = ", ".join(objects)
    if actions:
        act_str = actions[0]
        return f"{obj_str.capitalize()} detected {act_str} at {location}."
    return f"{obj_str.capitalize()} detected at {location}."


def generate_video_summary(analyses: list) -> str:
    """
    Bonus: Generate a 1-sentence summary of all frames processed.
    """
    locations = list(set(a["location"] for a in analyses))
    all_objects = []
    for a in analyses:
        all_objects.extend(a["detected_objects"])
    unique_objects = list(set(all_objects))
    high_risk = [a for a in analyses if a["risk_score"] >= 6]

    summary = (
        f"Security scan covered {len(analyses)} frames across "
        f"{', '.join(locations)}; detected {', '.join(unique_objects) if unique_objects else 'no significant objects'}; "
        f"{len(high_risk)} high-risk event(s) flagged for review."
    )
    return summary
