"""
A few extra native tools, registered the new easy way.

Import this module once at startup (ash.py does it) and every @tool
below is automatically live -- no edits to _deterministic_execute
needed.
"""

import re
import sys
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("ash.tools.extra")


def system_stats_tool(query: str) -> dict:
    try:
        import psutil
    except ImportError:
        return {"ok": False, "error": "psutil not installed (pip install psutil)"}

    cpu = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory()
    boot = datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.now() - boot

    temp_c = None
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            first = next(iter(temps.values()))
            if first:
                temp_c = first[0].current
    except Exception:
        pass

    return {
        "ok": True,
        "cpu_percent": cpu,
        "ram_used_percent": mem.percent,
        "ram_used_mb": round(mem.used / (1024 ** 2)),
        "ram_total_mb": round(mem.total / (1024 ** 2)),
        "uptime": str(uptime).split(".")[0],
        "cpu_temp_c": temp_c,
    }


def unit_converter_tool(query: str) -> dict:
    q = query.lower().strip()

    # Normalize common full-word unit names to the abbreviations the
    # regexes below match -- "convert 10 km to miles" was failing
    # before because the parser only recognized "mi", not "miles".
    _ALIASES = {
        "kilometers": "km", "kilometer": "km", "kms": "km",
        "meters": "m", "meter": "m",
        "centimeters": "cm", "centimeter": "cm",
        "millimeters": "mm", "millimeter": "mm",
        "miles": "mi", "mile": "mi",
        "yards": "yd", "yard": "yd",
        "feet": "ft", "foot": "ft",
        "inches": "in", "inch": "in",
        "kilograms": "kg", "kilogram": "kg",
        "grams": "g", "gram": "g",
        "pounds": "lb", "pound": "lb",
        "ounces": "oz", "ounce": "oz",
        "celsius": "c", "fahrenheit": "f",
        "liters": "l", "liter": "l", "litres": "l", "litre": "l",
        "milliliters": "ml", "milliliter": "ml", "millilitres": "ml",
        "gallons": "gal", "gallon": "gal", "cups": "cup", "pints": "pt", "pint": "pt",
        "quarts": "qt", "quart": "qt",
    }
    for word, abbr in sorted(_ALIASES.items(), key=lambda kv: -len(kv[0])):
        q = re.sub(rf"\b{word}\b", abbr, q)

    # "how many feet in 2 meters" / "how many pounds is 70 kg" -> "2 m to ft"
    # (the unit asked about comes first, the quantity second).
    m = re.search(r"how\s+many\s+([a-z]+)\s+(?:are\s+|is\s+)?(?:in|is|are)\s+(?:a\s+|an\s+)?"
                  r"([-+]?\d*\.?\d+)?\s*([a-z]+)", q)
    if m:
        want, num, have = m.group(1), m.group(2) or "1", m.group(3)
        q = f"{num} {have} to {want}"

    m = re.search(r"([-+]?\d*\.?\d+)\s*(c|celsius|f|fahrenheit)\s*(?:to|in|->)\s*(c|celsius|f|fahrenheit)", q)
    if m:
        val, frm, to = float(m.group(1)), m.group(2)[0], m.group(3)[0]
        if frm == "c" and to == "f":
            return {"ok": True, "result": round(val * 9 / 5 + 32, 2), "unit": "F"}
        if frm == "f" and to == "c":
            return {"ok": True, "result": round((val - 32) * 5 / 9, 2), "unit": "C"}
        return {"ok": True, "result": val, "unit": to.upper()}

    _LENGTH_TO_M = {"km": 1000, "m": 1, "cm": 0.01, "mm": 0.001, "mi": 1609.34, "yd": 0.9144, "ft": 0.3048, "in": 0.0254}
    m = re.search(r"([-+]?\d*\.?\d+)\s*(km|cm|mm|mi|yd|ft|in|m)\s*(?:to|in|->)\s*(km|cm|mm|mi|yd|ft|in|m)\b", q)
    if m:
        val, frm, to = float(m.group(1)), m.group(2), m.group(3)
        meters = val * _LENGTH_TO_M[frm]
        return {"ok": True, "result": round(meters / _LENGTH_TO_M[to], 4), "unit": to}

    _VOLUME_TO_L = {"l": 1, "ml": 0.001, "gal": 3.78541, "qt": 0.946353, "pt": 0.473176, "cup": 0.236588}
    m = re.search(r"([-+]?\d*\.?\d+)\s*(ml|gal|qt|pt|cup|l)\s*(?:to|in|->)\s*(ml|gal|qt|pt|cup|l)\b", q)
    if m:
        val, frm, to = float(m.group(1)), m.group(2), m.group(3)
        litres = val * _VOLUME_TO_L[frm]
        return {"ok": True, "result": round(litres / _VOLUME_TO_L[to], 4), "unit": to}

    _WEIGHT_TO_KG = {"kg": 1, "g": 0.001, "lb": 0.453592, "lbs": 0.453592, "oz": 0.0283495}
    m = re.search(r"([-+]?\d*\.?\d+)\s*(kg|g|lb|lbs|oz)\s*(?:to|in|->)\s*(kg|g|lb|lbs|oz)\b", q)
    if m:
        val, frm, to = float(m.group(1)), m.group(2), m.group(3)
        kg = val * _WEIGHT_TO_KG[frm]
        return {"ok": True, "result": round(kg / _WEIGHT_TO_KG[to], 4), "unit": to}

    return {"ok": False, "error": "Couldn't parse a 'X unit to unit' conversion from that query."}


_SMALL_NUMS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
               "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
               "thirty": 30, "forty": 40, "sixty": 60}
_UNIT_DELTA = {"second": "seconds", "sec": "seconds", "minute": "minutes", "min": "minutes",
               "hour": "hours", "hr": "hours", "day": "days", "week": "weeks"}
_IN_TIME = re.compile(
    r"\bin\s+(half\s+an?|\d+(?:\.\d+)?|an?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"fifteen|twenty|thirty|forty|sixty)\s*(second|sec|minute|min|hour|hr|day|week)s?\b")


def reminder_tool(query: str) -> dict:
    q = query.lower().strip()
    now = datetime.now()
    due = None
    when = None

    m = _IN_TIME.search(q)
    if m:
        amount, unit = m.group(1), m.group(2)
        if amount.startswith("half"):
            n = 0.5
        elif amount[0].isdigit():
            n = float(amount)
        else:
            n = _SMALL_NUMS[amount]
        due = now + timedelta(**{_UNIT_DELTA[unit]: n})
        when = m.group(0)
    else:
        m = re.search(r"\btomorrow\b", q)
        if m:
            due = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
            when = m.group(0)

    # What to be reminded of: drop the command and the time phrase.
    text = q
    if when:
        text = text.replace(when, " ", 1)
    text = re.sub(r"^\s*(?:please\s+)?(?:set\s+(?:a\s+|an\s+)?reminder|remind\s+me)\b\s*"
                  r"(?:to\b|that\b|about\b)?\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.")

    if due is None:
        return {"ok": False, "error": "Couldn't parse a time from that reminder (try 'in 20 minutes').", "text": text}
    if not text:
        return {"ok": False, "error": "What should I remind you about?"}

    return {"ok": True, "text": text, "due_at": due.isoformat(), "created_at": now.isoformat()}


_QUOTED = re.compile(r"\"([^\"]+)\"|(?:(?<=\s)|^)'([^']+)'(?=[\s.,?!)]|$)")
_COUNT_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:(?:can|could)\s+you\s+)?(?:count|tell\s+me|what(?:'s|\s+is)|how\s+many|"
    r"reading\s+time|word\s+count|char(?:acter)?\s+count)\b.*?\b(?:words?|characters?|letters?|time|count)\b"
    r"(?:\s+(?:in|of|for|is|are))?\s*(?:this|the|that|my)?\s*(?:text|sentence|phrase|passage|paragraph|"
    r"string|message)?\s*[:\-]?\s*", re.I)


def _text_to_count(query: str) -> str:
    """The text the person wants counted, without the request wrapped around it."""
    q = (query or "").strip()
    m = _QUOTED.search(q)
    if m:
        return (m.group(1) or m.group(2)).strip()
    if ":" in q:
        return q.split(":", 1)[1].strip()
    return _COUNT_LEAD.sub("", q, count=1).strip()


def word_count_tool(query: str) -> dict:
    text = _text_to_count(query)
    if not text:
        return {"ok": False, "error": "Tell me which text to count, e.g. word count of: your text here"}
    words = text.split()
    reading_seconds = round(len(words) / 200 * 60)  # ~200 wpm
    return {
        "ok": True,
        "words": len(words),
        "characters": len(text),
        "estimated_reading_time_sec": reading_seconds,
    }
