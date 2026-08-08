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
    }
    for word, abbr in sorted(_ALIASES.items(), key=lambda kv: -len(kv[0])):
        q = re.sub(rf"\b{word}\b", abbr, q)

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

    _WEIGHT_TO_KG = {"kg": 1, "g": 0.001, "lb": 0.453592, "lbs": 0.453592, "oz": 0.0283495}
    m = re.search(r"([-+]?\d*\.?\d+)\s*(kg|g|lb|lbs|oz)\s*(?:to|in|->)\s*(kg|g|lb|lbs|oz)\b", q)
    if m:
        val, frm, to = float(m.group(1)), m.group(2), m.group(3)
        kg = val * _WEIGHT_TO_KG[frm]
        return {"ok": True, "result": round(kg / _WEIGHT_TO_KG[to], 4), "unit": to}

    return {"ok": False, "error": "Couldn't parse a 'X unit to unit' conversion from that query."}


def reminder_tool(query: str) -> dict:
    q = query.lower()
    now = datetime.now()
    due = None

    m = re.search(r"in (\d+)\s*(minute|min|hour|hr|day)s?", q)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {
            "minute": timedelta(minutes=n), "min": timedelta(minutes=n),
            "hour": timedelta(hours=n), "hr": timedelta(hours=n),
            "day": timedelta(days=n),
        }[unit]
        due = now + delta

    text = re.sub(r"remind me (to|that)?\s*", "", q, flags=re.IGNORECASE).strip()

    if due is None:
        return {"ok": False, "error": "Couldn't parse a time from that reminder (try 'in 20 minutes').", "text": text}

    return {"ok": True, "text": text, "due_at": due.isoformat(), "created_at": now.isoformat()}


def word_count_tool(query: str) -> dict:
    words = query.split()
    chars = len(query)
    reading_seconds = round(len(words) / 200 * 60)  # ~200 wpm
    return {
        "ok": True,
        "words": len(words),
        "characters": chars,
        "estimated_reading_time_sec": reading_seconds,
    }