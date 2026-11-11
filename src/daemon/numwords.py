"""
src/daemon/numwords.py

"Forty five minutes", "until three": spoken numbers as digits.

Speech-to-text writes numbers as words about as often as digits, and the duration
parsers (quiet, focus) only read digits, so "quiet for forty five minutes" and
"do not disturb until three" were not understood and went to the model, which
cannot set either. `digits()` rewrites runs of number words ("forty five",
"twenty-five", "three") as digits and touches nothing else; "an hour" and "half an
hour" are left for the parsers that already know them.
"""

from __future__ import annotations

import re

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_W = "|".join(sorted(list(_UNITS) + list(_TENS), key=len, reverse=True))
_RUN = re.compile(r"\b(?:(?:" + "|".join(_TENS) + r")(?:[\s-]+(?:" + "|".join(k for k in _UNITS if 0 < _UNITS[k] < 10) +
                  r"))?|" + "|".join(_UNITS) + r")\b", re.I)


def _value(run: str) -> int:
    total = 0
    for w in re.split(r"[\s-]+", run.lower()):
        total += _TENS.get(w, _UNITS.get(w, 0))
    return total


def digits(text: str) -> str:
    """The text with number words as digits: "quiet for forty five minutes" -> "quiet for 45 minutes"."""
    return _RUN.sub(lambda m: str(_value(m.group(0))), text or "")
