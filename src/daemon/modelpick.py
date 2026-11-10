"""
src/daemon/modelpick.py

Which model answers: the small one for small talk, the big one for the rest.

A greeting, a thanks or "I'm tired" does not need the big model, and on a
laptop the difference is seconds. Only when a second model is set up (daemon.json
`llm.light_model`, e.g. "qwen2.5:1.5b", or ASH_OLLAMA_LIGHT_MODEL); otherwise
this never matters. The rule is deliberately narrow -- when unsure, the big
model:

    light    a short line that is plainly chat: hello, thanks, how are you,
             I'm tired / bored / happy, good one, tell me a joke, nice
    default  anything with a question word or a task verb (who, what, why,
             how to, explain, write, translate, summarize, calculate, list ...),
             anything with digits, code or a link, anything long, a picture, and
             anything said while a persona or the clipboard is in play
"""

from __future__ import annotations

import re

_HEAVY = re.compile(r"\b(?:who|what|when|where|why|which|how\s+(?:do|does|can|to|much|many|long|far)|explain|write|code|translate|"
                    r"summari[sz]e|calculate|compare|list|define|describe|plan|help\s+me|tell\s+me\s+about|"
                    r"difference|meaning|mean|steps|recipe|fix|debug|error)\b", re.I)
_CHAT = re.compile(r"^\s*(?:(?:hey|hi|hello|yo|oh|ok(?:ay)?|well|so|wow|haha|lol|thanks?|thank\s+you|cheers|nice|cool|great|good\s+one|"
                   r"awesome|sweet|bye|goodbye|see\s+you|how(?:'s|\s+is|\s+are)\s+(?:it\s+going|you|things)|you(?:'re|\s+are)\s+(?:funny|great|cool|the\s+best|nice)|"
                   r"i(?:'m|\s+am)\s+(?:so\s+)?(?:tired|bored|happy|sad|stressed|excited|hungry|sleepy|back|here|fine|good|ok(?:ay)?)|"
                   r"tell\s+me\s+a\s+joke|make\s+me\s+laugh|i\s+(?:love|like)\s+you|you\s+there|are\s+you\s+there|what'?s\s+up|sup)\b"
                   r"[\w\s,'!.?]*)$", re.I)


def pick(text: str, has_image: bool = False, busy_context: bool = False) -> str:
    """"light" or "default"."""
    t = (text or "").strip()
    if not t or has_image or busy_context or len(t) > 60 or len(t.split()) > 8:
        return "default"
    if re.search(r"\d|https?://|[`{}()<>=/\\]", t):
        return "default"
    if _HEAVY.search(t) and not re.match(r"^\s*(?:hey|hi|hello)?[,\s]*how(?:'s|\s+is|\s+are)\s+(?:it\s+going|you|things)", t, re.I):
        return "default"
    return "light" if _CHAT.match(t) else "default"
