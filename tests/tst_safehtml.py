"""
tst_safehtml.py -- a reply cannot write the page.

The older web front end turned a reply into HTML with marked.parse() and put it in innerHTML. Replies can carry text from files,
OCR, window titles and web pages, and marked passes raw HTML through: a reply containing <img src=x onerror=...> ran script in the
page. static/safehtml.js escapes every "<" before marked sees it, allows only http, https and mailto links, and strips what could
run or load. The pure parts are run under node; the DOM part is checked by reading the source (no browser here).

Run:  PYTHONIOENCODING=utf-8 python tests/tst_safehtml.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "static", "safehtml.js")

SCRIPT = r"""
const m = require(process.argv[1]);
const out = {
  img: m.escapeHtml('<img src=x onerror=alert(1)>'),
  script: m.escapeHtml('<script>alert(document.cookie)</script>'),
  amp: m.escapeHtml('Tom & Jerry "quote" it\'s'),
  num: m.escapeHtml(42),
  h: ['https://example.com/a?b=1', 'http://example.com', 'mailto:a@b.co', '/relative/path', 'javascript:alert(1)',
      ' JavaScript:alert(1)', 'data:text/html;base64,PHNjcmlwdD4=', 'vbscript:x', '//evil.example/x', '', 'java\tscript:alert(1)'].map(m.safeHref),
};
console.log(JSON.stringify(out));
"""


def t1_pure():
    print("\n=== Escaping and links (node) ===")
    node = shutil.which("node")
    if not node:
        print("  (node is not installed: the pure-function checks are skipped)")
        return
    r = subprocess.run([node, "-e", SCRIPT, JS], capture_output=True, text=True, timeout=30)
    check("the script runs", r.returncode == 0, r.stderr[:200])
    got = json.loads(r.stdout or "{}")
    check("an <img onerror> reaches marked with no angle bracket in it", "<" not in got["img"] and "&lt;img" in got["img"], got["img"])
    check("a <script> too", "<" not in got["script"], got["script"])
    check("& and quotes are escaped once", got["amp"] == "Tom &amp; Jerry &quot;quote&quot; it&#39;s", got["amp"])
    check("a number is text", got["num"] == "42")
    want = [True, True, True, True, False, False, False, False, True, True, False]
    names = ["https", "http", "mailto", "relative", "javascript:", " JavaScript:", "data:", "vbscript:", "// (same scheme)", "empty", "java<tab>script:"]
    for n, g, w in zip(names, got["h"], want):
        check(f"link {n}: {'kept' if w else 'dropped'}", g is w, str(g))


def t2_wiring():
    print("\n=== Wiring ===")
    r = open(os.path.join(ROOT, "static", "renderer.js"), encoding="utf-8").read()
    check("replies go through safeMarkdown, not straight into innerHTML", "safeMarkdown(txt, marked, document)" in r
          and "innerHTML = marked.parse" not in r)
    t = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    check("the page loads safehtml.js before renderer.js", t.index("safehtml.js") < t.index("renderer.js"))
    j = open(JS, encoding="utf-8").read()
    check("the DOM pass removes scripts, frames, event handlers and unsafe links and images",
          all(w in j for w in ("script,iframe,object,embed", "^on", "safeHref(a.getAttribute('href'))", "img.remove()")))
    check("escaping comes before marked", j.index("escapeHtml(text)") < j.index("t.innerHTML = html") and "markedLib.parse(escapeHtml(text))" in j)


if __name__ == "__main__":
    t1_pure()
    t2_wiring()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
