"""tst_learnweb.py -- learning something ASH has never seen: the problem is broken into concepts; each is checked against the user's own history first, else googled (or queued with no
connection and done when the network is back); each is saved with a skill that goes to the same search; then 'is that right?' -- a no forgets what was just saved. Fakes only: nothing is
searched, saved to the user's skills or written outside a temp folder. Also: the life-presence step really runs on the tick (a same-named method had shadowed it). Offline.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_learnweb.py"""
import inspect
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
from src.daemon import learnweb as lw  # noqa: E402

check = tw.check


def main():
    print("\n=== L1. Reading the ask, breaking the problem ===")
    check("'learn how to ...' is read", lw.parse("Learn how to make a thumbnail in photoshop from google") == "make a thumbnail in photoshop", str(lw.parse("Learn how to make a thumbnail in photoshop from google")))
    check("other ways of asking", lw.parse("go and learn about rigging in blender") == "rigging in blender" and lw.parse("teach yourself how to solder") == "solder")
    check("'From Google, learn how to ...' (28 Nov, said three times, went to the model)", lw.parse("From Google, learn how to open Discord and type in it.") == "open Discord and type in it"
          and lw.parse("From Google, learn how to open a YouTube video.") == "open a YouTube video", str(lw.parse("From Google, learn how to open a YouTube video.")))
    check("'learn from Google how to ...'", lw.parse("Ash, Learn from Google how to chunk tasks to smaller tasks.") == "chunk tasks to smaller tasks")
    check("a plain sentence is not an ask", lw.parse("I learned something today") is None and lw.parse("what time is it") is None)
    check("concepts: the task and the tool", lw.concepts("make a thumbnail in photoshop") == ["make a thumbnail", "photoshop"], str(lw.concepts("make a thumbnail in photoshop")))
    check("a list is split, capped at five", lw.concepts("python, flask, docker, redis, celery and nginx and git") == ["python", "flask", "docker", "redis", "celery"])

    print("\n=== L2. The method ===")
    d = tempfile.mkdtemp()
    st = lw.Store(os.path.join(d, "learned.json"), clock=lambda: 1000.0)
    taught = []
    hist = lambda c: ["you wrote: photoshop layers, then export as png"] if c == "photoshop" else []
    web = lambda c: "Open the canvas, add text, and export." if "thumbnail" in c else ""
    out = lw.learn("make a thumbnail in photoshop", st, web, hist, teach=taught.append)
    check("your own history comes first, the web fills the rest", out["from_history"] == ["photoshop"] and out["from_web"] == ["make a thumbnail"] and not out["queued"], str(out))
    check("each concept is saved with where it came from", st.items["photoshop"]["source"] == "your own history" and st.items["make a thumbnail"]["source"] == "google")
    check("and a skill that goes to the same search", sorted(t["phrase"] for t in taught) == ["make a thumbnail", "photoshop"] and taught[0]["steps"][0]["do"] == "search")
    check("what it says ends by asking", lw.say(out).endswith("Is that right?") and "from the web: make a thumbnail" in lw.say(out), lw.say(out))
    check("it survives a restart", lw.Store(os.path.join(d, "learned.json")).items.keys() == st.items.keys())
    print("\n=== L3. Offline: queued, done later ===")
    def down(c):
        raise OSError("no network")
    st2 = lw.Store(os.path.join(d, "l2.json"), clock=lambda: 1.0)
    out = lw.learn("rigging in blender", st2, down, lambda c: [], teach=taught.append)
    check("no connection: queued and said so", out["queued"] == ["rigging", "blender"] and "queued" in lw.say(out) and st2.queue == ["rigging", "blender"], lw.say(out))
    check("no fetch at all (privacy on) queues too", lw.learn("solder", lw.Store(None), None, lambda c: [])["queued"] == ["solder"])
    done = lw.drain(st2, lambda c: "A short explanation of " + c, teach=taught.append)
    check("when the network is back the queue is done", done == ["rigging", "blender"] and not st2.queue and "rigging" in st2.items, str(done))
    check("the status lists what it learned", "rigging" in st2.status() and lw.Store(None).status().startswith("Nothing learned"))
    check("forget removes it", lw.forget(st2, ["rigging"]) == 1 and "rigging" not in st2.items)
    check("snippets are read from a results page", lw.snippet_from_html('<a class="result__snippet" href="x">Use <b>layers</b> &amp; text</a>') == "Use layers & text")
    check("the method is in steps", len(lw.METHOD) == 5 and lw.asks_method("how do you learn from google") and lw.asks_status("what are you learning"))

    print("\n=== L3b. Online but the search came back empty is not 'offline' (28 Nov) ===")
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        if "duckduckgo" in url:
            return 202, "<html>anomaly</html>"
        if "bing.com" in url:
            return 200, '<li class="b_algo"><h2>t</h2><p class="b_lineclamp2">May 13, 2015 &#0183; Break a big task into steps you can finish in one sitting.</p></li>'
        return 200, "{}"
    real_get, real_online = lw._get, lw.is_online
    lw._get = fake_get
    try:
        got = lw.web_fetch("chunk tasks")
        check("DuckDuckGo's bot check falls through to Bing, and the date is dropped", got == "Break a big task into steps you can finish in one sitting." and len(calls) == 2, f"{got!r} {calls}")

        def only_wiki(url, timeout):
            if "wikipedia" in url:
                return 200, '{"query":{"search":[{"title":"Chunking","snippet":"a <span class=\\"searchmatch\\">task</span> method"}]}}'
            return 200, ""
        lw._get = only_wiki
        check("and then to Wikipedia", lw.web_fetch("chunking") == "Chunking: a task method", lw.web_fetch("chunking"))
        lw._get = lambda url, timeout: (200, "")
        out = lw.learn("chunk tasks", lw.Store(None), lw.web_fetch, lambda c: [])
        check("the web answered and had nothing: said so, NOT queued as offline", out["empty"] == ["chunk tasks"] and not out["queued"] and "reached the web" in lw.say(out), str(out))

        def dead(url, timeout):
            raise lw.NoConnection("no route")
        lw._get = dead
        out = lw.learn("chunk tasks", lw.Store(None), lw.web_fetch, lambda c: [])
        check("no network at all: queued, and said so", out["queued"] == ["chunk tasks"] and not out["empty"] and "no connection" in lw.say(out).lower(), str(out))
    finally:
        lw._get, lw.is_online = real_get, real_online
    check("Bing snippets are read", lw.snippet_from_bing('<li class="b_algo"><p class="b_lineclamp3">Plain words about a thing that is long enough to count.</p></li>').startswith("Plain words"))

    print("\n=== L4. In the runtime ===")
    import tst_modellast as tm

    def rig(fetch):
        rt = tm.rig()
        rt.cfg.life_dir = tempfile.mkdtemp()
        rt._learn_fetch = lambda: fetch
        rt._learn_history = lambda c: []
        rt.cfg.skills_path = os.path.join(tempfile.mkdtemp(), "skills.json")
        return rt
    rt = rig(lambda c: "Something about " + c)
    used, said = rt.say("learn how to make a thumbnail in photoshop")
    check("learned by rule, no model, and asked if right", not used and said and said[-1].endswith("Is that right?") and sorted(rt._skills_obj().skills) == ["make a thumbnail", "photoshop"], f"{used} {said} {sorted(rt._skills_obj().skills)}")
    used, said = rt.say("no")
    check("a no forgets them", said and said[-1].startswith("Okay -- forgot 2"), f"{said}")
    rt = rig(down)
    used, said = rt.say("go and learn about rigging in blender")
    check("offline it queues and says so", not used and said and "queued" in said[-1], f"{used} {said}")
    used, said = rt.say("what are you learning")
    check("the queue is listed", said and "rigging" in said[-1], f"{said}")
    from src.daemon import ambient
    src = inspect.getsource(ambient.AmbientRuntime._social_step)
    check("the life-presence step is on the tick (it was shadowed by the camera step of the same name)", "_life_presence_step" in src and "_learn_step" in src)
    check("the two steps are different methods", ambient.AmbientRuntime._life_presence_step is not ambient.AmbientRuntime._presence_step)
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
