// ASH tabs: report the open tabs to ASH on this computer. Nothing else.
//
// Sent: for each tab its id, url, title, whether it is the active tab of its window, whether it is playing
// sound, and whether the window is private. ASH decides what to keep (hosts only; private, banking, mail,
// health and similar pages are dropped on its side). Never read: page content, forms, cookies, history.
// config.json (port + token) is written next to this file by ASH when it starts.

let cfg = null;
let timer = null;

async function loadConfig() {
  if (cfg) return cfg;
  try {
    const r = await fetch(chrome.runtime.getURL("config.json"));
    cfg = await r.json();
  } catch (e) {
    cfg = null;
  }
  return cfg;
}

async function report() {
  const c = await loadConfig();
  if (!c || !c.port || !c.token) return;
  let tabs = [];
  let focused = false;
  try {
    tabs = await chrome.tabs.query({});
    const w = await chrome.windows.getLastFocused();
    focused = !!(w && w.focused);
    const focusedId = w ? w.id : -1;
    tabs = tabs.map(t => ({
      id: t.id,
      url: t.url || "",
      title: t.title || "",
      // "active" only counts in the window the user is actually in
      active: !!t.active && t.windowId === focusedId,
      audible: !!t.audible,
      incognito: !!t.incognito,
    }));
  } catch (e) {
    return;
  }
  try {
    await fetch(`http://127.0.0.1:${c.port}/tabs`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-ASH-Token": c.token },
      body: JSON.stringify({ focused, tabs }),
    });
  } catch (e) {
    cfg = null;           // ASH not running, or restarted with a new config: look again next time
  }
}

function soon() {
  if (timer) return;
  timer = setTimeout(() => { timer = null; report(); }, 800);   // one report for a burst of changes
}

chrome.tabs.onActivated.addListener(soon);
chrome.tabs.onUpdated.addListener((id, info) => { if (info.url || info.title || info.audible !== undefined || info.status === "complete") soon(); });
chrome.tabs.onRemoved.addListener(soon);
chrome.tabs.onCreated.addListener(soon);
chrome.windows.onFocusChanged.addListener(soon);
chrome.alarms.create("ash-heartbeat", { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener(a => { if (a.name === "ash-heartbeat") report(); });
report();
