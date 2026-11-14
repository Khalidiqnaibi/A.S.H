// safehtml.js -- turn a reply into HTML without letting the reply write the page.
//
// Replies can carry text from files, OCR, window titles and the web. marked.parse() passes raw HTML through, and the old front end
// put its result in innerHTML: a reply containing <img src=x onerror=...> ran script in the page. Here every "<" in the text is
// escaped BEFORE marked sees it (so only markdown syntax can make tags), links may only be http, https or mailto, and anything
// that could run or load is removed afterwards.

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function safeHref(href) {
  try {
    const u = new URL(String(href).trim(), 'http://placeholder.invalid/');
    return u.protocol === 'http:' || u.protocol === 'https:' || u.protocol === 'mailto:';
  } catch (e) {
    return false;
  }
}

function safeMarkdown(text, markedLib, doc) {
  const html = markedLib.parse(escapeHtml(text));
  const t = doc.createElement('template');
  t.innerHTML = html;
  t.content.querySelectorAll('script,iframe,object,embed,style,link,meta,base,form').forEach(n => n.remove());
  t.content.querySelectorAll('*').forEach(n => {
    for (const a of Array.from(n.attributes)) {
      if (/^on/i.test(a.name)) n.removeAttribute(a.name);          // no event handlers, whatever got through
    }
  });
  t.content.querySelectorAll('a[href]').forEach(a => {
    if (!safeHref(a.getAttribute('href'))) a.removeAttribute('href');
    else { a.setAttribute('rel', 'noopener noreferrer'); a.setAttribute('target', '_blank'); }
  });
  t.content.querySelectorAll('img').forEach(img => {
    const src = img.getAttribute('src') || '';
    if (!safeHref(src)) img.remove();
  });
  return t.innerHTML;
}

if (typeof module !== 'undefined') module.exports = { escapeHtml, safeHref, safeMarkdown };
