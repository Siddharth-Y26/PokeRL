/* PokeRL Coach — console version.
 *
 * Paste this whole file into the browser DevTools console while a Showdown battle page is
 * open. No Tampermonkey, no extension, no page reload.
 *
 * The userscript version wraps window.WebSocket, which only works if it runs before the page
 * creates its socket (@run-at document-start). Pasted into the console the socket already
 * exists, so this hooks app.receive instead — the client's own handler for every raw protocol
 * frame (window.app is exposed by the client at load; see js/oldclient/client.js).
 *
 * Wrapping rather than replacing: the original is always called, so the battle UI keeps
 * working exactly as before even if anything here throws.
 *
 * Advice only. It never sends a move to Showdown; you play every turn yourself.
 *
 * Re-pasting is safe — it detaches the previous hook first.
 */

(function () {
  'use strict';

  var COACH_URL = 'ws://localhost:8765';

  // window.app is created when the client boots. Pasted into the console it already exists;
  // loaded as a <script> by the self-hosted page it does not yet, so wait for it rather than
  // giving up. Same file serves both uses.
  if (typeof window.app === 'undefined' || !window.app.receive) {
    var waited = 0;
    var poll = setInterval(function () {
      waited += 100;
      if (window.app && window.app.receive) { clearInterval(poll); attach(); }
      else if (waited > 30000) {
        clearInterval(poll);
        console.error('[pokerl-coach] window.app never appeared — is this a Showdown page?');
      }
    }, 100);
    return;
  }
  attach();

  function attach() {
  // Detach a previous attach so hooks do not stack.
  if (window.__pokerlCoach && window.__pokerlCoach.detach) {
    window.__pokerlCoach.detach();
  }

  var socket = null, ready = false, queue = [];
  var buffers = {};                       // battle_tag -> lines awaiting a |request|
  var originalReceive = window.app.receive;

  // ---------------------------------------------------------------- panel

  var existing = document.getElementById('pokerl-coach');
  if (existing) existing.remove();

  var panel = document.createElement('div');
  panel.id = 'pokerl-coach';
  panel.innerHTML =
    '<div class="prc-head" id="prc-head" title="click to collapse">' +
    '<span class="prc-title">PokeRL Coach</span>' +
    '<span><span class="prc-status" id="prc-status">starting</span>' +
    '<span class="prc-toggle" id="prc-toggle">&minus;</span></span></div>' +
    '<div class="prc-body" id="prc-body"><p class="prc-empty">Waiting for your turn…</p></div>' +
    '<div class="prc-foot" id="prc-foot">Advice only — you play every move.</div>';

  var style = document.createElement('style');
  style.textContent =
    '#pokerl-coach{position:fixed;top:58px;right:10px;width:320px;max-height:calc(100vh - 70px);'+
    'overflow-y:auto;' +
    'z-index:2147483000;font:12px/1.45 -apple-system,"Segoe UI",Roboto,sans-serif;background:#1e232b;' +
    'color:#e7ebf0;border:1px solid #39414d;border-radius:8px;box-shadow:0 6px 22px rgba(0,0,0,.45)}' +
    '#pokerl-coach .prc-head{display:flex;justify-content:space-between;align-items:center;padding:8px 10px;' +
    'border-bottom:1px solid #39414d;background:#252b34;border-radius:8px 8px 0 0}' +
    '#pokerl-coach .prc-title{font-weight:600}' +
    '#pokerl-coach .prc-status{font-size:10px;color:#93a0b4;margin-right:8px}' +
    '#pokerl-coach .prc-head{cursor:pointer;user-select:none}' +
    '#pokerl-coach .prc-toggle{font-size:14px;color:#93a0b4;padding:0 2px}' +
    '#pokerl-coach.prc-collapsed .prc-body,#pokerl-coach.prc-collapsed .prc-foot{display:none}' +
    '#pokerl-coach .prc-status.ok{color:#6ddb8f}#pokerl-coach .prc-status.warn{color:#f0b429}' +
    '#pokerl-coach .prc-body{padding:10px}#pokerl-coach .prc-empty{color:#93a0b4;margin:0}' +
    '#pokerl-coach .prc-play{font-size:15px;font-weight:600;color:#fff}' +
    '#pokerl-coach .prc-sub{color:#93a0b4;font-size:11px;margin-bottom:8px}' +
    '#pokerl-coach .prc-split{display:flex;height:16px;border-radius:3px;overflow:hidden;' +
    'margin-bottom:6px;background:#39414d}' +
    '#pokerl-coach .prc-attack{background:#d9564f}#pokerl-coach .prc-switch{background:#3d8bd4}' +
    '#pokerl-coach .prc-split span{font-size:10px;line-height:16px;text-align:center;color:#fff;' +
    'white-space:nowrap;overflow:hidden;display:block}' +
    '#pokerl-coach .prc-section{margin-top:10px}' +
    '#pokerl-coach .prc-label{text-transform:uppercase;font-size:9px;letter-spacing:.08em;color:#7d8a9e;margin-bottom:3px}' +
    '#pokerl-coach .prc-why{background:#2a313b;border-left:3px solid #6bb8f0;padding:6px 8px;' +
    'margin-bottom:5px;border-radius:0 4px 4px 0}' +
    '#pokerl-coach ul{margin:0;padding-left:16px}#pokerl-coach li{margin-bottom:3px}' +
    '#pokerl-coach .prc-opt{display:flex;justify-content:space-between}' +
    '#pokerl-coach .prc-opt span:last-child{color:#93a0b4}' +
    '#pokerl-coach .prc-foot{padding:6px 10px;border-top:1px solid #39414d;font-size:10px;color:#7d8a9e;text-align:center}' +
    '#pokerl-coach .prc-forced{color:#f0b429}' +
    '.prc-recommended{outline:3px solid #6ddb8f !important;outline-offset:1px;border-radius:4px}';

  document.head.appendChild(style);
  document.body.appendChild(panel);

  document.getElementById('prc-head').onclick = function () {
    var collapsed = panel.classList.toggle('prc-collapsed');
    document.getElementById('prc-toggle').innerHTML = collapsed ? '&plus;' : '&minus;';
  };

  function setStatus(text, cls) {
    var el = document.getElementById('prc-status');
    if (el) { el.textContent = text; el.className = 'prc-status ' + (cls || ''); }
  }

  function esc(t) { var d = document.createElement('div'); d.textContent = t == null ? '' : String(t); return d.innerHTML; }
  function pct(v) { return Math.round((v || 0) * 100) + '%'; }

  // ---------------------------------------------------------------- coach socket

  function connect() {
    socket = new WebSocket(COACH_URL);
    socket.onopen = function () {
      ready = true;
      setStatus('connected', 'ok');
      while (queue.length) socket.send(queue.shift());
    };
    socket.onmessage = function (e) {
      var advice; try { advice = JSON.parse(e.data); } catch (err) { return; }
      render(advice);
    };
    socket.onclose = function () {
      ready = false;
      setStatus('coach offline — is pokerl.coach.server running?', 'warn');
      setTimeout(connect, 3000);
    };
    socket.onerror = function () {};
  }

  function send(payload) {
    var text = JSON.stringify(payload);
    if (ready && socket.readyState === 1) socket.send(text);
    else if (queue.length < 50) queue.push(text);
  }

  // ---------------------------------------------------------------- frame hook

  function handleFrame(data) {
    if (typeof data !== 'string' || data.charAt(0) !== '>') return;
    var nl = data.indexOf('\n');
    if (nl < 0) return;

    var tag = data.substr(1, nl - 1).trim();
    if (tag.indexOf('battle-') !== 0) return;
    if (tag.indexOf('battle-gen9randombattle-') !== 0) {
      setStatus('coach only knows gen9randombattle', 'warn');
      return;
    }
    if (!buffers[tag]) buffers[tag] = [];

    var request = null;
    var lines = data.substr(nl + 1).split('\n');
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (!line || line.charAt(0) !== '|') continue;
      if (line.indexOf('|request|') === 0) {
        var raw = line.slice(9);
        if (raw.trim()) { try { request = JSON.parse(raw); } catch (err) {} }
      } else {
        buffers[tag].push(line);
      }
    }

    if (request) {
      send({ battle_tag: tag, lines: buffers[tag].splice(0), request: request });
      setStatus('thinking…');
    }
  }

  window.app.receive = function (data) {
    try { handleFrame(data); } catch (err) { console.error('[pokerl-coach]', err); }
    return originalReceive.apply(this, arguments);
  };

  // ---------------------------------------------------------------- rendering

  function clearHighlight() {
    var marked = document.querySelectorAll('.prc-recommended');
    for (var i = 0; i < marked.length; i++) marked[i].classList.remove('prc-recommended');
  }

  function normalise(t) { return (t || '').toLowerCase().replace(/[^a-z0-9]/g, ''); }

  // Matched on button text, not on an action index: the client numbers its buttons by the
  // request's move order while poke-env's action space has its own indexing, and a silent
  // off-by-one would confidently highlight the wrong move.
  function highlight(rec) {
    clearHighlight();
    if (!rec) return;
    var wanted = rec.kind === 'switch'
      ? normalise(rec.label.replace('switch to ', ''))
      : normalise(rec.label.replace(' (terastallise)', ''));
    if (!wanted) return;
    var buttons = document.querySelectorAll(
      rec.kind === 'switch' ? 'button[name="chooseSwitch"]' : 'button[name="chooseMove"]');
    for (var i = 0; i < buttons.length; i++) {
      var label = normalise(buttons[i].textContent);
      if (label && (label === wanted || label.indexOf(wanted) === 0)) {
        buttons[i].classList.add('prc-recommended');
        return;
      }
    }
  }

  function render(a) {
    var body = document.getElementById('prc-body');
    if (!body) return;

    if (a.error) { setStatus('error', 'warn'); body.innerHTML = '<p class="prc-empty">' + esc(a.error) + '</p>'; return; }
    if (a.finished) { setStatus('battle over', 'ok'); body.innerHTML = '<p class="prc-empty">Battle finished.</p>'; clearHighlight(); return; }
    if (a.waiting) { setStatus('waiting', 'ok'); return; }

    setStatus('turn ' + a.turn + ' · ' + a.latency_ms + ' ms', 'ok');
    var rec = a.recommendation, html = '';

    if (a.forced) {
      html += '<div class="prc-play prc-forced">Forced switch</div><div class="prc-sub">' +
              'Your Pokémon fainted, so this is not a choice. It would bring in ' +
              esc(rec.label.replace('switch to ', '')) + '.</div>';
    } else {
      html += '<div class="prc-play">' + esc(rec.label) + '</div>' +
              '<div class="prc-sub">' + pct(rec.probability) + ' · ' + esc(a.confidence) +
              ' · eval ' + a.value.toFixed(1) + '</div>' +
              '<div class="prc-split"><div class="prc-attack" style="width:' + pct(a.p_attack) +
              '"><span>' + pct(a.p_attack) + ' attack</span></div><div class="prc-switch" style="width:' +
              pct(a.p_switch) + '"><span>' + pct(a.p_switch) + ' switch</span></div></div>';
      if (a.best_move && a.best_switch) {
        html += '<div class="prc-sub">best attack: ' + esc(a.best_move.label) +
                ' · best switch: ' + esc(a.best_switch.label) + '</div>';
      }
    }

    if (a.counterfactuals && a.counterfactuals.length) {
      html += '<div class="prc-section"><div class="prc-label">Why</div>';
      for (var i = 0; i < a.counterfactuals.length; i++)
        html += '<div class="prc-why">' + esc(a.counterfactuals[i].sentence) + '</div>';
      html += '</div>';
    }
    if (a.facts && a.facts.length) {
      html += '<div class="prc-section"><div class="prc-label">On the board</div><ul>';
      for (var j = 0; j < a.facts.length; j++) html += '<li>' + esc(a.facts[j]) + '</li>';
      html += '</ul></div>';
    }
    if (a.attributions && a.attributions.length) {
      html += '<div class="prc-section"><div class="prc-label">Weighing most</div><ul>';
      for (var k = 0; k < a.attributions.length; k++)
        html += '<li>' + esc(a.attributions[k].bucket) + ' (' + pct(Math.abs(a.attributions[k].weight)) + ')</li>';
      html += '</ul></div>';
    }
    if (a.options && a.options.length > 1) {
      html += '<div class="prc-section"><div class="prc-label">All options</div>';
      for (var n = 0; n < a.options.length; n++)
        html += '<div class="prc-opt"><span>' + esc(a.options[n].label) + '</span><span>' +
                pct(a.options[n].probability) + '</span></div>';
      html += '</div>';
    }

    body.innerHTML = html;
    highlight(rec);
  }

  window.__pokerlCoach = {
    detach: function () {
      window.app.receive = originalReceive;
      if (socket) { socket.onclose = null; socket.close(); }
      var el = document.getElementById('pokerl-coach');
      if (el) el.remove();
      clearHighlight();
      console.log('[pokerl-coach] detached');
    }
  };

  connect();
  console.log('[pokerl-coach] attached. Advice appears on your next turn. ' +
              'Run __pokerlCoach.detach() to remove.');
  }
})();
