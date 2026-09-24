// ==UserScript==
// @name         PokeRL Coach
// @namespace    pokerl
// @version      0.1.0
// @description  Live RL coaching panel for a local Pokemon Showdown battle.
// @match        *://play.pokemonshowdown.com/*
// @match        *://*.psim.us/*
// @match        http://localhost:8000/*
// @grant        none
// @run-at       document-start
// ==/UserScript==

/*
 * Why a userscript rather than a fork of the client?
 *
 * The local Showdown server ships no UI at all -- server/static/index.html just redirects
 * localhost:8000 to the hosted client at psim.us, which then connects back to your local
 * server. So there is no local client source to edit. Hooking the socket from the page gets a
 * working panel without building smogon/pokemon-showdown-client, which is the right trade
 * while the *content* of the advice is still being iterated on. Fork the client later, once
 * what the panel should say has settled.
 *
 * Why hook the socket instead of scraping the DOM?
 *
 * The coach needs the private |request| payload -- your team, your moves, your PP. Showdown
 * only ever sends that to the player themselves, which is also why a spectator bot cannot do
 * this job. It arrives on the websocket and is never fully rendered into the page.
 *
 * This script is advice-only. It never sends a |/choose to Showdown; the human plays every
 * move themselves.
 */

(function () {
  'use strict';

  const COACH_URL = 'ws://localhost:8765';
  const RECONNECT_MS = 3000;

  let coachSocket = null;
  let coachReady = false;
  const pending = [];          // battle_tag -> queued payloads while the coach is down
  const lineBuffer = {};       // battle_tag -> protocol lines not yet forwarded

  // ---------------------------------------------------------------------------------
  // Coach connection
  // ---------------------------------------------------------------------------------

  function connectCoach() {
    coachSocket = new WebSocket(COACH_URL);

    coachSocket.onopen = () => {
      coachReady = true;
      setStatus('connected', 'ok');
      while (pending.length) coachSocket.send(pending.shift());
    };

    coachSocket.onmessage = (event) => {
      let advice;
      try {
        advice = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      render(advice);
    };

    coachSocket.onclose = () => {
      coachReady = false;
      setStatus('coach offline - retrying', 'warn');
      setTimeout(connectCoach, RECONNECT_MS);
    };

    coachSocket.onerror = () => { /* onclose follows and handles the retry */ };
  }

  function sendToCoach(payload) {
    const text = JSON.stringify(payload);
    if (coachReady && coachSocket.readyState === WebSocket.OPEN) {
      coachSocket.send(text);
    } else if (pending.length < 50) {
      pending.push(text);
    }
  }

  // ---------------------------------------------------------------------------------
  // Showdown socket interception
  // ---------------------------------------------------------------------------------

  // Showdown frames look like:
  //   >battle-gen9randombattle-123
  //   |move|p1a: Rotom|Thunderbolt|p2a: Dragapult
  //   |turn|5
  // A frame with no leading '>' belongs to the global/lobby stream, not a battle room.
  function handleServerFrame(data) {
    if (typeof data !== 'string' || data[0] !== '>') return;

    const newline = data.indexOf('\n');
    if (newline === -1) return;

    const tag = data.slice(1, newline).trim();
    if (!tag.startsWith('battle-')) return;

    // The trained policy and both encoders assume gen9randombattle; anything else would get
    // advice from a model that has never seen the format.
    if (!tag.startsWith('battle-gen9randombattle-')) {
      setStatus('unsupported format - coach only knows gen9randombattle', 'warn');
      return;
    }

    if (!lineBuffer[tag]) lineBuffer[tag] = [];

    let request = null;
    for (const line of data.slice(newline + 1).split('\n')) {
      if (!line.startsWith('|')) continue;
      if (line.startsWith('|request|')) {
        const payload = line.slice('|request|'.length);
        if (payload.trim()) {
          try {
            request = JSON.parse(payload);
          } catch (err) { /* keep the buffered lines; skip the malformed request */ }
        }
      } else {
        lineBuffer[tag].push(line);
      }
    }

    // A |request| is Showdown asking for a decision, which is exactly when advice is useful.
    if (request) {
      sendToCoach({ battle_tag: tag, lines: lineBuffer[tag].splice(0), request: request });
      setStatus('thinking...', 'busy');
    }
  }

  const NativeWebSocket = window.WebSocket;

  function PatchedWebSocket(url, protocols) {
    const socket = protocols === undefined
      ? new NativeWebSocket(url)
      : new NativeWebSocket(url, protocols);

    // Only the Showdown server socket matters; leave the coach's own socket alone.
    if (typeof url === 'string' && url.indexOf('8765') === -1) {
      socket.addEventListener('message', (event) => {
        try {
          handleServerFrame(event.data);
        } catch (err) {
          console.error('[pokerl-coach]', err);
        }
      });
    }
    return socket;
  }

  PatchedWebSocket.prototype = NativeWebSocket.prototype;
  PatchedWebSocket.CONNECTING = NativeWebSocket.CONNECTING;
  PatchedWebSocket.OPEN = NativeWebSocket.OPEN;
  PatchedWebSocket.CLOSING = NativeWebSocket.CLOSING;
  PatchedWebSocket.CLOSED = NativeWebSocket.CLOSED;
  window.WebSocket = PatchedWebSocket;

  // ---------------------------------------------------------------------------------
  // Panel
  // ---------------------------------------------------------------------------------

  let panel = null;

  function buildPanel() {
    panel = document.createElement('div');
    panel.id = 'pokerl-coach';
    panel.innerHTML = `
      <div class="prc-head" id="prc-head" title="click to collapse">
        <span class="prc-title">PokeRL Coach</span>
        <span>
          <span class="prc-status" id="prc-status">starting</span>
          <span class="prc-toggle" id="prc-toggle">&minus;</span>
        </span>
      </div>
      <div class="prc-body" id="prc-body">
        <p class="prc-empty">Start a gen9randombattle. Advice appears when it is your turn.</p>
      </div>
      <div class="prc-foot" id="prc-foot">Advice only &mdash; you always play the move yourself.</div>
    `;

    const style = document.createElement('style');
    style.textContent = `
      #pokerl-coach {
        /* 58px, not 12px: the client's top bar holds the username button, and a panel at
           the very top covers it so you cannot choose a name. */
        position: fixed; top: 58px; right: 12px; width: 320px;
        max-height: calc(100vh - 70px);
        overflow-y: auto; z-index: 2147483000;
        font: 12px/1.45 -apple-system, "Segoe UI", Roboto, sans-serif;
        background: #1e232b; color: #e7ebf0;
        border: 1px solid #39414d; border-radius: 8px;
        box-shadow: 0 6px 22px rgba(0,0,0,.45);
      }
      #pokerl-coach .prc-head {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 10px; border-bottom: 1px solid #39414d; background: #252b34;
        border-radius: 8px 8px 0 0;
      }
      #pokerl-coach .prc-title { font-weight: 600; letter-spacing: .02em; }
      #pokerl-coach .prc-head { cursor: pointer; user-select: none; }
      #pokerl-coach .prc-toggle { font-size: 14px; color: #93a0b4; padding-left: 8px; }
      #pokerl-coach.prc-collapsed .prc-body,
      #pokerl-coach.prc-collapsed .prc-foot { display: none; }
      #pokerl-coach .prc-status { font-size: 10px; color: #93a0b4; }
      #pokerl-coach .prc-status.ok { color: #6ddb8f; }
      #pokerl-coach .prc-status.warn { color: #f0b429; }
      #pokerl-coach .prc-status.busy { color: #6bb8f0; }
      #pokerl-coach .prc-body { padding: 10px; }
      #pokerl-coach .prc-empty { color: #93a0b4; margin: 0; }
      #pokerl-coach .prc-play {
        font-size: 15px; font-weight: 600; color: #fff; margin-bottom: 2px;
      }
      #pokerl-coach .prc-sub { color: #93a0b4; font-size: 11px; margin-bottom: 8px; }
      #pokerl-coach .prc-split { display: flex; height: 16px; border-radius: 3px;
        overflow: hidden; margin-bottom: 4px; background: #39414d; }
      #pokerl-coach .prc-attack { background: #d9564f; }
      #pokerl-coach .prc-switch { background: #3d8bd4; }
      #pokerl-coach .prc-split span {
        font-size: 10px; line-height: 16px; text-align: center; color: #fff;
        white-space: nowrap; overflow: hidden;
      }
      #pokerl-coach .prc-section { margin-top: 10px; }
      #pokerl-coach .prc-label {
        text-transform: uppercase; font-size: 9px; letter-spacing: .08em;
        color: #7d8a9e; margin-bottom: 3px;
      }
      #pokerl-coach .prc-why {
        background: #2a313b; border-left: 3px solid #6bb8f0;
        padding: 6px 8px; margin-bottom: 5px; border-radius: 0 4px 4px 0;
      }
      #pokerl-coach ul { margin: 0; padding-left: 16px; }
      #pokerl-coach li { margin-bottom: 3px; }
      #pokerl-coach .prc-opt { display: flex; justify-content: space-between; }
      #pokerl-coach .prc-opt span:last-child { color: #93a0b4; }
      #pokerl-coach .prc-foot {
        padding: 6px 10px; border-top: 1px solid #39414d;
        font-size: 10px; color: #7d8a9e; text-align: center;
      }
      #pokerl-coach .prc-forced { color: #f0b429; }
      .prc-recommended {
        outline: 3px solid #6ddb8f !important;
        outline-offset: 1px; border-radius: 4px;
      }
    `;
    document.head.appendChild(style);
    document.body.appendChild(panel);

    document.getElementById('prc-head').onclick = () => {
      const collapsed = panel.classList.toggle('prc-collapsed');
      document.getElementById('prc-toggle').innerHTML = collapsed ? '&plus;' : '&minus;';
    };
  }

  function setStatus(text, cls) {
    const el = document.getElementById('prc-status');
    if (el) {
      el.textContent = text;
      el.className = 'prc-status ' + (cls || '');
    }
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
  }

  function pct(value) { return Math.round((value || 0) * 100) + '%'; }

  function render(advice) {
    const body = document.getElementById('prc-body');
    if (!body) return;

    if (advice.error) {
      setStatus('error', 'warn');
      body.innerHTML = '<p class="prc-empty">' + escapeHtml(advice.error) + '</p>';
      return;
    }
    if (advice.finished) {
      setStatus('battle over', 'ok');
      body.innerHTML = '<p class="prc-empty">Battle finished.</p>';
      clearHighlight();
      return;
    }
    if (advice.waiting) { setStatus('waiting', 'ok'); return; }

    setStatus('turn ' + advice.turn + ' - ' + advice.latency_ms + ' ms', 'ok');

    const rec = advice.recommendation;
    const attack = pct(advice.p_attack);
    const sw = pct(advice.p_switch);

    let html = '';

    if (advice.forced) {
      html += '<div class="prc-play prc-forced">Forced switch</div>'
           +  '<div class="prc-sub">Your Pokemon fainted, so this is not a choice. '
           +  'It would bring in ' + escapeHtml(rec.label.replace('switch to ', '')) + '.</div>';
    } else {
      html += '<div class="prc-play">' + escapeHtml(rec.label) + '</div>'
           +  '<div class="prc-sub">' + pct(rec.probability) + ' confidence &middot; '
           +  escapeHtml(advice.confidence) + ' &middot; eval ' + advice.value.toFixed(1) + '</div>';

      html += '<div class="prc-split">'
           +  '<div class="prc-attack" style="width:' + attack + '"><span>' + attack + ' attack</span></div>'
           +  '<div class="prc-switch" style="width:' + sw + '"><span>' + sw + ' switch</span></div>'
           +  '</div>';

      if (advice.best_move && advice.best_switch) {
        html += '<div class="prc-sub">best attack: ' + escapeHtml(advice.best_move.label)
             +  ' &middot; best switch: ' + escapeHtml(advice.best_switch.label) + '</div>';
      }
    }

    if (advice.counterfactuals && advice.counterfactuals.length) {
      html += '<div class="prc-section"><div class="prc-label">Why</div>';
      for (const cf of advice.counterfactuals) {
        html += '<div class="prc-why">' + escapeHtml(cf.sentence) + '</div>';
      }
      html += '</div>';
    }

    if (advice.facts && advice.facts.length) {
      html += '<div class="prc-section"><div class="prc-label">On the board</div><ul>';
      for (const fact of advice.facts) html += '<li>' + escapeHtml(fact) + '</li>';
      html += '</ul></div>';
    }

    if (advice.attributions && advice.attributions.length) {
      html += '<div class="prc-section"><div class="prc-label">Weighing most</div><ul>';
      for (const a of advice.attributions) {
        html += '<li>' + escapeHtml(a.bucket) + ' (' + pct(Math.abs(a.weight)) + ')</li>';
      }
      html += '</ul></div>';
    }

    if (advice.options && advice.options.length > 1) {
      html += '<div class="prc-section"><div class="prc-label">All options</div>';
      for (const opt of advice.options) {
        html += '<div class="prc-opt"><span>' + escapeHtml(opt.label) + '</span><span>'
             +  pct(opt.probability) + '</span></div>';
      }
      html += '</div>';
    }

    body.innerHTML = html;
    highlight(rec);
  }

  // ---------------------------------------------------------------------------------
  // Highlight the recommended button in the battle controls
  // ---------------------------------------------------------------------------------

  function clearHighlight() {
    document.querySelectorAll('.prc-recommended')
      .forEach((el) => el.classList.remove('prc-recommended'));
  }

  // Matched on the button's visible text rather than on an action index. The client's move
  // buttons are numbered by the request's move order while poke-env's action space has its
  // own indexing, and a silent off-by-one would highlight the wrong move -- worse than not
  // highlighting at all.
  function normalise(text) {
    return (text || '').toLowerCase().replace(/[^a-z0-9]/g, '');
  }

  function highlight(rec) {
    clearHighlight();
    if (!rec) return;

    const wanted = rec.kind === 'switch'
      ? normalise(rec.label.replace('switch to ', ''))
      : normalise(rec.label.replace(' (terastallise)', ''));
    if (!wanted) return;

    const selector = rec.kind === 'switch'
      ? 'button[name="chooseSwitch"]'
      : 'button[name="chooseMove"]';

    for (const button of document.querySelectorAll(selector)) {
      const label = normalise(button.textContent);
      if (label && (label === wanted || label.indexOf(wanted) === 0)) {
        button.classList.add('prc-recommended');
        return;
      }
    }
  }

  // ---------------------------------------------------------------------------------

  function start() {
    buildPanel();
    connectCoach();
    console.log('[pokerl-coach] ready; forwarding gen9randombattle rooms to ' + COACH_URL);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
