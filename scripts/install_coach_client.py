"""Serve the Showdown client from the local server, with the coach already loaded.

Without this, using the coach means an action every session -- pasting a script into the
console, or installing a browser extension. This makes it permanent: open
http://localhost:8000 and the client comes up with the panel already attached.

How it works
------------
The local server's only static page is a redirect that bounces you to the hosted client at
psim.us (see server/static/index.html). This replaces that page with a real client: the same
markup psim.us serves, loading the same scripts from play.pokemonshowdown.com, but served from
localhost and with two things injected.

1. A server override. ``Config.defaultserver`` points at sim3.psim.us:443, and the client only
   applies it when the address bar matches ``Config.routes.client``. Served from localhost
   neither holds, so the client would either refuse to start or connect to the *official*
   server. Pointing both at localhost fixes that, and it has a second effect that matters more:
   the client bypasses SockJS and opens a direct WebSocket when -- and only when --
   ``Config.server.host`` is exactly ``'localhost'``. SockJS makes HTTP requests to loopback,
   which browsers block from a public site (Private Network Access, hence "Couldn't connect to
   server!" on 127-0-0-1.insecure.psim.us). A direct WebSocket to loopback is still allowed.

2. The coach script, loaded from the same origin. ``--no-coach`` leaves it out, which gives a
   plain local client for playing against an agent (scripts/play_vs_agent.py) without a panel
   that reports "coach offline" whenever the coach backend is not running.

Because the page is now served from localhost, the page and the server are both loopback, so
Private Network Access does not apply at all and no browser permission is involved.

The original page is kept as index.html.orig; ``--revert`` puts it back.

Usage:
    python scripts/install_coach_client.py
    python scripts/install_coach_client.py --no-coach
    python scripts/install_coach_client.py --revert
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "pokemon-showdown" / "server" / "static"
INDEX = STATIC / "index.html"
BACKUP = STATIC / "index.html.orig"
COACH_SRC = ROOT / "client" / "coach-console.js"
COACH_DEST = STATIC / "pokerl-coach.js"

CLIENT_URL = "https://play.pokemonshowdown.com/"
CONFIG_TAG = "play.pokemonshowdown.com/config/config.js"

OVERRIDE = """
<!-- PokeRL: point the client at this server instead of the official one. -->
<script>
(function () {
  var host = document.location.hostname;                       // "localhost"
  var port = parseInt(document.location.port || '8000', 10);   // 8000

  // testclient is the client's own mode for running against a local server, and without it
  // this page does not work. Storage.initPrefs compares the page origin against
  // play.pokemonshowdown.com; served from localhost that fails, so it takes the cross-origin
  // path, opens a hidden iframe to psim.us to sync prefs, and blocks on a handshake that also
  // overwrites Config.server (storage.js: `case 'c': Config.server = JSON.parse(...)`).
  // Symptom: Format and Team stuck on "Loading...", and "Connecting..." forever.
  // Storage.initTestClient instead keeps our Config.server and loads prefs and teams locally.
  Config.testclient = true;

  // Config.routes.client is deliberately left alone. Dex.resourcePrefix is built from it
  // ("//" + Config.routes.client + "/"), so pointing it at localhost sends every sprite,
  // avatar, the logo and data/teambuilder-tables.js to this server, which does not have them:
  // the UI loses its images and the challenge dialog cannot open. It only needed overriding
  // for a `hostname === Config.routes.client` check that Config.testclient already satisfies.

  // host must be exactly "localhost" for the client to bypass SockJS and open a direct
  // WebSocket -- SockJS's HTTP requests to loopback are what browsers block.
  Config.defaultserver = {
    id: 'localhost',
    host: host,
    port: port,
    httpport: port,
    altport: port,
    registered: false
  };
  Config.server = Config.defaultserver;
})();
</script>
"""

COACH_INCLUDE = """
<!-- PokeRL coach panel. Waits for the client to boot, then attaches. -->
<script src="/pokerl-coach.js"></script>
"""

STORAGE_TAG = "oldclient/storage.js"

# Injected immediately after storage.js, which is the only window where this works.
NO_PROXY_POPUP = r"""
<!-- PokeRL: answer the login server locally; this server runs --no-security. -->
<script>
(function () {
  // Two problems this solves, both from running the client off its own domain.
  //
  // 1. Storage.initTestClient replaces $.get/$.post with versions that open a manual
  //    copy-paste popup whenever there is no session key -- for every request, not just
  //    logins ("Because of your browser's security restrictions for testclient.html...").
  //
  // 2. app.user.loaded is set only inside the success callback of
  //    $.post(getActionPHP(), {act: 'upkeep'}). Under testclient that URL is
  //    https://play.pokemonshowdown.com/~~localhost/action.php, which replies 200 but sends
  //    no Access-Control-Allow-Origin, so the browser blocks it, the callback never runs and
  //    the username button sits on "Loading..." forever with no way to choose a name.
  //
  // We need none of it: with --no-security the server accepts any name via /trn and does not
  // validate the assertion. So action.php is answered here, with no network call at all.
  //
  // Registered accounts are therefore not supported on this local server -- which is the
  // point of --no-security.
  if (typeof $ === 'undefined' || typeof Storage === 'undefined') return;
  var get = $.get, post = $.post;

  function localAction(data, callback) {
    var act = data && data.act;
    var reply;
    if (act === 'upkeep') {
      // Shape the client expects: a ']' sentinel then JSON. No username => "Choose name".
      reply = ']{"assertion":"","username":"","loggedin":false}';
    } else if (act === 'getassertion') {
      // finishRename rejects an empty assertion, and anything containing '<' or a newline,
      // with "Something is interfering with our connection to the login server." So a
      // placeholder gets it past the client; see finishRename below for why it never
      // reaches the server.
      reply = 'pokerl-local';
    } else {
      reply = ']{}';
    }
    if (typeof callback === 'function') setTimeout(function () { callback(reply); }, 0);
    return true;
  }

  function wrap(orig) {
    return function (uri, data, callback, type) {
      if (typeof data === 'function') { type = callback; callback = data; data = undefined; }
      if (typeof uri === 'string' && uri.indexOf('action.php') !== -1) {
        return localAction(data, callback);
      }
      // Relative paths (testclient loads data/text/<lang>.js this way) belong to the client's
      // own resource host, not to this server.
      if (typeof uri === 'string' && uri.charAt(0) !== '/' &&
          !/^(https?:)?\/\//.test(uri) && window.Dex && Dex.resourcePrefix) {
        uri = Dex.resourcePrefix + uri;
      }
      return orig(uri, data, callback, type);
    };
  }

  // initTestClient installs its wrapper from a whenAppLoaded callback registered while
  // storage.js ran. Trackers fire callbacks in registration order, so registering here --
  // after storage.js, before client.js -- means ours runs afterwards and wins.
  Storage.whenAppLoaded(function (app) {
    $.get = wrap(get);
    $.post = wrap(post);

    // The server skips token verification only for an *empty* token (--no-security sets
    // Config.noguestsecurity; users.ts validateToken). The placeholder above would be
    // verified as a real assertion and refused with "The assertion you sent us is corrupt
    // or incorrect", so the name is never taken. Send it empty instead.
    var finishRename = app.user.finishRename;
    app.user.finishRename = function (name, assertion) {
      if (assertion !== 'pokerl-local') return finishRename.apply(this, arguments);
      app.trigger('loggedin');
      app.send('/trn ' + name + ',0,');
    };
  });
})();
</script>
"""


def install(coach: bool = True) -> None:
    if coach and not COACH_SRC.exists():
        raise SystemExit(f"missing {COACH_SRC}")
    if not STATIC.is_dir():
        raise SystemExit(f"missing {STATIC} -- run scripts/setup_showdown.ps1 first")

    print(f"fetching client markup from {CLIENT_URL}")
    # Cloudflare serves 403 to urllib's default User-Agent, so send a browser one.
    request = urllib.request.Request(
        CLIENT_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", "replace")
    print(f"  {len(html):,} bytes")

    if CONFIG_TAG not in html:
        raise SystemExit(
            "could not find the config.js script tag in the fetched page; the client's "
            "markup has changed and this script needs updating"
        )

    # The override must run after config.js has defined Config, and before client.js reads it.
    marker_end = html.index(CONFIG_TAG)
    insert_at = html.index("</script>", marker_end) + len("</script>")
    html = html[:insert_at] + OVERRIDE + html[insert_at:]

    if STORAGE_TAG not in html:
        raise SystemExit(
            "could not find the storage.js script tag; the client's markup has changed"
        )
    storage_end = html.index("</script>", html.index(STORAGE_TAG)) + len("</script>")
    html = html[:storage_end] + NO_PROXY_POPUP + html[storage_end:]

    if coach:
        if "</body>" in html:
            html = html.replace("</body>", COACH_INCLUDE + "</body>", 1)
        else:
            html += COACH_INCLUDE

    if not BACKUP.exists():
        shutil.copy2(INDEX, BACKUP)
        print(f"  backed up original -> {BACKUP.name}")

    INDEX.write_text(html, encoding="utf-8")
    print(f"  wrote {INDEX} ({len(html):,} bytes)")

    if not coach:
        # A panel left over from an earlier install is harmless (nothing loads it) but stale.
        COACH_DEST.unlink(missing_ok=True)
        print()
        print("Open http://localhost:8000 -- the plain client, no coach panel.")
        return

    shutil.copy2(COACH_SRC, COACH_DEST)
    print(f"  wrote {COACH_DEST}")
    print()
    print("Open http://localhost:8000 -- the client loads with the coach attached.")
    print("The coach backend must be running:")
    print("  python -m pokerl.coach.server --run results/ppo_v1_selfplay_seed0")


def revert() -> None:
    if BACKUP.exists():
        shutil.copy2(BACKUP, INDEX)
        BACKUP.unlink()
        print(f"restored {INDEX}")
    else:
        print("no backup found; nothing to restore")
    if COACH_DEST.exists():
        COACH_DEST.unlink()
        print(f"removed {COACH_DEST}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true", help="restore the original redirect page")
    ap.add_argument("--no-coach", action="store_true", help="serve the client without the panel")
    args = ap.parse_args()
    revert() if args.revert else install(coach=not args.no_coach)


if __name__ == "__main__":
    main()
