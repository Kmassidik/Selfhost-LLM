# How to add a new glicc.id subdomain to the box's tunnel

What this covers: pointing a new `<name>.glicc.id` hostname at a local port on this box,
using the existing Cloudflare Tunnel (`dalang-music-box`) and the credentials already on
the box. This is the pattern `add_image_subdomain.sh` and `add_llminferences_subdomain.sh`
both follow — read this before writing a third one from scratch.

## The pieces, and where they live

| Piece | Where | Notes |
|---|---|---|
| API credentials | `/root/.env.cloudlfare` (note the typo in the filename — it's real, don't "fix" it without updating every script that reads it) | `DOMAIN`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_ZONE_ID`, `CLOUDFLARE_TUNNEL_ID`. The last of these is **stale** — it points at an unrelated tunnel (`mac-studio`). Nothing reads it; the scripts look up the tunnel by **name**, not by this ID. Don't trust it. |
| The tunnel that actually serves this box | Named `dalang-music-box` in Cloudflare, id `7d57d508-c1a0-4aa5-9c7b-177388620390` as of 2026-10-23. Confirm by listing tunnels and checking which one's ingress already has `llm.glicc.id` in it (see "Before you touch anything" below) — **don't assume the name is still right**, because this account has 5 tunnels and at least one other (`Glicc AI Server`) looks superficially similar and belongs to a different machine entirely. |
| The connector running on this box | `systemctl status cloudflared` — reads `/etc/cloudflared/token`, a *connector* token (authenticates the daemon to the tunnel), completely different from the API token above. You cannot use this one to call the Cloudflare API. Ingress rules for a token-based tunnel like this live in Cloudflare's control plane, not in a local `config.yml` — there isn't one on this box. |
| The pattern script | `/root/add_image_subdomain.sh` (original) and `/root/add_llminferences_subdomain.sh` (same script, `HOST`/`SERVICE` changed) | Copy one of these and edit the two variables rather than writing a new script. |

## Before you touch anything: check the token still works

API tokens for this expire/get rotated without warning (happened once already, 2026-10-23 —
the token was valid as of 2026-10-01, dead three weeks later with no local trace of why).

```bash
cd /root && set -a; . .env.cloudlfare; set +a
curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/user/tokens/verify"
```

`"success":true` and `"status":"active"` means good. `"Invalid API Token"` means it's dead —
tell the user, they generate a new one (Cloudflare dashboard &rarr; profile &rarr; API Tokens
&rarr; Create Token, scoped to `Account: Cloudflare Tunnel:Edit` + `Zone: DNS:Edit` for
`glicc.id`), they paste it into `.env.cloudlfare` themselves (`nano ~/.env.cloudlfare` or
`sed -i "s/^CLOUDFLARE_API_TOKEN=.*/CLOUDFLARE_API_TOKEN=NEWVALUE/" ~/.env.cloudlfare`) — do
not ask them to paste the raw token into chat, there is never a need to see it directly.

## Before you touch anything, part 2: confirm which tunnel actually owns the hostname

```bash
cd /root && set -a; . .env.cloudlfare; set +a
API="https://api.cloudflare.com/client/v4"
curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel?is_deleted=false" \
  | python3 -c "import sys,json; [print(t['name'],t['id']) for t in json.load(sys.stdin)['result']]"
```

Then, for the candidate tunnel's id:

```bash
curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel/<TUNNEL_ID>/configurations" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); [print(r.get('hostname'),'->',r.get('service')) for r in (d['result'] or {}).get('config',{}).get('ingress',[])]"
```

Look for `llm.glicc.id` (or whichever existing hostname you know is right) in the output —
that confirms you've got the right tunnel before you write to it.

## Adding the new hostname

```bash
cd /root
sed "s/^HOST=.*/HOST=\"yourname\"/; s|^SERVICE=.*|SERVICE=\"http://127.0.0.1:YOURPORT\"|" \
  add_image_subdomain.sh > add_yourname_subdomain.sh
chmod +x add_yourname_subdomain.sh
bash add_yourname_subdomain.sh
```

What it does, in order: looks up the tunnel by name &rarr; reads its current ingress rules
&rarr; merges in your new hostname (keeping every existing one, dropping any old rule for the
*same* hostname so re-running is safe) &rarr; `PUT`s the merged config back &rarr; creates or
updates the CNAME DNS record (`<name>.glicc.id` &rarr; `<tunnel_id>.cfargotunnel.com`,
proxied) &rarr; prints `LIVE: https://<name>.glicc.id/`.

## Verifying it

DNS propagation and your own machine's DNS cache are two different things — a fresh record
can resolve everywhere except your own terminal for a few minutes if you queried it (and got
a negative/NXDOMAIN answer) before Cloudflare finished provisioning. If `curl` says
"Could not resolve host" right after creation but `dig` or `host` already show the right IPs,
bypass your local cache to confirm the service itself is fine before troubleshooting further:

```bash
curl -s --resolve <name>.glicc.id:443:<ip-from-dig> https://<name>.glicc.id/
```

A `401` (if the backend needs auth) or `200` means it's genuinely live — the earlier
resolution failure was local-cache noise, not a real problem. It clears on its own; flushing
the local DNS cache (`dscacheutil -flushcache` on macOS) speeds that up but needs `sudo`.

## What NOT to do

- Don't hardcode a tunnel ID anywhere new — always resolve by name, and verify by ingress
  content, because stale/wrong IDs already exist in this exact `.env` file and nothing
  catches that silently.
- Don't skip the "which tunnel actually owns this hostname" check just because a script's
  `TUNNEL_NAME` variable looks plausible. It happened to be right this time; verifying first
  is one curl call and costs nothing.
- Don't write the API token into any file this project tracks in git. It already lives at
  `/root/.env.cloudlfare`, outside any repo; that's the right place for it to stay.
