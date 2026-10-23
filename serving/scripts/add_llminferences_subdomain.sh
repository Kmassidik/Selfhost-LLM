#!/usr/bin/env bash
# Add image.glicc.id -> 127.0.0.1:8097 to the EXISTING box tunnel, without
# disturbing the audio (music) route. Merges the ingress; upserts the DNS.
set -euo pipefail
ENV_FILE="${ENV_FILE:-/root/.env.cloudlfare}"
TUNNEL_NAME="dalang-music-box"
HOST="llminferences"
SERVICE="http://127.0.0.1:8082"

[ -f "$ENV_FILE" ] || { echo "no $ENV_FILE"; exit 1; }
set -a; . "$ENV_FILE"; set +a
: "${DOMAIN:?}" "${CLOUDFLARE_API_TOKEN:?}" "${CLOUDFLARE_ACCOUNT_ID:?}" "${CLOUDFLARE_ZONE_ID:?}"
API="https://api.cloudflare.com/client/v4"
auth=(-H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json")
pg(){ python3 -c "import sys,json;d=json.load(sys.stdin);print($1)"; }

TID=$(curl -s --max-time 30 "${auth[@]}" "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel?is_deleted=false" \
      | pg "next((t['id'] for t in d['result'] if t['name']=='$TUNNEL_NAME'),'')")
[ -n "$TID" ] || { echo "tunnel $TUNNEL_NAME not found"; exit 1; }
echo "tunnel ${TID:0:8}… host $HOST.$DOMAIN -> $SERVICE"

CUR=$(curl -s --max-time 30 "${auth[@]}" "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel/$TID/configurations")
echo "existing hostnames:"; printf '%s' "$CUR" | pg "[r.get('hostname') for r in ((d.get('result') or {}).get('config') or {}).get('ingress',[])]"

NEW=$(printf '%s' "$CUR" | python3 -c "
import sys,json
host='$HOST.$DOMAIN'; svc='$SERVICE'
d=json.load(sys.stdin)
cfg=(d.get('result') or {}).get('config') or {}
ing=cfg.get('ingress') or [{'service':'http_status:404'}]
keep=[r for r in ing if r.get('hostname') and r.get('hostname')!=host]   # other hostnames, minus any old copy of ours
catch=[r for r in ing if 'hostname' not in r][:1] or [{'service':'http_status:404'}]
keep.append({'hostname':host,'service':svc})
cfg['ingress']=keep+catch
print(json.dumps({'config':cfg}))
")
OK=$(curl -s --max-time 30 -X PUT "${auth[@]}" "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel/$TID/configurations" -d "$NEW" | pg "d.get('success')")
echo "ingress updated: $OK"
printf '%s' "$NEW" | pg "[r.get('hostname','(catch-all)') for r in d['config']['ingress']]"

REC=$(curl -s --max-time 30 "${auth[@]}" "$API/zones/$CLOUDFLARE_ZONE_ID/dns_records?name=$HOST.$DOMAIN" | pg "d['result'][0]['id'] if d['result'] else ''")
BODY="{\"type\":\"CNAME\",\"name\":\"$HOST\",\"content\":\"$TID.cfargotunnel.com\",\"proxied\":true}"
if [ -z "$REC" ]; then curl -s --max-time 30 -X POST "${auth[@]}" "$API/zones/$CLOUDFLARE_ZONE_ID/dns_records" -d "$BODY" >/dev/null; echo "dns created"
else curl -s --max-time 30 -X PUT "${auth[@]}" "$API/zones/$CLOUDFLARE_ZONE_ID/dns_records/$REC" -d "$BODY" >/dev/null; echo "dns updated"; fi
echo "connector: $(systemctl is-active cloudflared)"
echo "LIVE: https://$HOST.$DOMAIN/"
