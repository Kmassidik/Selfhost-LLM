#!/usr/bin/env bash
# Add/update a public hostname on the dalang-music-box tunnel, MERGING into the existing
# ingress so other hostnames (music.glicc.id) are preserved. Never prints a secret.
#   HOST=llm SERVICE=http://127.0.0.1:8081 ./deploy/cf-add-hostname.sh
set -euo pipefail
ENV_FILE="${ENV_FILE:-/root/.env.cloudlfare}"; TUNNEL_NAME="${TUNNEL_NAME:-dalang-music-box}"
HOST="${HOST:?set HOST=<subdomain>}"; SERVICE="${SERVICE:?set SERVICE=http://127.0.0.1:PORT}"
[ -f "$ENV_FILE" ] || { echo "no $ENV_FILE"; exit 1; }
set -a; . "$ENV_FILE"; set +a
API="https://api.cloudflare.com/client/v4"
auth=(-H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json")
TID=$(curl -s --max-time 30 "${auth[@]}" "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel?is_deleted=false" | python3 -c "import sys,json;d=json.load(sys.stdin);print(next((t[\"id\"] for t in d[\"result\"] if t[\"name\"]==\"$TUNNEL_NAME\"),\"\"))")
[ -n "$TID" ] || { echo "tunnel $TUNNEL_NAME not found"; exit 1; }
echo "tunnel $TUNNEL_NAME = ${TID:0:8}..."
CUR=$(curl -s --max-time 30 "${auth[@]}" "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel/$TID/configurations")
echo "current ingress:"; echo "$CUR" | python3 -c "import sys,json;[print(\"  \",r.get(\"hostname\",\"(catch-all)\"),\"->\",r.get(\"service\")) for r in json.load(sys.stdin).get(\"result\",{}).get(\"config\",{}).get(\"ingress\",[])]"
NEW=$(echo "$CUR" | HOST="$HOST" DOMAIN="$DOMAIN" SERVICE="$SERVICE" python3 -c "import sys,json,os
d=json.load(sys.stdin);ing=d.get(\"result\",{}).get(\"config\",{}).get(\"ingress\",[])
h=os.environ[\"HOST\"]+\".\"+os.environ[\"DOMAIN\"]
k=[r for r in ing if r.get(\"hostname\") and r[\"hostname\"]!=h]
k+=[{\"hostname\":h,\"service\":os.environ[\"SERVICE\"]},{\"service\":\"http_status:404\"}]
print(json.dumps({\"config\":{\"ingress\":k}}))")
curl -s --max-time 30 -X PUT "${auth[@]}" "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/cfd_tunnel/$TID/configurations" -d "$NEW" >/dev/null && echo "ingress merged (music preserved + $HOST added)"
REC=$(curl -s --max-time 30 "${auth[@]}" "$API/zones/$CLOUDFLARE_ZONE_ID/dns_records?name=$HOST.$DOMAIN" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[\"result\"][0][\"id\"] if d[\"result\"] else \"\")")
BODY="{\"type\":\"CNAME\",\"name\":\"$HOST\",\"content\":\"$TID.cfargotunnel.com\",\"proxied\":true}"
if [ -z "$REC" ]; then curl -s --max-time 30 -X POST "${auth[@]}" "$API/zones/$CLOUDFLARE_ZONE_ID/dns_records" -d "$BODY" >/dev/null; echo "dns created"; else curl -s --max-time 30 -X PUT "${auth[@]}" "$API/zones/$CLOUDFLARE_ZONE_ID/dns_records/$REC" -d "$BODY" >/dev/null; echo "dns updated"; fi
echo "live in ~30s: https://$HOST.$DOMAIN"
