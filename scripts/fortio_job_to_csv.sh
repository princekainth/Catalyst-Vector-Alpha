#!/usr/bin/env bash
set -euo pipefail

LABEL=${1:-cva-enabled-3k}
QPS=${2:-3000}
THREADS=${3:-60}
DUR=${4:-60s}
SVC=${5:-http://nginx.default.svc.cluster.local/}

PREFIX=${LABEL%%-*}
STAMP=$(date +%Y%m%d-%H%M%S)

qps_tag() { local q="$1"; if (( q % 1000 == 0 )); then echo "$((q/1000))k"; else echo "${q}qps"; fi; }

JOB="${PREFIX}-json-${STAMP}"
QPS_TAG="$(qps_tag "$QPS")"
OUTDIR="results/${PREFIX}"
OUT="${PREFIX}_${STAMP}_${QPS_TAG}.json"
CSV="${PREFIX}_runs.csv"
mkdir -p "${OUTDIR}"

# 1) Fortio Job
kubectl create job "$JOB" --image=fortio/fortio -- fortio load -config-dir disabled -c "$THREADS" -qps "$QPS" -t "$DUR" -labels "$LABEL" -json - "$SVC"
kubectl patch job "$JOB" -p '{"spec":{"backoffLimit":0}}'
kubectl wait --for=condition=complete "job/$JOB" --timeout=6m >/dev/null

# 3) Extract JSON block robustly
LOG_CONTENT="$(kubectl logs "job/$JOB")"
JSON_START=$(echo "$LOG_CONTENT" | grep -n -m1 '{' | cut -d: -f1 || true)
if [[ -z "$JSON_START" ]]; then
  echo "❌ Could not locate JSON start in logs"
  exit 1
fi
JSON="$(echo "$LOG_CONTENT" | tail -n +$JSON_START)"

# 4) Save artifact
echo "$JSON" > "${OUTDIR}/${OUT}"

# 5) CSV header
if ! [[ -s "$CSV" ]]; then
  echo 'id,label,actual_qps,p50,p95,p99,count,errors,error_rate' > "$CSV"
fi

# 6) Extract metrics
ID="$(date +%Y-%m-%d-%H%M%S)_${LABEL//-/_}"
ACTUAL_QPS="$(jq -r '.RequestedQPS' <<<"$JSON")"
P50="$(jq -r '.DurationHistogram.Percentiles[] | select(.Percentile==50) | .Value' <<<"$JSON")"
P95="$(jq -r '.DurationHistogram.Percentiles[] | select(.Percentile==95) | .Value' <<<"$JSON")"
P99="$(jq -r '.DurationHistogram.Percentiles[] | select(.Percentile==99) | .Value' <<<"$JSON")"
COUNT="$(jq -r '.Exactly.Count' <<<"$JSON" 2>/dev/null || jq -r '.RetCodes | to_entries | map(.value) | add' <<<"$JSON")"
ERRORS="$(jq -r '[.RetCodes | to_entries[] | select(.key|tostring!="200") | .value] | add // 0' <<<"$JSON")"

# 7) Error rate
ERR_RATE="$(python3 - <<'PYIN'
c=${COUNT:-0}
e=${ERRORS:-0}
print((e/float(c)) if float(c) else 0.0)
PYIN
)"

# 8) Write CSV row
printf '%q,%q,%s,%s,%s,%s,%s,%s,%s
'   "$ID" "$LABEL" "$ACTUAL_QPS" "$P50" "$P95" "$P99" "$COUNT" "$ERRORS" "$ERR_RATE" >> "$CSV"

# 9) Commit
git add "${OUTDIR}/${OUT}" "$CSV" >/dev/null 2>&1 || true
git commit -m "${PREFIX}: ${QPS} qps (label=${LABEL}) captured; p50=${P50}, p95=${P95}, p99=${P99}, qps=${ACTUAL_QPS}" >/dev/null 2>&1 || true

echo "✅ Saved -> ${OUTDIR}/${OUT} and appended -> ${CSV}"
