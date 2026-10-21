#!/bin/bash
# Live-demo helper for the real Train-Ticket: create (and later undo) a CPU bottleneck on ONE service, so there is
# something the autoscaler should scale. On this node memory, not CPU, is the scarce resource: ts-order-service costs
# about 80 ms of CPU per request but its 1-core limit is never reached at the load the node can take. Lowering its CPU
# limit makes it saturate at a load the node CAN take, which is a standard way to stage a bottleneck.
#
#   bash scripts/21_tt_demo_bottleneck.sh create [cpu_limit]     # default 300m; a rolling restart of ONE deployment (~1-2 min)
#   bash scripts/21_tt_demo_bottleneck.sh restore                # puts the saved limit back
#   bash scripts/21_tt_demo_bottleneck.sh status
#
# This is an OPERATOR change to the demo setup, done with the same limited service account the agent uses; it is not
# something the agent does. The original limit is saved in ~/.tt_demo_state and restore refuses to run without it.
set -e
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/tt-agent-config}"
SVC="${SVC:-ts-order-service}"; STATE="$HOME/.tt_demo_state"
cur() { kubectl get deploy "$SVC" -o jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu}'; }
patch() { kubectl patch deploy "$SVC" --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/resources/limits/cpu\",\"value\":\"$1\"}]"; }
case "${1:-status}" in
  create)
    NEW="${2:-300m}"
    [ -s "$STATE" ] && { echo "already created (saved original: $(cat "$STATE")); run 'restore' first"; exit 1; }
    ORIG="$(cur)"; [ -n "$ORIG" ] || { echo "cannot read the current cpu limit of $SVC (no limit set, or the cluster is unreachable / the token expired)"; exit 1; }
    printf '%s' "$ORIG" > "$STATE"; echo "saved original cpu limit of $SVC: $ORIG"
    patch "$NEW"; kubectl rollout status deploy/"$SVC" --timeout=300s
    echo "$SVC cpu limit is now $(cur): it saturates at about $(python3 -c "print(round(float('${NEW%m}')/1000/0.082,1) if '$NEW'.endswith('m') else round(float('$NEW')/0.082,1))") req/s of its own traffic";;
  restore)
    [ -s "$STATE" ] || { rm -f "$STATE"; echo "no saved limit: nothing to restore (current limit: $(cur))"; exit 1; }
    ORIG="$(cat "$STATE")"; patch "$ORIG"; kubectl rollout status deploy/"$SVC" --timeout=300s
    rm -f "$STATE"; echo "$SVC cpu limit restored to $(cur)";;
  status) echo "$SVC cpu limit: $(cur)   saved original: $( [ -s "$STATE" ] && cat "$STATE" || echo none )";;
  *) echo "usage: $0 create [cpu] | restore | status"; exit 2;;
esac
