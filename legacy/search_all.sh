# Search every rung the policy cannot clear (cp009 up), saving the winning
# segments. A search often climbs several rungs from one start, so the later
# files overlap -- that is fine, the per-rung quota in the bank sorts it out.
for i in $(seq 9 36); do
  f=$(printf "demos/search_cp%03d.npz" $i)
  [ -f "$f" ] && { echo "  skip cp$i (have it)"; continue; }
  echo "=== cp$i ==="
  uv run python climb.py runs_debug/champion.pt --from-cp $i --plans 220 \
      --save "$f" 2>&1 | tr '\r' '\n' | grep -E "route-verified|transitions from"
done
echo "BATCH DONE"
