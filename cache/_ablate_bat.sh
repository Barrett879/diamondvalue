cd "/Users/barrettcollins/Desktop/MLB Predict"
for b in framing_bat2 platoon2_bat batvelo_bat batproc_bat; do
  python3 scripts/exp_feature_blocks.py bat $b > cache/abl_${b}.log 2>&1
done
echo "BAT ABLATIONS DONE"
