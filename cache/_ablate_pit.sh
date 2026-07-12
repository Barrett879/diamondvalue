cd "/Users/barrettcollins/Desktop/MLB Predict"
for b in framing_pit2 relspin_pit mix_pit; do
  python3 scripts/exp_feature_blocks.py pit $b > cache/abl_${b}.log 2>&1
done
echo "PIT ABLATIONS DONE"
