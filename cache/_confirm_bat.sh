cd "/Users/barrettcollins/Desktop/MLB Predict"
for b in platoon2_bat batvelo_bat batproc_bat; do
  python3 scripts/exp_feature_blocks.py bat $b 2025 > cache/conf_${b}.log 2>&1
done
echo "BAT CONFIRM DONE"
