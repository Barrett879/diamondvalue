cd "/Users/barrettcollins/Desktop/MLB Predict"
for b in airdensity_bat carryanom_bat; do python3 scripts/exp_feature_blocks.py bat $b > cache/abl_${b}.log 2>&1; done
echo "BAT R7 ABL DONE"
