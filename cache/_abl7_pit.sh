cd "/Users/barrettcollins/Desktop/MLB Predict"
for b in airdensity_pit carryanom_pit; do python3 scripts/exp_feature_blocks.py pit $b > cache/abl_${b}.log 2>&1; done
echo "PIT R7 ABL DONE"
