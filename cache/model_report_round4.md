# Model report (v1, walk-forward)

Train seasons [2021, 2022, 2023], test season 2025. Every stat is evaluated at the per-game count the site displays, against three point-in-time baselines: league average (B1), the player's season-to-date average excluding the game (B2), and a Marcel-style rate times expected opportunities (B3).

## Plain-English summary

**Batting**
- `PA` [PASS]: beats season-average and Marcel baselines (model MAE 0.829 vs season-avg 0.860, Marcel 0.913).
- `H` [PASS]: beats season-average and Marcel baselines (model MAE 0.675 vs season-avg 0.684, Marcel 0.687).
- `TB` [PASS]: beats season-average and Marcel baselines (model MAE 1.261 vs season-avg 1.287, Marcel 1.293).
- `HR` [PASS]: beats season-average and Marcel baselines (model MAE 0.191 vs season-avg 0.192, Marcel 0.200).
- `BB` [PASS]: beats season-average and Marcel baselines (model MAE 0.424 vs season-avg 0.429, Marcel 0.437).
- `SO` [PASS]: beats season-average and Marcel baselines (model MAE 0.664 vs season-avg 0.678, Marcel 0.675).
- `b2` [MARGINAL]: better calibrated (deviance) but MAE near baseline (model MAE 0.263 vs season-avg 0.260, Marcel 0.266).
- `b3` [WEAK]: no better than a season average; show with caveat or drop (model MAE 0.025 vs season-avg 0.025, Marcel 0.027).
- `R` [PASS]: beats season-average and Marcel baselines (model MAE 0.540 vs season-avg 0.541, Marcel 0.552).
- `RBI` [PASS]: beats season-average and Marcel baselines (model MAE 0.579 vs season-avg 0.580, Marcel 0.591).
- `SB` [WEAK]: no better than a season average; show with caveat or drop (model MAE 0.114 vs season-avg 0.124, Marcel 0.123).

**Pitching**
- `p_outs` [PASS]: beats season-average and Marcel baselines (model MAE 2.809 vs season-avg 2.971, Marcel 3.047).
- `p_BF` [PASS]: beats season-average and Marcel baselines (model MAE 2.735 vs season-avg 2.908, Marcel 3.093).
- `p_pitches` [PASS]: beats season-average and Marcel baselines (model MAE 9.522 vs season-avg 9.995, Marcel 10.741).
- `p_K` [PASS]: beats season-average and Marcel baselines (model MAE 1.776 vs season-avg 1.917, Marcel 1.889).
- `p_BB` [PASS]: beats season-average and Marcel baselines (model MAE 1.021 vs season-avg 1.083, Marcel 1.027).
- `p_H` [PASS]: beats season-average and Marcel baselines (model MAE 1.739 vs season-avg 1.841, Marcel 1.765).
- `p_HR` [PASS]: beats season-average and Marcel baselines (model MAE 0.697 vs season-avg 0.725, Marcel 0.700).
- `p_ER` [PASS]: beats season-average and Marcel baselines (model MAE 1.588 vs season-avg 1.653, Marcel 1.599).

## Batting detail

| stat   |     n |   model_MAE |   b1_MAE |   b2_MAE |   b3_MAE |   model_dev |   b2_dev |   b3_dev |
|:-------|------:|------------:|---------:|---------:|---------:|------------:|---------:|---------:|
| PA     | 50887 |      0.8288 |   1.0979 |   0.8602 |   0.9135 |      0.5155 |   0.6668 |   0.6108 |
| H      | 50887 |      0.6753 |   0.7137 |   0.684  |   0.6866 |      1.0338 |   1.4102 |   1.0622 |
| TB     | 50887 |      1.2606 |   1.2813 |   1.2869 |   1.2933 |      2.0144 |   2.6306 |   2.0645 |
| HR     | 50887 |      0.1907 |   0.1942 |   0.1921 |   0.1996 |      0.4852 |   0.8255 |   0.4913 |
| BB     | 50887 |      0.4239 |   0.4407 |   0.4292 |   0.4371 |      0.8013 |   1.1933 |   0.8191 |
| SO     | 50887 |      0.6641 |   0.7    |   0.6782 |   0.675  |      0.9882 |   1.3275 |   1.0205 |
| b2     | 50887 |      0.2628 |   0.2583 |   0.2596 |   0.2657 |      0.5937 |   1.0062 |   0.6003 |
| b3     | 50887 |      0.0251 |   0.0245 |   0.0249 |   0.0266 |      0.1074 |   0.28   |   0.1068 |
| R      | 50887 |      0.5403 |   0.5521 |   0.5408 |   0.5524 |      0.9383 |   1.339  |   0.9532 |
| RBI    | 50887 |      0.5793 |   0.586  |   0.58   |   0.5914 |      1.1451 |   1.6753 |   1.1566 |
| SB     | 50887 |      0.114  |   0.1118 |   0.1235 |   0.1228 |      0.3603 |   0.6354 |   0.348  |

## Pitching detail

| stat      |    n |   model_MAE |   b1_MAE |   b2_MAE |   b3_MAE |   model_dev |   b2_dev |   b3_dev |
|:----------|-----:|------------:|---------:|---------:|---------:|------------:|---------:|---------:|
| p_outs    | 4860 |      2.8086 |   3.0989 |   2.9713 |   3.0466 |      1.0175 |   1.1682 |   1.2238 |
| p_BF      | 4860 |      2.7354 |   3.1753 |   2.908  |   3.0933 |      0.755  |   0.8828 |   1.016  |
| p_pitches | 4860 |      9.5223 |  11.5232 |   9.9953 |  10.7408 |      2.5354 |   2.9536 |   3.4844 |
| p_K       | 4860 |      1.7763 |   1.9791 |   1.9172 |   1.8891 |      1.1552 |   1.506  |   1.2576 |
| p_BB      | 4860 |      1.0214 |   1.0599 |   1.0827 |   1.0274 |      1.1192 |   2.3634 |   1.1254 |
| p_H       | 4860 |      1.7387 |   1.7657 |   1.8415 |   1.7648 |      1.0135 |   1.5274 |   1.0713 |
| p_HR      | 4860 |      0.6971 |   0.7088 |   0.7246 |   0.7003 |      1.1175 |   2.8564 |   1.1203 |
| p_ER      | 4860 |      1.5883 |   1.6142 |   1.653  |   1.5991 |      1.7732 |   3.774  |   1.7957 |
