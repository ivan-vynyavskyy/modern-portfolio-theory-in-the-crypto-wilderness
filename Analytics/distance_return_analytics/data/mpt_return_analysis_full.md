(erc_20_transfers_env) Apptainer> python mpt_return_analysis.py 
[12:38:01] ======================================================================
[12:38:01] MPT RETURN ANALYSIS — FULL SCALE
[12:38:01] ======================================================================

[12:38:05] Total rows: 239,620,083

[12:38:05] ======================================================================
[12:38:05] SECTION 2: Full-data Pearson Correlation — returns vs features
[12:38:05] ======================================================================
[12:38:05]   Computing Pearson on full data via DuckDB ...
[12:39:43]   Done in 98.7s
[12:39:43] Pearson r — features vs returns (full data):
               baseline  better_return  safer_risk  max_sharpe  equal_weight  mcap_weight  market
feature                                                                                          
month           -0.0001        -0.0012     -0.0012     -0.0011       -0.0015      -0.0036 -0.0121
log_value_usd   -0.0001         0.0003      0.0002      0.0003        0.0006      -0.0022 -0.0403
num_tokens      -0.0000        -0.0003     -0.0003     -0.0002        0.0001       0.0002 -0.0014

[12:39:43]   → saved ./return_analysis_results_full/pearson_returns_full_data.csv
[12:39:43] 
  2b. Pearson correlation — features vs binary win/lose (full data) ...
[12:41:12]   Done in 88.5s
[12:41:12] Pearson r — features vs P(win) where win = ret > 0 (full data):
               baseline  better_return  safer_risk  max_sharpe  equal_weight  mcap_weight
feature                                                                                  
month           -0.0868        -0.0798     -0.0900     -0.0763       -0.0965      -0.0782
log_value_usd   -0.0140        -0.0220     -0.0133     -0.0229       -0.0215      -0.0108
num_tokens      -0.0132        -0.0146     -0.0107     -0.0127       -0.0121      -0.0116

[12:41:12]   → saved ./return_analysis_results_full/pearson_win_lose_full_data.csv
[12:41:12] ======================================================================
[12:41:12] SECTION 3: Large-sample correlation (50M rows) — Pearson + Spearman
[12:41:12] ======================================================================
[12:41:12]   Drawing 50M sample ...
[12:45:19]   Sample loaded: 50,000,000 rows in 247.7s
[12:51:58] Pearson r (sample 50M):
target         baseline  better_return  equal_weight  market  max_sharpe  mcap_weight  safer_risk
feature                                                                                          
log_value_usd   -0.0003         0.0004        0.0009 -0.1004      0.0003      -0.0044      0.0003
month            0.0000        -0.0002       -0.0002 -0.3286     -0.0002      -0.0074     -0.0002
num_tokens      -0.0001        -0.0006        0.0003 -0.0178     -0.0005      -0.0012     -0.0006

[12:51:58] Spearman ρ (sample 50M):
target         baseline  better_return  equal_weight  market  max_sharpe  mcap_weight  safer_risk
feature                                                                                          
log_value_usd   -0.1063        -0.1062       -0.1081 -0.1239     -0.1191      -0.1000     -0.0995
month           -0.3383        -0.3354       -0.3725 -0.4648     -0.3357      -0.3419     -0.3462
num_tokens      -0.0213        -0.0220       -0.0114 -0.0157     -0.0342      -0.0146     -0.0006

[12:51:58]   → saved ./return_analysis_results_full/correlation_returns_sample.csv
[12:51:58] 
  3b. Binary win/lose correlation (sample 50M) ...
[12:54:45] Pearson r — features vs P(win) (sample 50M):
target         baseline  better_return  equal_weight  max_sharpe  mcap_weight  safer_risk
feature                                                                                  
log_value_usd   -0.1024        -0.0945       -0.0976     -0.1101      -0.0977     -0.0928
month           -0.2302        -0.2230       -0.2420     -0.2431      -0.2354     -0.2379
num_tokens      -0.0251        -0.0223       -0.0172     -0.0262      -0.0261     -0.0144

[12:54:45] Spearman ρ — features vs P(win) (sample 50M):
target         baseline  better_return  equal_weight  max_sharpe  mcap_weight  safer_risk
feature                                                                                  
log_value_usd   -0.1083        -0.0995       -0.1029     -0.1161      -0.1025     -0.0980
month           -0.3064        -0.2996       -0.3200     -0.3248      -0.3163     -0.3163
num_tokens      -0.0310        -0.0284       -0.0188     -0.0371      -0.0306     -0.0175

[12:54:45]   → saved ./return_analysis_results_full/correlation_win_lose_sample.csv
[12:54:45] ======================================================================
[12:54:45] SECTION 4: Return Distribution (full data)
[12:54:45] ======================================================================
[12:54:45]   4a. Descriptive statistics ...
[12:56:17]   Done in 92.3s
[12:56:17] Return descriptive statistics (full data):
                       mean           std     min     p05     p25  median     p75     p95           max  positive_pct  zero_pct  negative_pct
portfolio                                                                                                                                    
baseline       2.833163e+06  3.289321e+10 -1.0000 -0.3410 -0.1489 -0.0187  0.0725  0.4835  4.617276e+14         39.83       0.0         58.61
better_return  8.945796e+29  9.825822e+32 -1.0000 -0.3352 -0.1461 -0.0233  0.0722  0.4684  1.184642e+36         39.80       0.0         58.82
safer_risk     8.216638e+29  9.496025e+32 -1.0000 -0.3281 -0.1396 -0.0200  0.0702  0.4628  1.184642e+36         39.92       0.0         58.68
max_sharpe     1.136238e+30  1.436157e+33 -1.0000 -0.3501 -0.1521 -0.0192  0.0722  0.4696  2.132356e+36         39.89       0.0         58.76
equal_weight   1.166047e+30  1.065113e+33 -1.0000 -0.3033 -0.1384 -0.0282  0.0791  0.4687  1.184642e+36         40.53       0.0         57.91
mcap_weight    3.642900e-02  8.856858e+00 -1.0000 -0.3142 -0.1186 -0.0024  0.0478  0.4366  6.846213e+04         40.89       0.0         57.24
market         1.153000e-02  1.601800e-01 -0.4872 -0.2726 -0.0617 -0.0052  0.0752  0.2726  4.493000e-01         49.23       0.0         50.77

[12:56:17]   → saved ./return_analysis_results_full/return_descriptive_stats.csv
[12:56:17]   4b. Return bucket distribution ...
[12:56:30]   Bucket query done in 13.0s
[12:56:30] Return distribution (% of wallets per bucket):
portfolio     baseline  better_return  equal_weight  market  max_sharpe  mcap_weight  safer_risk
bucket                                                                                          
< -50%            0.94           0.78          0.28    0.00        1.11         0.53        0.59
[-50%, -20%)     16.34          15.84         13.76    7.87       16.67        13.18       14.93
[-20%, -5%)      25.21          26.38         30.23   24.37       23.87        21.95       26.17
[-5%, 0%)        16.12          15.82         13.63   18.53       17.11        21.58       16.99
[0%, 5%)         12.53          12.02         11.96   16.27       12.32        16.63       12.55
[5%, 20%)        13.35          14.01         14.42   17.27       13.56        12.14       13.85
[20%, 50%)        9.28           9.40          9.74   15.69        9.49         8.37        9.24
≥ 50%             4.67           4.37          4.41    0.00        4.52         3.74        4.28

[12:56:30] Return distribution (count per bucket):
portfolio     baseline  better_return  equal_weight    market  max_sharpe  mcap_weight  safer_risk
bucket                                                                                            
< -50%         2261382        1870649        679629         0     2666132      1259399     1424082
[-50%, -20%)  39153880       37950133      32975930  18856390    39947928     31579804    35769964
[-20%, -5%)   60399944       63206845      72436259  58392954    57192935     52594356    62708725
[-5%, 0%)     38618490       37908569      32667644  44401826    41000764     51719417    40713213
[0%, 5%)      30025163       28804450      28652613  38989597    29522743     39857479    30073922
[5%, 20%)     31984083       33581466      34548171  41378816    32498308     29098672    33176837
[20%, 50%)    22232363       22528450      23347744  37600500    22744813     20066742    22136825
≥ 50%         11192702       10460190      10560017         0    10821748      8950695    10266623

[12:56:30]   → saved ./return_analysis_results_full/return_distribution_pct.csv
[12:56:30]   → saved ./return_analysis_results_full/return_distribution_count.csv
[12:56:30] ======================================================================
[12:56:30] SECTION 5: Baseline vs Optimised Return Comparison (full data)
[12:56:30] ======================================================================
[12:56:30]   5a. Win rates — strategy beats baseline ...
[12:56:53]   Done in 22.8s
[12:56:53] Strategy vs Baseline — win/tie/loss rates + return differences:
               win_pct  tie_pct  loss_pct  mean_diff_vs_baseline  median_diff_vs_baseline
strategy                                                                                 
better_return    47.77     0.19     50.48           8.962588e+29                -0.000000
safer_risk       50.21     0.11     48.12           8.230648e+29                 0.000000
max_sharpe       48.99     0.00     49.44           1.138779e+30                -0.000012
equal_weight     49.59     0.00     48.85           1.166047e+30                 0.000015
mcap_weight      52.51     0.01     45.60          -2.842097e+06                 0.000220

[12:56:53]   → saved ./return_analysis_results_full/strategy_vs_baseline_win_rates.csv
[12:56:53] ======================================================================
[12:56:53] SECTION 6: Median returns per L1-distance bucket
[12:56:53] ======================================================================
[12:56:53] 
--- BETTER_RETURN distance buckets ---
[12:58:16]   Median returns per distance bucket (better_return):
                    n  median_baseline  median_better_return  median_safer_risk  median_max_sharpe  median_equal_weight  median_mcap_weight  median_market
dist_bucket                                                                                                                                               
0_exact       1355933        -0.000823             -0.029508          -0.029508          -0.012213            -0.022701           -0.000293      -0.005197
(0,1]        94684513        -0.015423             -0.020079          -0.019719          -0.016158            -0.027407           -0.001148      -0.011847
(1,20]       35897197        -0.008770             -0.018475          -0.013750          -0.016629            -0.025357           -0.001662      -0.005197
(20,40]      28811264        -0.014146             -0.024577          -0.016963          -0.024236            -0.028549           -0.003559       0.004513
(40,60]      25416274        -0.015730             -0.028023          -0.016113          -0.024704            -0.028618           -0.004520       0.004513
(60,80]      26730856        -0.040571             -0.035383          -0.035183          -0.035035            -0.035474           -0.019510       0.004513
(80,100]     26723982        -0.034372             -0.033182          -0.027690          -0.019648            -0.030788           -0.008389       0.014496
NaN                64         0.013820             -0.046609          -0.040195          -0.021401            -0.010026           -0.002537       0.033701

[12:58:16]   → saved ./return_analysis_results_full/returns_by_distance_bucket_better_return.csv
[12:58:16] 
--- SAFER_RISK distance buckets ---
[12:59:35]   Median returns per distance bucket (safer_risk):
                     n  median_baseline  median_better_return  median_safer_risk  median_max_sharpe  median_equal_weight  median_mcap_weight  median_market
dist_bucket                                                                                                                                                
0_exact        3166206        -0.000518             -0.010642          -0.010538          -0.014582            -0.024067           -0.000543      -0.005197
(0,1]        148645067        -0.012906             -0.017522          -0.017341          -0.015516            -0.022934           -0.001552      -0.005197
(1,20]        26935642        -0.021329             -0.027785          -0.024559          -0.025765            -0.035184           -0.003526      -0.005197
(20,40]       23343909        -0.028660             -0.033135          -0.028006          -0.030038            -0.036827           -0.005855      -0.005197
(40,60]       20608987        -0.027179             -0.033597          -0.021692          -0.027497            -0.035230           -0.004004       0.004513
(60,80]       12123101        -0.051407             -0.052483          -0.030207          -0.032543            -0.044043           -0.006333      -0.011847
(80,100]       4797162        -0.060925             -0.060679          -0.022586          -0.028329            -0.044596           -0.005290      -0.011847
NaN                  9        -0.048407             -0.099934          -0.111724          -0.003762            -0.049186           -0.127339      -0.036959

[12:59:35]   → saved ./return_analysis_results_full/returns_by_distance_bucket_safer_risk.csv
[12:59:35] 
--- MAX_SHARPE distance buckets ---
[13:00:52]   Median returns per distance bucket (max_sharpe):
                    n  median_baseline  median_better_return  median_safer_risk  median_max_sharpe  median_equal_weight  median_mcap_weight  median_market
dist_bucket                                                                                                                                               
0_exact       1924760        -0.015870             -0.013612          -0.020021          -0.020021            -0.019665           -0.001455       0.014496
(0,1]        40892964        -0.012223             -0.022685          -0.020794          -0.020884            -0.031145           -0.002693      -0.011847
(1,20]       32137999        -0.021632             -0.025435          -0.022725          -0.018816            -0.028342           -0.002568      -0.005197
(20,40]      29019292        -0.019957             -0.022921          -0.019051          -0.013929            -0.018575           -0.001820       0.027044
(40,60]      29846879        -0.024704             -0.026637          -0.022077          -0.018492            -0.024817           -0.002650       0.023311
(60,80]      59086586        -0.013546             -0.016758          -0.016184          -0.016636            -0.025020           -0.001710      -0.005197
(80,100]     46711526        -0.022178             -0.030321          -0.022044          -0.023875            -0.037164           -0.003050      -0.011847
NaN                77        -0.000122             -0.031069          -0.014641          -0.079429            -0.004768           -0.003467      -0.011847

[13:00:52]   → saved ./return_analysis_results_full/returns_by_distance_bucket_max_sharpe.csv
[13:00:52] ======================================================================
[13:00:52] SECTION 7: Regression models — returns as targets
[13:00:52] ======================================================================
[13:00:52]   Drawing 10M sample for regression ...
[13:03:51]   Regression sample: 10,000,000 rows in 178.5s
[13:03:51]   Winsorising return & alpha columns at p5/p95 ...
[13:03:51]     ret_pct_baseline: clipped to [-36.2509, 93.0989]
[13:03:51]     ret_pct_better_return: clipped to [-32.8101, 89.9698]
[13:03:52]     ret_pct_safer_risk: clipped to [-32.8101, 86.7065]
[13:03:52]     ret_pct_max_sharpe: clipped to [-34.0079, 87.4332]
[13:03:53]     ret_pct_equal_weight: clipped to [-27.4919, 88.3279]
[13:03:53]     ret_pct_mcap_weight: clipped to [-31.5396, 85.9310]
[13:03:53]     alpha_baseline: clipped to [-0.4530, 0.8657]
[13:03:54]     alpha_better_return: clipped to [-0.3998, 0.8144]
[13:03:54]     alpha_safer_risk: clipped to [-0.4085, 0.8144]
[13:03:54]     alpha_max_sharpe: clipped to [-0.4085, 0.6792]
[13:03:55]     alpha_equal_weight: clipped to [-0.3321, 0.6836]
[13:03:55]     alpha_mcap_weight: clipped to [-0.3497, 0.6213]
[13:03:55]   Winsorisation done (rows unchanged: 10,000,000)
[13:03:55] 
  --- 7a. OLS Regression ---
[13:03:57] 
  OLS: ret_baseline (%)
[13:03:57]   R² = 0.0239,  Adj-R² = 0.0239,  N = 9,902,219
[13:03:57]   Coefficients (ret_baseline %):
                    coef   std_err      t_stat  p_value significant
const          16.553556  0.021938  754.566963      0.0         ***
month          -0.317634  0.000655 -484.897766      0.0         ***
log_value_usd   0.073151  0.004021   18.191357      0.0         ***
num_tokens      0.571571  0.005584  102.350618      0.0         ***

[13:03:57]   → saved ./return_analysis_results_full/ols_return_coefs_baseline.csv
[13:03:59] 
  OLS: ret_better_return (%)
[13:03:59]   R² = 0.0229,  Adj-R² = 0.0229,  N = 9,910,620
[13:03:59]   Coefficients (ret_better_return %):
                    coef   std_err      t_stat  p_value significant
const          17.514208  0.021084  830.696291      0.0         ***
month          -0.301620  0.000633 -476.271037      0.0         ***
log_value_usd  -0.032461  0.003885   -8.355512      0.0         ***
num_tokens      0.297625  0.005281   56.359885      0.0         ***

[13:03:59]   → saved ./return_analysis_results_full/ols_return_coefs_better_return.csv
[13:04:01] 
  OLS: ret_safer_risk (%)
[13:04:01]   R² = 0.0247,  Adj-R² = 0.0247,  N = 9,910,458
[13:04:01]   Coefficients (ret_safer_risk %):
                    coef   std_err      t_stat   p_value significant
const          16.767407  0.020477  818.840745  0.000000         ***
month          -0.303911  0.000613 -495.747388  0.000000         ***
log_value_usd   0.011009  0.003764    2.925028  0.003444          **
num_tokens      0.402170  0.005169   77.799122  0.000000         ***

[13:04:01]   → saved ./return_analysis_results_full/ols_return_coefs_safer_risk.csv
[13:04:02] 
  OLS: ret_max_sharpe (%)
[13:04:03]   R² = 0.0197,  Adj-R² = 0.0197,  N = 9,912,175
[13:04:03]   Coefficients (ret_max_sharpe %):
                    coef   std_err      t_stat  p_value significant
const          15.253689  0.020218  754.448005      0.0         ***
month          -0.265584  0.000606 -437.941156      0.0         ***
log_value_usd  -0.029430  0.003722   -7.906631      0.0         ***
num_tokens      0.466693  0.005082   91.823974      0.0         ***

[13:04:03]   → saved ./return_analysis_results_full/ols_return_coefs_max_sharpe.csv
[13:04:04] 
  OLS: ret_equal_weight (%)
[13:04:04]   R² = 0.0309,  Adj-R² = 0.0309,  N = 9,902,219
[13:04:04]   Coefficients (ret_equal_weight %):
                    coef   std_err      t_stat  p_value significant
const          17.030754  0.019503  873.254534      0.0         ***
month          -0.319118  0.000582 -547.993617      0.0         ***
log_value_usd   0.075598  0.003575   21.147268      0.0         ***
num_tokens      0.724408  0.004965  145.916582      0.0         ***

[13:04:04]   → saved ./return_analysis_results_full/ols_return_coefs_equal_weight.csv
[13:04:06] 
  OLS: ret_mcap_weight (%)
[13:04:06]   R² = 0.0232,  Adj-R² = 0.0232,  N = 9,894,918
[13:04:06]   Coefficients (ret_mcap_weight %):
                    coef   std_err      t_stat  p_value significant
const          15.333034  0.019379  791.214563      0.0         ***
month          -0.277478  0.000578 -479.946939      0.0         ***
log_value_usd   0.083901  0.003545   23.668799      0.0         ***
num_tokens      0.421881  0.004961   85.038687      0.0         ***

[13:04:06]   → saved ./return_analysis_results_full/ols_return_coefs_mcap_weight.csv
[13:04:06] 
  OLS Summary (returns):
                   R2  Adj_R2     F_stat  F_pvalue
target                                            
baseline       0.0239  0.0239   80720.33       0.0
better_return  0.0229  0.0229   77452.44       0.0
safer_risk     0.0247  0.0247   83757.48       0.0
max_sharpe     0.0197  0.0197   66384.51       0.0
equal_weight   0.0309  0.0309  105072.05       0.0
mcap_weight    0.0232  0.0232   78332.46       0.0

[13:04:06]   → saved ./return_analysis_results_full/ols_return_summary.csv
[13:04:06] 
  --- 7b. Quantile Regression ---
[13:04:07]   Quantile regression on 2M rows
[13:04:07]   QuantReg: ret_baseline, q=0.25 ...
[13:04:20]   QuantReg coefficients (ret_baseline, q=0.25):
                   coef   std_err      t_stat  p_value  quantile    target significant
Intercept     -7.737339  0.028498 -271.509275      0.0      0.25  baseline         ***
month         -0.146357  0.000825 -177.343141      0.0      0.25  baseline         ***
log_value_usd  0.540797  0.004882  110.769885      0.0      0.25  baseline         ***
num_tokens     0.064542  0.007350    8.781029      0.0      0.25  baseline         ***

[13:04:20]   QuantReg: ret_baseline, q=0.50 ...
[13:04:31]   QuantReg coefficients (ret_baseline, q=0.5):
                    coef   std_err      t_stat  p_value  quantile    target significant
Intercept      11.646688  0.051260  227.208222      0.0       0.5  baseline         ***
month          -0.212841  0.001524 -139.651028      0.0       0.5  baseline         ***
log_value_usd  -0.899507  0.009362  -96.077346      0.0       0.5  baseline         ***
num_tokens      0.676009  0.013183   51.279715      0.0       0.5  baseline         ***

[13:04:31]   QuantReg: ret_baseline, q=0.75 ...
[13:05:02]   QuantReg coefficients (ret_baseline, q=0.75):
                    coef   std_err      t_stat   p_value  quantile    target significant
Intercept      35.171403  0.057538  611.276955  0.000000      0.75  baseline         ***
month          -0.530652  0.001751 -303.037502  0.000000      0.75  baseline         ***
log_value_usd   0.049461  0.010898    4.538401  0.000006      0.75  baseline         ***
num_tokens      1.067878  0.014040   76.059085  0.000000      0.75  baseline         ***

[13:05:02]   QuantReg: ret_better_return, q=0.25 ...
[13:05:18]   QuantReg coefficients (ret_better_return, q=0.25):
                   coef   std_err      t_stat  p_value  quantile         target significant
Intercept     -9.421595  0.022518 -418.404474      0.0      0.25  better_return         ***
month         -0.079851  0.000666 -119.952959      0.0      0.25  better_return         ***
log_value_usd  0.153047  0.003795   40.326863      0.0      0.25  better_return         ***
num_tokens     0.131134  0.005909   22.191228      0.0      0.25  better_return         ***

[13:05:18]   QuantReg: ret_better_return, q=0.50 ...
[13:05:30]   QuantReg coefficients (ret_better_return, q=0.5):
                   coef   std_err      t_stat  p_value  quantile         target significant
Intercept      8.793579  0.020017  439.301026      0.0       0.5  better_return         ***
month         -0.184629  0.000598 -308.967039      0.0       0.5  better_return         ***
log_value_usd -0.596883  0.003670 -162.619116      0.0       0.5  better_return         ***
num_tokens     0.724710  0.005095  142.237805      0.0       0.5  better_return         ***

[13:05:30]   QuantReg: ret_better_return, q=0.75 ...
[13:05:49]   QuantReg coefficients (ret_better_return, q=0.75):
                    coef   std_err      t_stat  p_value  quantile         target significant
Intercept      35.785314  0.072365  494.508293      0.0      0.75  better_return         ***
month          -0.523283  0.002198 -238.045455      0.0      0.75  better_return         ***
log_value_usd   0.129598  0.013642    9.499956      0.0      0.75  better_return         ***
num_tokens      0.585434  0.017557   33.345612      0.0      0.75  better_return         ***

[13:05:49]   QuantReg: ret_safer_risk, q=0.25 ...
[13:06:02]   QuantReg coefficients (ret_safer_risk, q=0.25):
                   coef   std_err      t_stat  p_value  quantile      target significant
Intercept     -9.924385  0.061183 -162.207213      0.0      0.25  safer_risk         ***
month         -0.116947  0.001693  -69.069340      0.0      0.25  safer_risk         ***
log_value_usd  0.578502  0.010008   57.805565      0.0      0.25  safer_risk         ***
num_tokens     0.429495  0.016862   25.471805      0.0      0.25  safer_risk         ***

[13:06:02]   QuantReg: ret_safer_risk, q=0.50 ...
[13:06:13]   QuantReg coefficients (ret_safer_risk, q=0.5):
                    coef   std_err      t_stat  p_value  quantile      target significant
Intercept      11.960269  0.043992  271.876492      0.0       0.5  safer_risk         ***
month          -0.222330  0.001314 -169.232119      0.0       0.5  safer_risk         ***
log_value_usd  -0.778278  0.008067  -96.475428      0.0       0.5  safer_risk         ***
num_tokens      0.610502  0.011176   54.628036      0.0       0.5  safer_risk         ***

[13:06:13]   QuantReg: ret_safer_risk, q=0.75 ...
[13:06:30]   QuantReg coefficients (ret_safer_risk, q=0.75):
                    coef   std_err      t_stat   p_value  quantile      target significant
Intercept      35.664552  0.075071  475.078748  0.000000      0.75  safer_risk         ***
month          -0.507829  0.002278 -222.946160  0.000000      0.75  safer_risk         ***
log_value_usd  -0.023963  0.014216   -1.685632  0.091867      0.75  safer_risk            
num_tokens      0.415762  0.018115   22.950691  0.000000      0.75  safer_risk         ***

[13:06:30]   QuantReg: ret_max_sharpe, q=0.25 ...
[13:06:41]   QuantReg coefficients (ret_max_sharpe, q=0.25):
                   coef   std_err      t_stat   p_value  quantile      target significant
Intercept     -7.009648  0.062044 -112.979563  0.000000      0.25  max_sharpe         ***
month         -0.089973  0.001903  -47.287443  0.000000      0.25  max_sharpe         ***
log_value_usd -0.226233  0.011010  -20.547874  0.000000      0.25  max_sharpe         ***
num_tokens    -0.042487  0.015144   -2.805476  0.005024      0.25  max_sharpe          **

[13:06:41]   QuantReg: ret_max_sharpe, q=0.50 ...
[13:06:53]   QuantReg coefficients (ret_max_sharpe, q=0.5):
                   coef   std_err      t_stat  p_value  quantile      target significant
Intercept      8.972435  0.020553  436.553002      0.0       0.5  max_sharpe         ***
month         -0.186990  0.000615 -304.069937      0.0       0.5  max_sharpe         ***
log_value_usd -0.554806  0.003776 -146.943026      0.0       0.5  max_sharpe         ***
num_tokens     0.741980  0.005200  142.679967      0.0       0.5  max_sharpe         ***

[13:06:53]   QuantReg: ret_max_sharpe, q=0.75 ...
[13:07:06]   QuantReg coefficients (ret_max_sharpe, q=0.75):
                    coef   std_err       t_stat  p_value  quantile      target significant
Intercept      37.214032  0.031267  1190.212168      0.0      0.75  max_sharpe         ***
month          -0.529403  0.000935  -566.070587      0.0      0.75  max_sharpe         ***
log_value_usd  -0.102392  0.005954   -17.198262      0.0      0.75  max_sharpe         ***
num_tokens      0.710102  0.007512    94.532565      0.0      0.75  max_sharpe         ***

[13:07:06]   QuantReg: ret_equal_weight, q=0.25 ...
[13:07:23]   QuantReg coefficients (ret_equal_weight, q=0.25):
                   coef   std_err      t_stat  p_value  quantile        target significant
Intercept     -4.918885  0.057366  -85.745937      0.0      0.25  equal_weight         ***
month         -0.166184  0.001597 -104.030415      0.0      0.25  equal_weight         ***
log_value_usd -0.122376  0.009718  -12.593344      0.0      0.25  equal_weight         ***
num_tokens     0.496311  0.015841   31.330956      0.0      0.25  equal_weight         ***

[13:07:23]   QuantReg: ret_equal_weight, q=0.50 ...
[13:07:37]   QuantReg coefficients (ret_equal_weight, q=0.5):
                   coef   std_err      t_stat  p_value  quantile        target significant
Intercept      8.569706  0.049200  174.179910      0.0       0.5  equal_weight         ***
month         -0.224687  0.001463 -153.594738      0.0       0.5  equal_weight         ***
log_value_usd -0.372779  0.008986  -41.483742      0.0       0.5  equal_weight         ***
num_tokens     1.181294  0.012653   93.360197      0.0       0.5  equal_weight         ***

[13:07:37]   QuantReg: ret_equal_weight, q=0.75 ...
[13:07:51]   QuantReg coefficients (ret_equal_weight, q=0.75):
                    coef   std_err      t_stat  p_value  quantile        target significant
Intercept      29.122361  0.050340  578.510201      0.0      0.75  equal_weight         ***
month          -0.487366  0.001577 -309.076713      0.0      0.75  equal_weight         ***
log_value_usd   0.874263  0.009688   90.240114      0.0      0.75  equal_weight         ***
num_tokens      1.407934  0.012384  113.687884      0.0      0.75  equal_weight         ***

[13:07:51]   QuantReg: ret_mcap_weight, q=0.25 ...
[13:08:07]   QuantReg coefficients (ret_mcap_weight, q=0.25):
                   coef   std_err     t_stat  p_value  quantile       target significant
Intercept     -6.367765  0.067289 -94.633549      0.0      0.25  mcap_weight         ***
month         -0.124827  0.001918 -65.090425      0.0      0.25  mcap_weight         ***
log_value_usd  0.512309  0.011489  44.591946      0.0      0.25  mcap_weight         ***
num_tokens     0.190194  0.018291  10.398236      0.0      0.25  mcap_weight         ***

[13:08:07]   QuantReg: ret_mcap_weight, q=0.50 ...
[13:08:25]   QuantReg coefficients (ret_mcap_weight, q=0.5):
                   coef   std_err      t_stat  p_value  quantile       target significant
Intercept      6.838220  0.040738  167.858857      0.0       0.5  mcap_weight         ***
month         -0.121969  0.001211 -100.736622      0.0       0.5  mcap_weight         ***
log_value_usd -0.560638  0.007428  -75.479067      0.0       0.5  mcap_weight         ***
num_tokens     0.584765  0.010523   55.572063      0.0       0.5  mcap_weight         ***

[13:08:25]   QuantReg: ret_mcap_weight, q=0.75 ...
[13:08:38]   QuantReg coefficients (ret_mcap_weight, q=0.75):
                    coef   std_err      t_stat  p_value  quantile       target significant
Intercept      32.142247  0.063313  507.674482      0.0      0.75  mcap_weight         ***
month          -0.473768  0.001925 -246.167591      0.0      0.75  mcap_weight         ***
log_value_usd  -0.065834  0.011874   -5.544238      0.0      0.75  mcap_weight         ***
num_tokens      0.596226  0.015621   38.167607      0.0      0.75  mcap_weight         ***

[13:08:38]   → saved ./return_analysis_results_full/quantile_regression_returns_all.csv
[13:08:38] 
  --- 7c. Random Forest (feature importance) ---
[13:08:40]   Random Forest on 2M rows, 32 cores
[13:08:40] 
  Training RF for ret_baseline ...
[13:09:04]   RF ret_baseline: R²=0.6424, MAE=12.85  (24.3s)
[13:09:04]   Feature importances (ret_baseline):
         feature  importance    target  R2_test  MAE_test
0          month      0.7272  baseline   0.6424     12.85
1  log_value_usd      0.2592  baseline   0.6424     12.85
2     num_tokens      0.0136  baseline   0.6424     12.85

[13:09:04] 
  Training RF for ret_better_return ...
[13:09:28]   RF ret_better_return: R²=0.6056, MAE=14.00  (23.9s)
[13:09:28]   Feature importances (ret_better_return):
         feature  importance         target  R2_test  MAE_test
0          month      0.7991  better_return   0.6056      14.0
1  log_value_usd      0.1868  better_return   0.6056      14.0
2     num_tokens      0.0141  better_return   0.6056      14.0

[13:09:28] 
  Training RF for ret_safer_risk ...
[13:09:52]   RF ret_safer_risk: R²=0.6521, MAE=12.04  (24.3s)
[13:09:52]   Feature importances (ret_safer_risk):
         feature  importance      target  R2_test  MAE_test
0          month      0.7502  safer_risk   0.6521     12.04
1  log_value_usd      0.2392  safer_risk   0.6521     12.04
2     num_tokens      0.0107  safer_risk   0.6521     12.04

[13:09:52] 
  Training RF for ret_max_sharpe ...
[13:10:15]   RF ret_max_sharpe: R²=0.5952, MAE=13.36  (23.0s)
[13:10:15]   Feature importances (ret_max_sharpe):
         feature  importance      target  R2_test  MAE_test
0          month      0.8253  max_sharpe   0.5952     13.36
1  log_value_usd      0.1613  max_sharpe   0.5952     13.36
2     num_tokens      0.0134  max_sharpe   0.5952     13.36

[13:10:15] 
  Training RF for ret_equal_weight ...
[13:10:40]   RF ret_equal_weight: R²=0.6369, MAE=11.80  (24.5s)
[13:10:40]   Feature importances (ret_equal_weight):
         feature  importance        target  R2_test  MAE_test
0          month      0.8392  equal_weight   0.6369      11.8
1  log_value_usd      0.1431  equal_weight   0.6369      11.8
2     num_tokens      0.0178  equal_weight   0.6369      11.8

[13:10:40] 
  Training RF for ret_mcap_weight ...
[13:11:03]   RF ret_mcap_weight: R²=0.6018, MAE=12.70  (23.1s)
[13:11:03]   Feature importances (ret_mcap_weight):
         feature  importance       target  R2_test  MAE_test
0          month      0.7968  mcap_weight   0.6018      12.7
1  log_value_usd      0.1891  mcap_weight   0.6018      12.7
2     num_tokens      0.0141  mcap_weight   0.6018      12.7

[13:11:03]   → saved ./return_analysis_results_full/random_forest_return_importance.csv
[13:11:03] ======================================================================
[13:11:03] SECTION 8: ETH/BTC Holder Detection & Regressions
[13:11:03] ======================================================================
[13:11:03]   8a. Prevalence of ETH/BTC wrappers in baseline holdings ...
[13:16:44]   Done in 340.6s
[13:16:44]   holds_eth: 18,607,545 (7.77%)
[13:16:44]   holds_btc: 3,544,349 (1.48%)
[13:16:44]   holds_both: 777,651 (0.32%)
[13:16:44]   holds_neither: 218,245,840 (91.08%)
[13:16:44]   → saved ./return_analysis_results_full/eth_btc_prevalence.csv
[13:16:44] 
  8b. Median returns by ETH/BTC holder status ...
[13:19:30]   Done in 166.3s
[13:19:30]   Median returns by (holds_eth, holds_btc):
   holds_eth  holds_btc          n  median_baseline  median_better_return  median_safer_risk  median_max_sharpe  median_equal_weight  median_mcap_weight  median_market
0          0          0  218245840        -0.018398             -0.023053          -0.020047          -0.019081            -0.027947           -0.002163      -0.005197
1          0          1    2766698        -0.002603             -0.005385          -0.003568          -0.004991            -0.009770           -0.000450      -0.005197
2          1          0   17829894        -0.028576             -0.029357          -0.024869          -0.025654            -0.035723           -0.005562      -0.011847
3          1          1     777651        -0.007344             -0.009380          -0.006483          -0.002797            -0.017223           -0.002857      -0.011847

[13:19:30]   → saved ./return_analysis_results_full/returns_by_eth_btc_holder.csv
[13:19:30] 
  8c. Drawing regression sample with ETH/BTC flags ...
[13:25:48]   Winsorising ETH/BTC sample at p5/p95 ...
[13:25:54]   Sample: 10,000,000 rows in 384.4s
[13:25:54]   holds_eth=1: 362,057  holds_btc=1: 69,474
[13:25:55] 
  8d. OLS with ETH/BTC holder dummies ...
[13:25:58] 
  OLS (ETH/BTC): ret_baseline (%)
[13:25:58]   R² = 0.0241,  Adj-R² = 0.0241,  N = 9,901,975
[13:25:58]   Coefficients (ret_baseline %, with ETH/BTC):
                    coef   std_err      t_stat   p_value significant
const          16.571015  0.022126  748.945453  0.000000         ***
month          -0.319439  0.000660 -483.953365  0.000000         ***
log_value_usd   0.065858  0.004029   16.346122  0.000000         ***
num_tokens      0.576220  0.005746  100.289088  0.000000         ***
holds_eth       0.022899  0.058574    0.390939  0.695842            
holds_btc       3.060218  0.130701   23.413965  0.000000         ***

[13:25:58]   → saved ./return_analysis_results_full/ols_return_ethbtc_coefs_baseline.csv
[13:26:00] 
  OLS (ETH/BTC): ret_better_return (%)
[13:26:00]   R² = 0.0232,  Adj-R² = 0.0232,  N = 9,910,397
[13:26:00]   Coefficients (ret_better_return %, with ETH/BTC):
                    coef   std_err      t_stat  p_value significant
const          17.573882  0.021280  825.838915      0.0         ***
month          -0.304122  0.000638 -476.670985      0.0         ***
log_value_usd  -0.037322  0.003893   -9.587349      0.0         ***
num_tokens      0.277230  0.005456   50.816477      0.0         ***
holds_eth       0.514576  0.056618    9.088606      0.0         ***
holds_btc       4.354632  0.126375   34.458063      0.0         ***

[13:26:00]   → saved ./return_analysis_results_full/ols_return_ethbtc_coefs_better_return.csv
[13:26:02] 
  OLS (ETH/BTC): ret_safer_risk (%)
[13:26:03]   R² = 0.0249,  Adj-R² = 0.0249,  N = 9,910,182
[13:26:03]   Coefficients (ret_safer_risk %, with ETH/BTC):
                    coef   std_err      t_stat   p_value significant
const          16.772090  0.020673  811.314357  0.000000         ***
month          -0.305073  0.000618 -493.890284  0.000000         ***
log_value_usd   0.006069  0.003772    1.609173  0.107578            
num_tokens      0.405787  0.005339   75.997964  0.000000         ***
holds_eth       0.035360  0.054842    0.644768  0.519077            
holds_btc       2.388874  0.122381   19.520012  0.000000         ***

[13:26:03]   → saved ./return_analysis_results_full/ols_return_ethbtc_coefs_safer_risk.csv
[13:26:05] 
  OLS (ETH/BTC): ret_max_sharpe (%)
[13:26:05]   R² = 0.0202,  Adj-R² = 0.0202,  N = 9,912,004
[13:26:05]   Coefficients (ret_max_sharpe %, with ETH/BTC):
                    coef   std_err      t_stat  p_value significant
const          15.333305  0.020407  751.381613      0.0         ***
month          -0.270803  0.000611 -443.339112      0.0         ***
log_value_usd  -0.040291  0.003729  -10.803858      0.0         ***
num_tokens      0.433287  0.005252   82.494649      0.0         ***
holds_eth       2.727783  0.054222   50.307751      0.0         ***
holds_btc       5.069387  0.120995   41.897604      0.0         ***

[13:26:05]   → saved ./return_analysis_results_full/ols_return_ethbtc_coefs_max_sharpe.csv
[13:26:07] 
  OLS (ETH/BTC): ret_equal_weight (%)
[13:26:07]   R² = 0.0310,  Adj-R² = 0.0310,  N = 9,901,975
[13:26:07]   Coefficients (ret_equal_weight %, with ETH/BTC):
                    coef   std_err      t_stat  p_value significant
const          17.013534  0.019671  864.893009      0.0         ***
month          -0.319644  0.000587 -544.689464      0.0         ***
log_value_usd   0.068676  0.003582   19.172670      0.0         ***
num_tokens      0.740658  0.005108  144.994115      0.0         ***
holds_eth      -0.320617  0.052076   -6.156679      0.0         ***
holds_btc       2.248730  0.116201   19.352036      0.0         ***

[13:26:07]   → saved ./return_analysis_results_full/ols_return_ethbtc_coefs_equal_weight.csv
[13:26:10] 
  OLS (ETH/BTC): ret_mcap_weight (%)
[13:26:10]   R² = 0.0233,  Adj-R² = 0.0233,  N = 9,894,594
[13:26:10]   Coefficients (ret_mcap_weight %, with ETH/BTC):
                    coef   std_err      t_stat  p_value significant
const          15.356249  0.019551  785.439137      0.0         ***
month          -0.279877  0.000583 -480.237935      0.0         ***
log_value_usd   0.079385  0.003553   22.345658      0.0         ***
num_tokens      0.412476  0.005104   80.813540      0.0         ***
holds_eth       1.155412  0.051635   22.376432      0.0         ***
holds_btc       2.439342  0.115203   21.174333      0.0         ***

[13:26:10]   → saved ./return_analysis_results_full/ols_return_ethbtc_coefs_mcap_weight.csv
[13:26:10] 
  OLS Summary (returns, with ETH/BTC dummies):
                   R2  Adj_R2    F_stat  F_pvalue
target                                           
baseline       0.0241  0.0241  48873.18       0.0
better_return  0.0232  0.0232  46994.93       0.0
safer_risk     0.0249  0.0249  50511.99       0.0
max_sharpe     0.0202  0.0202  40959.71       0.0
equal_weight   0.0310  0.0310  63303.10       0.0
mcap_weight    0.0233  0.0233  47274.12       0.0

[13:26:10]   → saved ./return_analysis_results_full/ols_return_ethbtc_summary.csv
[13:26:10] 
  8e. OLS with ETH/BTC dummies → alpha (risk-adjusted return) ...
[13:26:12] 
  OLS (ETH/BTC): alpha_baseline (%)
[13:26:12]   R² = 0.0094,  Adj-R² = 0.0094,  N = 9,901,975
[13:26:12]   Coefficients (alpha_baseline %, with ETH/BTC):
                   coef   std_err      t_stat  p_value significant
const          7.102483  0.020297  349.931152      0.0         ***
month         -0.178786  0.000605 -295.270851      0.0         ***
log_value_usd -0.123308  0.003696  -33.363447      0.0         ***
num_tokens     0.285143  0.005271   54.100384      0.0         ***
holds_eth      0.271917  0.053732    5.060595      0.0         ***
holds_btc      1.104612  0.119896    9.213056      0.0         ***

[13:26:12]   → saved ./return_analysis_results_full/ols_alpha_ethbtc_coefs_baseline.csv
[13:26:15] 
  OLS (ETH/BTC): alpha_better_return (%)
[13:26:15]   R² = 0.0086,  Adj-R² = 0.0086,  N = 9,910,397
[13:26:15]   Coefficients (alpha_better_return %, with ETH/BTC):
                   coef   std_err      t_stat  p_value significant
const          7.294663  0.019267  378.618194      0.0         ***
month         -0.163840  0.000578 -283.635148      0.0         ***
log_value_usd -0.102754  0.003524  -29.154673      0.0         ***
num_tokens     0.086404  0.004939   17.493056      0.0         ***
holds_eth      0.418826  0.051261    8.170538      0.0         ***
holds_btc      1.929854  0.114417   16.866792      0.0         ***

[13:26:15]   → saved ./return_analysis_results_full/ols_alpha_ethbtc_coefs_better_return.csv
[13:26:17] 
  OLS (ETH/BTC): alpha_safer_risk (%)
[13:26:17]   R² = 0.0102,  Adj-R² = 0.0102,  N = 9,910,182
[13:26:17]   Coefficients (alpha_safer_risk %, with ETH/BTC):
                   coef   std_err      t_stat   p_value significant
const          7.122719  0.018932  376.228960  0.000000         ***
month         -0.173117  0.000566 -306.034962  0.000000         ***
log_value_usd -0.133396  0.003454  -38.619701  0.000000         ***
num_tokens     0.320521  0.004890   65.548851  0.000000         ***
holds_eth      0.095178  0.050224    1.895083  0.058081            
holds_btc      0.610084  0.112075    5.443537  0.000000         ***

[13:26:17]   → saved ./return_analysis_results_full/ols_alpha_ethbtc_coefs_safer_risk.csv
[13:26:20] 
  OLS (ETH/BTC): alpha_max_sharpe (%)
[13:26:20]   R² = 0.0072,  Adj-R² = 0.0072,  N = 9,912,004
[13:26:20]   Coefficients (alpha_max_sharpe %, with ETH/BTC):
                   coef   std_err      t_stat  p_value significant
const          4.265863  0.017235  247.509956      0.0         ***
month         -0.130596  0.000516 -253.147945      0.0         ***
log_value_usd -0.135410  0.003150  -42.991591      0.0         ***
num_tokens     0.315621  0.004436   71.150457      0.0         ***
holds_eth      1.441661  0.045795   31.481057      0.0         ***
holds_btc      2.845195  0.102189   27.842410      0.0         ***

[13:26:20]   → saved ./return_analysis_results_full/ols_alpha_ethbtc_coefs_max_sharpe.csv
[13:26:22] 
  OLS (ETH/BTC): alpha_equal_weight (%)
[13:26:23]   R² = 0.0131,  Adj-R² = 0.0131,  N = 9,901,975
[13:26:23]   Coefficients (alpha_equal_weight %, with ETH/BTC):
                   coef   std_err      t_stat   p_value significant
const          6.526531  0.016732  390.055196  0.000000         ***
month         -0.172520  0.000499 -345.618350  0.000000         ***
log_value_usd -0.039144  0.003047  -12.847256  0.000000         ***
num_tokens     0.487052  0.004345  112.094226  0.000000         ***
holds_eth     -0.015906  0.044296   -0.359094  0.719524            
holds_btc      0.451006  0.098840    4.562969  0.000005         ***

[13:26:23]   → saved ./return_analysis_results_full/ols_alpha_ethbtc_coefs_equal_weight.csv
[13:26:25] 
  OLS (ETH/BTC): alpha_mcap_weight (%)
[13:26:25]   R² = 0.0076,  Adj-R² = 0.0076,  N = 9,894,594
[13:26:25]   Coefficients (alpha_mcap_weight %, with ETH/BTC):
                   coef   std_err      t_stat   p_value significant
const          4.889458  0.016015  305.304588  0.000000         ***
month         -0.129134  0.000477 -270.504748  0.000000         ***
log_value_usd -0.002751  0.002910   -0.945475  0.344416            
num_tokens     0.161170  0.004181   38.549019  0.000000         ***
holds_eth      0.236760  0.042296    5.597683  0.000000         ***
holds_btc      1.193527  0.094366   12.647782  0.000000         ***

[13:26:25]   → saved ./return_analysis_results_full/ols_alpha_ethbtc_coefs_mcap_weight.csv
[13:26:25] 
  OLS Summary (alpha, with ETH/BTC dummies):
                   R2  Adj_R2    F_stat  F_pvalue
target                                           
baseline       0.0094  0.0094  18769.58       0.0
better_return  0.0086  0.0086  17162.79       0.0
safer_risk     0.0102  0.0102  20422.28       0.0
max_sharpe     0.0072  0.0072  14451.90       0.0
equal_weight   0.0131  0.0131  26320.54       0.0
mcap_weight    0.0076  0.0076  15107.90       0.0

[13:26:25]   → saved ./return_analysis_results_full/ols_alpha_ethbtc_summary.csv
[13:26:25] 
  8f. Random Forest with ETH/BTC features ...
[13:26:26]   RF on 2M rows with features: ['month', 'log_value_usd', 'num_tokens', 'holds_eth', 'holds_btc']
[13:26:26] 
  Training RF for ret_baseline (with ETH/BTC) ...
[13:26:53]   RF ret_baseline (ETH/BTC): R²=0.6456, MAE=12.77  (26.4s)
[13:26:53]   Feature importances (ret_baseline, with ETH/BTC):
         feature  importance    target  R2_test  MAE_test
0          month      0.7258  baseline   0.6456     12.77
1  log_value_usd      0.2588  baseline   0.6456     12.77
2     num_tokens      0.0134  baseline   0.6456     12.77
3      holds_eth      0.0017  baseline   0.6456     12.77
4      holds_btc      0.0002  baseline   0.6456     12.77

[13:26:53] 
  Training RF for ret_better_return (with ETH/BTC) ...
[13:27:19]   RF ret_better_return (ETH/BTC): R²=0.6041, MAE=13.95  (26.1s)
[13:27:19]   Feature importances (ret_better_return, with ETH/BTC):
         feature  importance         target  R2_test  MAE_test
0          month      0.7961  better_return   0.6041     13.95
1  log_value_usd      0.1873  better_return   0.6041     13.95
2     num_tokens      0.0138  better_return   0.6041     13.95
3      holds_eth      0.0026  better_return   0.6041     13.95
4      holds_btc      0.0003  better_return   0.6041     13.95

[13:27:19] 
  Training RF for ret_safer_risk (with ETH/BTC) ...
[13:27:48]   RF ret_safer_risk (ETH/BTC): R²=0.6516, MAE=12.03  (28.9s)
[13:27:48]   Feature importances (ret_safer_risk, with ETH/BTC):
         feature  importance      target  R2_test  MAE_test
0          month      0.7477  safer_risk   0.6516     12.03
1  log_value_usd      0.2386  safer_risk   0.6516     12.03
2     num_tokens      0.0113  safer_risk   0.6516     12.03
3      holds_eth      0.0019  safer_risk   0.6516     12.03
4      holds_btc      0.0004  safer_risk   0.6516     12.03

[13:27:48] 
  Training RF for ret_max_sharpe (with ETH/BTC) ...
[13:28:14]   RF ret_max_sharpe (ETH/BTC): R²=0.6006, MAE=13.22  (25.9s)
[13:28:14]   Feature importances (ret_max_sharpe, with ETH/BTC):
         feature  importance      target  R2_test  MAE_test
0          month      0.8182  max_sharpe   0.6006     13.22
1  log_value_usd      0.1601  max_sharpe   0.6006     13.22
2     num_tokens      0.0130  max_sharpe   0.6006     13.22
3      holds_eth      0.0076  max_sharpe   0.6006     13.22
4      holds_btc      0.0012  max_sharpe   0.6006     13.22

[13:28:14] 
  Training RF for ret_equal_weight (with ETH/BTC) ...
[13:28:39]   RF ret_equal_weight (ETH/BTC): R²=0.6379, MAE=11.78  (25.9s)
[13:28:39]   Feature importances (ret_equal_weight, with ETH/BTC):
         feature  importance        target  R2_test  MAE_test
0          month      0.8370  equal_weight   0.6379     11.78
1  log_value_usd      0.1420  equal_weight   0.6379     11.78
2     num_tokens      0.0180  equal_weight   0.6379     11.78
3      holds_eth      0.0027  equal_weight   0.6379     11.78
4      holds_btc      0.0002  equal_weight   0.6379     11.78

[13:28:39] 
  Training RF for ret_mcap_weight (with ETH/BTC) ...
[13:29:04]   RF ret_mcap_weight (ETH/BTC): R²=0.6066, MAE=12.55  (24.6s)
[13:29:04]   Feature importances (ret_mcap_weight, with ETH/BTC):
         feature  importance       target  R2_test  MAE_test
0          month      0.7912  mcap_weight   0.6066     12.55
1  log_value_usd      0.1867  mcap_weight   0.6066     12.55
2     num_tokens      0.0141  mcap_weight   0.6066     12.55
3      holds_eth      0.0078  mcap_weight   0.6066     12.55
4      holds_btc      0.0001  mcap_weight   0.6066     12.55

[13:29:04]   → saved ./return_analysis_results_full/random_forest_return_ethbtc_importance.csv
[13:29:04] ======================================================================
[13:29:04] SECTION 9: Alpha Analysis (full data)
[13:29:04] ======================================================================
[13:29:04]   9a. Alpha descriptive statistics ...
[13:30:05]   Done in 61.3s
[13:30:05] Alpha descriptive statistics (full data):
                 mean_alpha     std_alpha  median_alpha  p05_alpha  p95_alpha  positive_alpha_pct
portfolio                                                                                        
baseline       2.833163e+06  3.289321e+10     -0.019143    -0.2993     0.3916               38.30
better_return  8.945796e+29  9.825822e+32     -0.023187    -0.2903     0.3964               37.24
safer_risk     8.216638e+29  9.496025e+32     -0.018488    -0.2778     0.3763               37.91
max_sharpe     1.136238e+30  1.436157e+33     -0.019217    -0.3126     0.3820               37.47
equal_weight   1.166047e+30  1.065113e+33     -0.027755    -0.2529     0.3919               37.33
mcap_weight    3.040000e-02  8.854427e+00     -0.002986    -0.2669     0.3319               39.09

[13:30:05]   → saved ./return_analysis_results_full/alpha_descriptive_stats.csv
[13:30:05] 
  9b. Alpha improvement: optimised − baseline alpha ...
[13:30:36] Alpha improvement over baseline:
               mean_alpha_improvement  median_alpha_improvement  alpha_win_pct
strategy                                                                      
better_return            8.962588e+29                 -0.000000          47.85
safer_risk               8.230648e+29                  0.000000          50.25
max_sharpe               1.138779e+30                 -0.000091          48.35
equal_weight             1.166047e+30                  0.000001          49.37
mcap_weight             -2.842097e+06                  0.000837          52.65

[13:30:36]   → saved ./return_analysis_results_full/alpha_improvement_vs_baseline.csv
[13:30:36] ======================================================================
[13:30:36] SECTION 10: Additional Analytics
[13:30:36] ======================================================================
[13:30:36] 
  10a. Cross-return correlation matrix ...
[13:30:45]   Done in 8.8s
[13:30:45] Cross-return Pearson correlation matrix:
               baseline  better_return  safer_risk  max_sharpe  equal_weight  mcap_weight
baseline         1.0000         0.0002      0.0002      0.0001        0.0002       0.0004
better_return    0.0002         1.0000      0.9756      0.8637        0.9521       0.0000
safer_risk       0.0002         0.9756      1.0000      0.8213        0.9269       0.0000
max_sharpe       0.0001         0.8637      0.8213      1.0000        0.8189       0.0000
equal_weight     0.0002         0.9521      0.9269      0.8189        1.0000       0.0000
mcap_weight      0.0004         0.0000      0.0000      0.0000        0.0000       1.0000

[13:30:45]   → saved ./return_analysis_results_full/cross_return_correlation.csv
[13:30:45] 
  10b. Mean returns by num_tokens bucket ...
[13:32:04] Mean returns by num_tokens bucket:
  token_bucket          n  mean_baseline  mean_better_return  mean_safer_risk  mean_max_sharpe  mean_equal_weight  mean_mcap_weight  mean_market
0          1-2  144150124   4.676195e+06        1.235203e+30     1.235202e+30     1.528611e+30       1.235202e+30          0.038297     0.011529
1          3-5   73707651   1.156805e+00        4.688841e+29     2.192876e+29     6.673459e+29       9.777834e+29          0.032173     0.011613
2         6-10   16234960   1.936800e-02        5.050680e+28     9.101685e+28     9.962047e+28       1.326050e+30          0.036725     0.011775
3        11-20    4422902   2.759600e-02        1.967414e+28     5.836630e+28     3.464491e+28       1.312945e+30          0.037655     0.010247
4        21-50     999529  -4.682000e-03        1.203642e+28     3.480608e+28     2.131033e+28       1.561054e+30          0.054504     0.007712
5          50+     104917   1.323350e-01        9.773457e+27     4.346277e+28     2.876625e+28       5.626071e+30          0.284401     0.007273

[13:32:04]   → saved ./return_analysis_results_full/returns_by_token_bucket.csv
[13:32:04] 
  10c. Mean returns by portfolio value bucket ...
[13:33:17] Mean returns by portfolio value bucket:
  value_bucket         n  mean_baseline  mean_better_return  mean_safer_risk  mean_max_sharpe  mean_equal_weight  mean_mcap_weight  mean_market
0         $0-1  36188826   1.904521e+07        1.016661e+30     9.948350e+29     1.327780e+30       1.088000e+30          0.079335     0.016930
1        $1-10  33825301   8.712700e-02        6.285424e+29     5.922722e+29     8.485095e+29       6.428507e+29          0.070694     0.028409
2      $10-100  68249626   2.488300e-02        4.612199e+29     4.256968e+29     5.982395e+29       5.378733e+29          0.022182     0.010496
3      $100-1K  55940394   2.652900e-02        9.810462e+29     9.204549e+29     1.124865e+30       1.218909e+30          0.021327     0.004035
4      $1K-10K  32875467   1.365000e-03        1.335347e+30     1.141030e+30     1.664624e+30       2.085569e+30          0.019332     0.005694
5    $10K-100K   9983028  -2.192000e-03        1.915461e+30     1.794575e+30     2.690803e+30       3.544290e+30          0.015169     0.005070
6     $100K-1M   2066227   1.148000e-03        3.352883e+30     2.380498e+30     4.939384e+30       4.390896e+30          0.007120     0.003521
7         $1M+    491214   2.059000e-03        5.987568e+03     3.636666e+28     3.251670e+28       5.459194e+29          0.006139     0.004485

[13:33:17]   → saved ./return_analysis_results_full/returns_by_value_bucket.csv
[13:33:17] 
  10d. Extended OLS with polynomial + interaction + ETH/BTC features
[13:33:23] 
  Extended OLS: ret_baseline  R² = 0.0251
[13:33:23]   Extended OLS coefficients (ret_baseline):
                        coef   std_err      t_stat   p_value significant
const               8.425265  0.093569   90.043060  0.000000         ***
month              -0.319399  0.000662 -482.612134  0.000000         ***
log_value_usd      -0.007764  0.006549   -1.185363  0.235874            
num_tokens         -0.845680  0.028181  -30.008470  0.000000         ***
holds_eth          -0.084689  0.059691   -1.418785  0.155962            
holds_btc           3.680596  0.148238   24.828944  0.000000         ***
num_tokens_sq       0.003992  0.000227   17.601604  0.000000         ***
log_value_x_tokens -0.001279  0.001683   -0.759871  0.447332            
log_num_tokens      9.694680  0.124135   78.097867  0.000000         ***
eth_x_btc          -2.160939  0.311041   -6.947433  0.000000         ***

[13:33:23]   → saved ./return_analysis_results_full/ols_return_extended_coefs_baseline.csv
[13:33:25] 
  Extended OLS: ret_better_return  R² = 0.0234
[13:33:25]   Extended OLS coefficients (ret_better_return):
                         coef   std_err      t_stat   p_value significant
const               13.234426  0.090588  146.095240  0.000000         ***
month               -0.304049  0.000641 -474.539466  0.000000         ***
log_value_usd       -0.146916  0.006341  -23.170040  0.000000         ***
num_tokens          -0.649140  0.027283  -23.792502  0.000000         ***
holds_eth            0.468771  0.057789    8.111795  0.000000         ***
holds_btc            4.686839  0.143514   32.657609  0.000000         ***
num_tokens_sq        0.001858  0.000220    8.463008  0.000000         ***
log_value_x_tokens   0.021254  0.001630   13.041442  0.000000         ***
log_num_tokens       5.539143  0.120179   46.090616  0.000000         ***
eth_x_btc           -1.304298  0.301130   -4.331346  0.000015         ***

[13:33:25]   → saved ./return_analysis_results_full/ols_return_extended_coefs_better_return.csv
[13:33:28] 
  Extended OLS: ret_safer_risk  R² = 0.0255
[13:33:28]   Extended OLS coefficients (ret_safer_risk):
                         coef   std_err      t_stat   p_value significant
const               10.305014  0.087671  117.541260  0.000000         ***
month               -0.305166  0.000620 -492.125371  0.000000         ***
log_value_usd       -0.019415  0.006137   -3.163784  0.001557          **
num_tokens          -0.621177  0.026405  -23.524908  0.000000         ***
holds_eth           -0.046717  0.055928   -0.835297  0.403550            
holds_btc            2.934599  0.138894   21.128268  0.000000         ***
num_tokens_sq        0.003473  0.000213   16.343092  0.000000         ***
log_value_x_tokens  -0.011967  0.001577   -7.587048  0.000000         ***
log_num_tokens       7.480045  0.116311   64.310962  0.000000         ***
eth_x_btc           -1.932050  0.291436   -6.629415  0.000000         ***

[13:33:28]   → saved ./return_analysis_results_full/ols_return_extended_coefs_safer_risk.csv
[13:33:31] 
  Extended OLS: ret_max_sharpe  R² = 0.0211
[13:33:31]   Extended OLS coefficients (ret_max_sharpe):
                        coef   std_err      t_stat   p_value significant
const               8.272174  0.086709   95.401673  0.000000         ***
month              -0.270859  0.000613 -441.649884  0.000000         ***
log_value_usd      -0.105691  0.006069  -17.414049  0.000000         ***
num_tokens         -0.760646  0.026115  -29.126604  0.000000         ***
holds_eth           2.729392  0.055314   49.343223  0.000000         ***
holds_btc           6.184940  0.137369   45.024124  0.000000         ***
num_tokens_sq       0.003636  0.000210   17.299397  0.000000         ***
log_value_x_tokens -0.002192  0.001560   -1.405018  0.160016            
log_num_tokens      8.338912  0.115034   72.491112  0.000000         ***
eth_x_btc          -4.530334  0.288236  -15.717439  0.000000         ***

[13:33:31]   → saved ./return_analysis_results_full/ols_return_extended_coefs_max_sharpe.csv
[13:33:34] 
  Extended OLS: ret_equal_weight  R² = 0.0328
[13:33:34]   Extended OLS coefficients (ret_equal_weight):
                         coef   std_err      t_stat   p_value significant
const                7.500154  0.083148   90.202503  0.000000         ***
month               -0.319763  0.000588 -543.718278  0.000000         ***
log_value_usd        0.023378  0.005820    4.016737  0.000059         ***
num_tokens          -0.772010  0.025043  -30.827797  0.000000         ***
holds_eth           -0.480371  0.053043   -9.056282  0.000000         ***
holds_btc            2.845871  0.131728   21.604143  0.000000         ***
num_tokens_sq        0.004185  0.000202   20.762909  0.000000         ***
log_value_x_tokens  -0.015616  0.001496  -10.439540  0.000000         ***
log_num_tokens      11.019099  0.110309   99.892649  0.000000         ***
eth_x_btc           -1.913425  0.276399   -6.922692  0.000000         ***

[13:33:34]   → saved ./return_analysis_results_full/ols_return_extended_coefs_equal_weight.csv
[13:33:37] 
  Extended OLS: ret_mcap_weight  R² = 0.0239
[13:33:37]   Extended OLS coefficients (ret_mcap_weight):
                         coef   std_err      t_stat  p_value significant
const               10.046289  0.082440  121.862112      0.0         ***
month               -0.279353  0.000583 -479.085855      0.0         ***
log_value_usd       -0.059577  0.005770  -10.324447      0.0         ***
num_tokens          -0.780653  0.024829  -31.440686      0.0         ***
holds_eth            1.114958  0.052591   21.200527      0.0         ***
holds_btc            2.913307  0.130606   22.306049      0.0         ***
num_tokens_sq        0.001965  0.000200    9.833174      0.0         ***
log_value_x_tokens   0.029034  0.001483   19.576088      0.0         ***
log_num_tokens       6.883452  0.109370   62.937329      0.0         ***
eth_x_btc           -1.824550  0.274045   -6.657849      0.0         ***

[13:33:37]   → saved ./return_analysis_results_full/ols_return_extended_coefs_mcap_weight.csv
[13:33:41] 
  Extended OLS: alpha_baseline  R² = 0.0096
[13:33:41]   Extended OLS coefficients (alpha_baseline):
                        coef   std_err      t_stat   p_value significant
const               3.349558  0.085867   39.008574  0.000000         ***
month              -0.178816  0.000607 -294.426661  0.000000         ***
log_value_usd      -0.122402  0.006010  -20.365098  0.000000         ***
num_tokens         -0.248657  0.025862   -9.614881  0.000000         ***
holds_eth           0.235759  0.054778    4.303931  0.000017         ***
holds_btc           1.573004  0.136036   11.563139  0.000000         ***
num_tokens_sq       0.001685  0.000208    8.095988  0.000000         ***
log_value_x_tokens -0.012474  0.001545   -8.074970  0.000000         ***
log_num_tokens      4.213664  0.113917   36.988907  0.000000         ***
eth_x_btc          -1.790924  0.285438   -6.274294  0.000000         ***

[13:33:41]   → saved ./return_analysis_results_full/ols_alpha_extended_coefs_baseline.csv
[13:33:45] 
  Extended OLS: alpha_better_return  R² = 0.0086
[13:33:45]   Extended OLS coefficients (alpha_better_return):
                        coef   std_err      t_stat   p_value significant
const               6.507479  0.082026   79.334224  0.000000         ***
month              -0.164109  0.000580 -282.864877  0.000000         ***
log_value_usd      -0.152585  0.005742  -26.575726  0.000000         ***
num_tokens         -0.123784  0.024705   -5.010513  0.000001         ***
holds_eth           0.439743  0.052327    8.403735  0.000000         ***
holds_btc           2.176291  0.129951   16.747050  0.000000         ***
num_tokens_sq      -0.000400  0.000199   -2.013221  0.044091           *
log_value_x_tokens  0.012607  0.001476    8.542946  0.000000         ***
log_num_tokens      1.114193  0.108821   10.238762  0.000000         ***
eth_x_btc          -1.162343  0.272670   -4.262821  0.000020         ***

[13:33:45]   → saved ./return_analysis_results_full/ols_alpha_extended_coefs_better_return.csv
[13:33:48] 
  Extended OLS: alpha_safer_risk  R² = 0.0105
[13:33:48]   Extended OLS coefficients (alpha_safer_risk):
                        coef   std_err      t_stat   p_value significant
const               3.740441  0.080299   46.581459  0.000000         ***
month              -0.173386  0.000568 -305.282164  0.000000         ***
log_value_usd      -0.134503  0.005621  -23.930385  0.000000         ***
num_tokens         -0.127354  0.024185   -5.265897  0.000000         ***
holds_eth           0.048230  0.051225    0.941519  0.346439            
holds_btc           0.974980  0.127214    7.664074  0.000000         ***
num_tokens_sq       0.000845  0.000195    4.339041  0.000014         ***
log_value_x_tokens -0.011360  0.001445   -7.863666  0.000000         ***
log_num_tokens      3.741176  0.106530   35.118627  0.000000         ***
eth_x_btc          -1.398544  0.266928   -5.239400  0.000000         ***

[13:33:48]   → saved ./return_analysis_results_full/ols_alpha_extended_coefs_safer_risk.csv
[13:33:52] 
  Extended OLS: alpha_max_sharpe  R² = 0.0078
[13:33:52]   Extended OLS coefficients (alpha_max_sharpe):
                        coef   std_err      t_stat   p_value significant
const              -0.101302  0.073241   -1.383133  0.166624            
month              -0.130895  0.000518 -252.679553  0.000000         ***
log_value_usd      -0.120179  0.005127  -23.442347  0.000000         ***
num_tokens         -0.222786  0.022059  -10.099641  0.000000         ***
holds_eth           1.402680  0.046723   30.021458  0.000000         ***
holds_btc           3.437036  0.116032   29.621399  0.000000         ***
num_tokens_sq       0.001831  0.000178   10.313185  0.000000         ***
log_value_x_tokens -0.020629  0.001318  -15.655548  0.000000         ***
log_num_tokens      4.748429  0.097166   48.869398  0.000000         ***
eth_x_btc          -2.354121  0.243465   -9.669232  0.000000         ***

[13:33:52]   → saved ./return_analysis_results_full/ols_alpha_extended_coefs_max_sharpe.csv
[13:33:56] 
  Extended OLS: alpha_equal_weight  R² = 0.0142
[13:33:56]   Extended OLS coefficients (alpha_equal_weight):
                        coef   std_err      t_stat   p_value significant
const               0.795365  0.070757   11.240872  0.000000         ***
month              -0.172869  0.000500 -345.420445  0.000000         ***
log_value_usd      -0.000872  0.004953   -0.176008  0.860288            
num_tokens         -0.201917  0.021311   -9.474975  0.000000         ***
holds_eth          -0.109965  0.045138   -2.436210  0.014842           *
holds_btc           0.966112  0.112097    8.618552  0.000000         ***
num_tokens_sq       0.002238  0.000172   13.051878  0.000000         ***
log_value_x_tokens -0.031658  0.001273  -24.869748  0.000000         ***
log_num_tokens      6.176736  0.093870   65.800860  0.000000         ***
eth_x_btc          -1.789011  0.235208   -7.606093  0.000000         ***

[13:33:56]   → saved ./return_analysis_results_full/ols_alpha_extended_coefs_equal_weight.csv
[13:34:00] 
  Extended OLS: alpha_mcap_weight  R² = 0.0077
[13:34:00]   Extended OLS coefficients (alpha_mcap_weight):
                        coef   std_err      t_stat   p_value significant
const               3.321701  0.067546   49.176572  0.000000         ***
month              -0.129082  0.000478 -270.184936  0.000000         ***
log_value_usd      -0.037373  0.004728   -7.904607  0.000000         ***
num_tokens         -0.145214  0.020344   -7.138035  0.000000         ***
holds_eth           0.260826  0.043090    6.053049  0.000000         ***
holds_btc           1.599373  0.107011   14.945865  0.000000         ***
num_tokens_sq       0.000032  0.000164    0.197147  0.843712            
log_value_x_tokens  0.005930  0.001215    4.880043  0.000001         ***
log_num_tokens      1.942897  0.089611   21.681363  0.000000         ***
eth_x_btc          -1.715703  0.224537   -7.641085  0.000000         ***

[13:34:00]   → saved ./return_analysis_results_full/ols_alpha_extended_coefs_mcap_weight.csv

[13:34:00] ======================================================================
[13:34:00] RETURN ANALYSIS COMPLETE — 56.0 minutes
[13:34:00] All results saved to: ./return_analysis_results_full
[13:34:00] ======================================================================
[13:34:00] 
Output files:
[13:34:00]   alpha_descriptive_stats.csv                                      566 bytes
[13:34:00]   alpha_improvement_vs_baseline.csv                                300 bytes
[13:34:00]   correlation_returns_sample.csv                                   861 bytes
[13:34:00]   correlation_win_lose_sample.csv                                  759 bytes
[13:34:00]   cross_return_correlation.csv                                     351 bytes
[13:34:00]   eth_btc_prevalence.csv                                           125 bytes
[13:34:00]   ols_alpha_ethbtc_coefs_baseline.csv                              555 bytes
[13:34:00]   ols_alpha_ethbtc_coefs_better_return.csv                         574 bytes
[13:34:00]   ols_alpha_ethbtc_coefs_equal_weight.csv                          556 bytes
[13:34:00]   ols_alpha_ethbtc_coefs_max_sharpe.csv                            540 bytes
[13:34:00]   ols_alpha_ethbtc_coefs_mcap_weight.csv                           558 bytes
[13:34:00]   ols_alpha_ethbtc_coefs_safer_risk.csv                            532 bytes
[13:34:00]   ols_alpha_ethbtc_summary.csv                                     263 bytes
[13:34:00]   ols_alpha_extended_coefs_baseline.csv                            980 bytes
[13:34:00]   ols_alpha_extended_coefs_better_return.csv                       975 bytes
[13:34:00]   ols_alpha_extended_coefs_equal_weight.csv                        978 bytes
[13:34:00]   ols_alpha_extended_coefs_max_sharpe.csv                          975 bytes
[13:34:00]   ols_alpha_extended_coefs_mcap_weight.csv                         978 bytes
[13:34:00]   ols_alpha_extended_coefs_safer_risk.csv                          973 bytes
[13:34:00]   ols_return_coefs_baseline.csv                                    368 bytes
[13:34:00]   ols_return_coefs_better_return.csv                               366 bytes
[13:34:00]   ols_return_coefs_equal_weight.csv                                366 bytes
[13:34:00]   ols_return_coefs_max_sharpe.csv                                  370 bytes
[13:34:00]   ols_return_coefs_mcap_weight.csv                                 367 bytes
[13:34:00]   ols_return_coefs_safer_risk.csv                                  361 bytes
[13:34:00]   ols_return_ethbtc_coefs_baseline.csv                             548 bytes
[13:34:00]   ols_return_ethbtc_coefs_better_return.csv                        554 bytes
[13:34:00]   ols_return_ethbtc_coefs_equal_weight.csv                         553 bytes
[13:34:00]   ols_return_ethbtc_coefs_max_sharpe.csv                           519 bytes
[13:34:00]   ols_return_ethbtc_coefs_mcap_weight.csv                          557 bytes
[13:34:00]   ols_return_ethbtc_coefs_safer_risk.csv                           541 bytes
[13:34:00]   ols_return_ethbtc_summary.csv                                    262 bytes
[13:34:00]   ols_return_extended_coefs_baseline.csv                           940 bytes
[13:34:00]   ols_return_extended_coefs_better_return.csv                      959 bytes
[13:34:00]   ols_return_extended_coefs_equal_weight.csv                       962 bytes
[13:34:00]   ols_return_extended_coefs_max_sharpe.csv                         909 bytes
[13:34:00]   ols_return_extended_coefs_mcap_weight.csv                        954 bytes
[13:34:00]   ols_return_extended_coefs_safer_risk.csv                         962 bytes
[13:34:00]   ols_return_summary.csv                                           266 bytes
[13:34:00]   pearson_returns_full_data.csv                                    275 bytes
[13:34:00]   pearson_win_lose_full_data.csv                                   249 bytes
[13:34:00]   quantile_regression_returns_all.csv                            7,095 bytes
[13:34:00]   random_forest_return_ethbtc_importance.csv                     1,765 bytes
[13:34:00]   random_forest_return_importance.csv                            1,058 bytes
[13:34:00]   return_analysis_log.txt                                       71,359 bytes
[13:34:00]   return_descriptive_stats.csv                                     949 bytes
[13:34:00]   return_distribution_count.csv                                    658 bytes
[13:34:00]   return_distribution_pct.csv                                      488 bytes
[13:34:00]   returns_by_distance_bucket_better_return.csv                     819 bytes
[13:34:00]   returns_by_distance_bucket_max_sharpe.csv                        823 bytes
[13:34:00]   returns_by_distance_bucket_safer_risk.csv                        827 bytes
[13:34:00]   returns_by_eth_btc_holder.csv                                    484 bytes
[13:34:00]   returns_by_token_bucket.csv                                      911 bytes
[13:34:00]   returns_by_value_bucket.csv                                    1,182 bytes
[13:34:00]   strategy_vs_baseline_win_rates.csv                               362 bytes

Done.