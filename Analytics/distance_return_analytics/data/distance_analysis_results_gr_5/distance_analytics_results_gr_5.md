[23:46:52] ======================================================================
[23:46:52] MPT DISTANCE ANALYSIS — FILTERED: num_tokens >= 5
[23:46:52] ======================================================================

[23:46:57] Total rows (num_tokens >= 5): 32,351,085
[23:46:57] Adjusted sample sizes: correlation=25,880,868, regression=10,000,000

[23:46:57] ======================================================================
[23:46:57] SECTION 2: Full-data Pearson Correlation (all rows)
[23:46:57] ======================================================================
[23:46:57]   Computing Pearson on full data via DuckDB ...
[23:47:46]   Done in 49.4s
[23:47:46] Pearson r (full data, 0-100% scale):
               better_return  safer_risk  max_sharpe
feature                                             
month                 0.0744      0.0812      0.0869
log_value_usd         0.0125      0.0860     -0.0139
num_tokens            0.1934      0.2555      0.1409

[23:47:46]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/pearson_full_data.csv
[23:47:46] ======================================================================
[23:47:46] SECTION 3: Large-sample correlation (50M rows) — Pearson + Spearman
[23:47:46] ======================================================================
[23:47:46]   Drawing 26M sample ...
[23:48:49]   Sample loaded: 25,880,868 rows in 63.1s
[23:50:16] Pearson r (sample 26M):
target         better_return  max_sharpe  safer_risk
feature                                             
log_value_usd         0.0142     -0.0127      0.0786
month                 0.0556      0.0630      0.0668
num_tokens            0.1957      0.1429      0.2559

[23:50:16] Spearman ρ (sample 26M):
target         better_return  max_sharpe  safer_risk
feature                                             
log_value_usd         0.0135     -0.0253      0.0843
month                 0.0604      0.0703      0.0732
num_tokens            0.2794      0.1968      0.3647

[23:50:16]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/correlation_sample.csv
[23:50:16] ======================================================================
[23:50:16] SECTION 4: Distance Distribution (full data)
[23:50:16] ======================================================================
[23:50:16]   4a. Zero-distance & near-zero counts ...
[23:50:43] Zero-distance counts (exact 0  +  near-zero ≤1%):
               exact_0_count  exact_0_pct  near_0_count_(0,1%]  near_0_pct_(0,1%]  combined_<=1%_count  combined_<=1%_pct
optimisation                                                                                                             
better_return             18       0.0001               247737             0.7658               247755             0.7658
safer_risk                70       0.0002               157160             0.4858               157230             0.4860
max_sharpe                 6       0.0000                51970             0.1606                51976             0.1607

[23:50:43]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/zero_distance_counts.csv
[23:50:43]   4b. General distance distribution per bucket ...
[23:51:12] Distance distribution (% of wallets per bucket):
optimisation  better_return  max_sharpe  safer_risk
bucket                                             
0                      0.00        0.00        0.00
(0,1]                  0.77        0.16        0.49
(1,20]                11.02        4.78       10.30
(20,40]               17.14        7.90       23.41
(40,60]               21.17       14.28       30.89
(60,80]               21.56       22.30       24.60
(80,100]              28.35       50.58       10.32

[23:51:12] Distance distribution (count per bucket):
optimisation  better_return  max_sharpe  safer_risk
bucket                                             
0                        18           6          70
(0,1]                247737       51970      157160
(1,20]              3564395     1546601     3331366
(20,40]             5545933     2555036     7571784
(40,60]             6848886     4620927     9993431
(60,80]             6973285     7214747     7957617
(80,100]            9170766    16361721     3339648

[23:51:12]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/distance_distribution_pct.csv
[23:51:12]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/distance_distribution_count.csv
[23:51:12] ======================================================================
[23:51:12] SECTION 5: Profile of wallets with distance ≤1% (per optimisation)
[23:51:12] ======================================================================
[23:51:12] 
--- BETTER_RETURN (distance ≤1%) ---
[23:51:34]   Stats for better_return (dist ≤1%):
                         value
n                 2.477550e+05
mean_tokens       5.620730e+00
median_tokens     5.000000e+00
min_tokens        5.000000e+00
max_tokens        9.100000e+01
std_tokens        1.431365e+00
mean_value        3.151221e+05
median_value      2.503445e+02
min_value         0.000000e+00
max_value         1.175950e+10
std_value         3.128271e+07
mean_log_value    5.453518e+00
median_log_value  5.526825e+00

[23:51:34]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/low_dist_stats_better_return.csv
[23:51:44]   num_tokens distribution (top 20) for better_return dist ≤1%:
    num_tokens     cnt    pct  cum_pct
0            5  167567  67.63    67.63
1            6   47907  19.34    86.97
2            7   16506   6.66    93.63
3            8    7268   2.93    96.56
4            9    3498   1.41    97.97
5           10    1839   0.74    98.71
6           11    1051   0.42    99.13
7           12     631   0.25    99.38
8           13     426   0.17    99.55
9           14     280   0.11    99.66
10          15     162   0.07    99.73
11          16     171   0.07    99.80
12          17      95   0.04    99.84
13          18      78   0.03    99.87
14          19      62   0.03    99.90
15          20      45   0.02    99.92
16          21      19   0.01    99.93
17          22      29   0.01    99.94
18          23      25   0.01    99.95
19          24       8   0.00    99.95

[23:51:44]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/low_dist_tokens_dist_better_return.csv
[23:52:07]   Value distribution for better_return dist ≤1%:
  value_bucket    cnt    pct
0         $0-1  35852  14.47
1        $1-10  27474  11.09
2      $10-100  41979  16.94
3      $100-1K  47995  19.37
4      $1K-10K  53021  21.40
5    $10K-100K  30189  12.19
6     $100K-1M   8721   3.52
7         $1M+   2524   1.02

[23:52:07]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/low_dist_value_dist_better_return.csv
[23:52:07] 
--- SAFER_RISK (distance ≤1%) ---
[23:52:29]   Stats for safer_risk (dist ≤1%):
                         value
n                 1.572300e+05
mean_tokens       5.583438e+00
median_tokens     5.000000e+00
min_tokens        5.000000e+00
max_tokens        9.100000e+01
std_tokens        1.387033e+00
mean_value        3.462849e+05
median_value      8.368349e+01
min_value         0.000000e+00
max_value         1.175950e+10
std_value         3.459218e+07
mean_log_value    4.799011e+00
median_log_value  4.438921e+00

[23:52:29]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/low_dist_stats_safer_risk.csv
[23:52:39]   num_tokens distribution (top 20) for safer_risk dist ≤1%:
    num_tokens     cnt    pct  cum_pct
0            5  108819  69.21    69.21
1            6   29266  18.61    87.82
2            7    9892   6.29    94.11
3            8    4270   2.72    96.83
4            9    2061   1.31    98.14
5           10    1102   0.70    98.84
6           11     601   0.38    99.22
7           12     345   0.22    99.44
8           13     236   0.15    99.59
9           14     145   0.09    99.68
10          15     120   0.08    99.76
11          16     110   0.07    99.83
12          17      62   0.04    99.87
13          18      33   0.02    99.89
14          19      42   0.03    99.92
15          20      20   0.01    99.93
16          21      14   0.01    99.94
17          22      22   0.01    99.95
18          23      14   0.01    99.96
19          24       8   0.01    99.97

[23:52:39]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/low_dist_tokens_dist_safer_risk.csv
[23:53:02]   Value distribution for safer_risk dist ≤1%:
  value_bucket    cnt    pct
0         $0-1  33782  21.49
1        $1-10  19692  12.52
2      $10-100  27201  17.30
3      $100-1K  26808  17.05
4      $1K-10K  26238  16.69
5    $10K-100K  16292  10.36
6     $100K-1M   5419   3.45
7         $1M+   1798   1.14

[23:53:02]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/low_dist_value_dist_safer_risk.csv
[23:53:02] 
--- MAX_SHARPE (distance ≤1%) ---
[23:53:24]   Stats for max_sharpe (dist ≤1%):
                         value
n                 5.197600e+04
mean_tokens       5.615765e+00
median_tokens     5.000000e+00
min_tokens        5.000000e+00
max_tokens        4.800000e+01
std_tokens        1.355554e+00
mean_value        5.592747e+05
median_value      1.645681e+02
min_value         0.000000e+00
max_value         4.522355e+09
std_value         2.897614e+07
mean_log_value    5.220634e+00
median_log_value  5.109383e+00

[23:53:24]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/low_dist_stats_max_sharpe.csv
[23:53:34]   num_tokens distribution (top 20) for max_sharpe dist ≤1%:
    num_tokens    cnt    pct  cum_pct
0            5  35080  67.49    67.49
1            6  10100  19.43    86.92
2            7   3498   6.73    93.65
3            8   1493   2.87    96.52
4            9    750   1.44    97.96
5           10    426   0.82    98.78
6           11    237   0.46    99.24
7           12    129   0.25    99.49
8           13    100   0.19    99.68
9           14     34   0.07    99.75
10          15     22   0.04    99.79
11          16     30   0.06    99.85
12          17     11   0.02    99.87
13          18     11   0.02    99.89
14          19     13   0.03    99.92
15          20      8   0.02    99.94
16          21      3   0.01    99.95
17          22      3   0.01    99.96
18          23      9   0.02    99.98
19          24      2   0.00    99.98

[23:53:34]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/low_dist_tokens_dist_max_sharpe.csv
[23:53:57]   Value distribution for max_sharpe dist ≤1%:
  value_bucket    cnt    pct
0         $0-1  10260  19.74
1        $1-10   5887  11.33
2      $10-100   8172  15.72
3      $100-1K   8376  16.12
4      $1K-10K   9800  18.85
5    $10K-100K   6549  12.60
6     $100K-1M   2236   4.30
7         $1M+    696   1.34

[23:53:57]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/low_dist_value_dist_max_sharpe.csv
[23:53:57] ======================================================================
[23:53:57] SECTION 6: Wallet characteristics per distance bucket
[23:53:57] ======================================================================
[23:53:57] 
--- BETTER_RETURN ---
[23:54:20]   Bucket characteristics for better_return:
                   n  mean_tokens  median_tokens  mean_value_usd  median_value_usd  mean_log_value  median_log_value  p25_tokens  p75_tokens  p25_value  p75_value
dist_bucket                                                                                                                                                       
0_exact           18         5.06            5.0         4310.05              7.87            4.24              2.18         5.0         5.0       3.69    3397.24
(0,1]         247737         5.62            5.0       315144.70            250.36            5.45              5.53         5.0         6.0       9.14    3919.86
(1,20]       3564395         6.16            5.0       105451.80            531.16            5.97              6.28         5.0         6.0      47.22    3474.55
(20,40]      5545933         6.71            6.0        46641.65            660.27            6.21              6.49         5.0         7.0      85.10    3486.40
(40,60]      6848886         7.72            6.0        65733.80            745.86            6.32              6.62         5.0         9.0      94.20    3908.71
(60,80]      6973285         8.98            7.0       141953.60            779.63            6.36              6.66         5.0        10.0      90.16    4256.28
(80,100]     9170766        10.22            7.0       296873.73            663.30            6.14              6.50         6.0        11.0      51.13    4310.63
NaN               65        14.51           10.0         6390.50             56.22            4.86              4.05         6.0        20.0      13.62    1403.42

[23:54:20]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/bucket_stats_better_return.csv
[23:54:30]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/bucket_token_dist_better_return.csv
[23:54:30] 
--- SAFER_RISK ---
[23:54:55]   Bucket characteristics for safer_risk:
                   n  mean_tokens  median_tokens  mean_value_usd  median_value_usd  mean_log_value  median_log_value  p25_tokens  p75_tokens  p25_value  p75_value
dist_bucket                                                                                                                                                       
0_exact           70         5.19            5.0         7772.64             48.11            4.64              3.88         5.0         5.0       3.87    2119.54
(0,1]         157160         5.58            5.0       346435.64             83.71            4.80              4.44         5.0         6.0       2.18    2387.94
(1,20]       3331366         5.86            5.0        83609.75            473.95            5.88              6.16         5.0         6.0      41.51    3225.56
(20,40]      7571784         6.57            6.0        47059.39            531.49            6.03              6.28         5.0         7.0      66.69    2946.78
(40,60]      9993431         7.88            6.0        89216.79            629.66            6.16              6.45         5.0         9.0      72.75    3511.13
(60,80]      7957617         9.84            7.0       195739.14            882.70            6.39              6.78         6.0        11.0      85.70    4847.20
(80,100]     3339648        12.75            9.0       520042.48           1414.25            6.76              7.26         6.0        14.0     119.46    7624.63
NaN                9        26.00           21.0        12440.22            208.92            5.96              5.35        11.0        35.0      37.69    2867.64

[23:54:55]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/bucket_stats_safer_risk.csv
[23:55:05]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/bucket_token_dist_safer_risk.csv
[23:55:05] 
--- MAX_SHARPE ---
[23:55:28]   Bucket characteristics for max_sharpe:
                    n  mean_tokens  median_tokens  mean_value_usd  median_value_usd  mean_log_value  median_log_value  p25_tokens  p75_tokens  p25_value  p75_value
dist_bucket                                                                                                                                                        
0_exact             6         5.17            5.0        97415.74             54.33            5.14              3.42         5.0         5.0       7.34    1476.16
(0,1]           51970         5.62            5.0       559328.03            164.60            5.22              5.11         5.0         6.0       3.44    4245.77
(1,20]        1546601         6.10            5.0       154926.71            667.83            6.13              6.51         5.0         6.0      51.05    4480.66
(20,40]       2555036         6.51            6.0        51247.71            676.64            6.18              6.52         5.0         7.0      79.06    3630.38
(40,60]       4620927         7.31            6.0        51943.37            769.26            6.33              6.65         5.0         8.0      99.07    3912.35
(60,80]       7214747         8.24            6.0        62139.00            784.79            6.37              6.67         5.0         9.0     104.47    3944.20
(80,100]     16361721         9.18            7.0       231473.22            621.10            6.12              6.43         5.0        10.0      55.15    4002.87
NaN                77         9.64            7.0         2831.24            135.49            5.19              4.92         6.0        10.0      26.21    1113.85

[23:55:28]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/bucket_stats_max_sharpe.csv
[23:55:39]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/bucket_token_dist_max_sharpe.csv
[23:55:39] ======================================================================
[23:55:39] SECTION 7: Regression models
[23:55:39] ======================================================================
[23:55:39]   Drawing 10M sample for regression ...
[23:56:34]   Regression sample: 10,000,000 rows in 55.0s
[23:56:34] 
  --- 7a. OLS Regression ---
[23:56:36] 
  OLS: better_return
[23:56:37]   R² = 0.0431,  Adj-R² = 0.0431,  N = 10,000,000
[23:56:37]   Coefficients (better_return):
                    coef   std_err       t_stat  p_value significant
const          47.470545  0.033406  1421.025533      0.0         ***
month           0.063939  0.000842    75.933542      0.0         ***
log_value_usd  -0.037449  0.002954   -12.678516      0.0         ***
num_tokens      0.881816  0.001351   652.714346      0.0         ***

[23:56:37]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/ols_coefs_better_return.csv
[23:56:38] 
  OLS: safer_risk
[23:56:38]   R² = 0.0713,  Adj-R² = 0.0713,  N = 10,000,000
[23:56:38]   Coefficients (safer_risk):
                    coef   std_err       t_stat  p_value significant
const          39.083679  0.026540  1472.646895      0.0         ***
month           0.052128  0.000669    77.922987      0.0         ***
log_value_usd   0.136805  0.002347    58.297976      0.0         ***
num_tokens      0.902522  0.001073   840.868759      0.0         ***

[23:56:38]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/ols_coefs_safer_risk.csv
[23:56:40] 
  OLS: max_sharpe
[23:56:40]   R² = 0.0337,  Adj-R² = 0.0337,  N = 10,000,000
[23:56:40]   Coefficients (max_sharpe):
                    coef   std_err       t_stat  p_value significant
const          59.866944  0.029659  2018.524031      0.0         ***
month           0.234477  0.000748   313.644258      0.0         ***
log_value_usd  -0.065858  0.002622   -25.113199      0.0         ***
num_tokens      0.573360  0.001199   478.014826      0.0         ***

[23:56:40]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/ols_coefs_max_sharpe.csv
[23:56:40] 
  OLS Summary:
                   R2  Adj_R2     F_stat  F_pvalue
target                                            
better_return  0.0431  0.0431  150058.60       0.0
safer_risk     0.0713  0.0713  255890.84       0.0
max_sharpe     0.0337  0.0337  116397.62       0.0

[23:56:40]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/ols_summary.csv
[23:56:40] 
  --- 7b. Quantile Regression ---
[23:56:41]   Quantile regression on 2M rows
[23:56:41]   QuantReg: better_return, q=0.25 ...
[23:56:52]   QuantReg coefficients (better_return, q=0.25):
                    coef   std_err      t_stat  p_value  quantile         target significant
Intercept      20.794917  0.114178  182.127100      0.0      0.25  better_return         ***
month           0.061287  0.002776   22.073809      0.0      0.25  better_return         ***
log_value_usd   0.059582  0.010060    5.922895      0.0      0.25  better_return         ***
num_tokens      1.499334  0.006434  233.049689      0.0      0.25  better_return         ***

[23:56:52]   QuantReg: better_return, q=0.50 ...
[23:57:02]   QuantReg coefficients (better_return, q=0.5):
                    coef   std_err      t_stat  p_value  quantile         target significant
Intercept      44.227521  0.114208  387.253787      0.0       0.5  better_return         ***
month           0.088782  0.002879   30.838496      0.0       0.5  better_return         ***
log_value_usd  -0.204011  0.010102  -20.195013      0.0       0.5  better_return         ***
num_tokens      1.368447  0.004647  294.478031      0.0       0.5  better_return         ***

[23:57:02]   QuantReg: better_return, q=0.75 ...
[23:57:17]   QuantReg coefficients (better_return, q=0.75):
                    coef   std_err      t_stat  p_value  quantile         target significant
Intercept      73.423580  0.114857  639.258812      0.0      0.75  better_return         ***
month           0.068877  0.002899   23.762491      0.0      0.75  better_return         ***
log_value_usd  -0.499467  0.010520  -47.478088      0.0      0.75  better_return         ***
num_tokens      0.862369  0.003570  241.569171      0.0      0.75  better_return         ***

[23:57:17]   QuantReg: safer_risk, q=0.25 ...
[23:57:34]   QuantReg coefficients (safer_risk, q=0.25):
                    coef   std_err      t_stat   p_value  quantile      target significant
Intercept      20.146297  0.085310  236.155085  0.000000      0.25  safer_risk         ***
month           0.039995  0.002092   19.120741  0.000000      0.25  safer_risk         ***
log_value_usd   0.001159  0.007472    0.155059  0.876775      0.25  safer_risk            
num_tokens      1.453978  0.004893  297.177483  0.000000      0.25  safer_risk         ***

[23:57:34]   QuantReg: safer_risk, q=0.50 ...
[23:57:46]   QuantReg coefficients (safer_risk, q=0.5):
                    coef   std_err      t_stat  p_value  quantile      target significant
Intercept      36.503985  0.079887  456.947419      0.0       0.5  safer_risk         ***
month           0.042963  0.002014   21.334982      0.0       0.5  safer_risk         ***
log_value_usd  -0.109237  0.007066  -15.459032      0.0       0.5  safer_risk         ***
num_tokens      1.431326  0.003251  440.338067      0.0       0.5  safer_risk         ***

[23:57:46]   QuantReg: safer_risk, q=0.75 ...
[23:57:57]   QuantReg coefficients (safer_risk, q=0.75):
                    coef   std_err      t_stat  p_value  quantile      target significant
Intercept      53.468635  0.084225  634.830592      0.0      0.75  safer_risk         ***
month           0.047247  0.002119   22.294694      0.0      0.75  safer_risk         ***
log_value_usd  -0.040599  0.007833   -5.183142      0.0      0.75  safer_risk         ***
num_tokens      1.206479  0.002615  461.310759      0.0      0.75  safer_risk         ***

[23:57:57]   QuantReg: max_sharpe, q=0.25 ...
[23:58:08]   QuantReg coefficients (max_sharpe, q=0.25):
                    coef   std_err      t_stat  p_value  quantile      target significant
Intercept      39.171489  0.139063  281.682153      0.0      0.25  max_sharpe         ***
month           0.339401  0.003485   97.396154      0.0      0.25  max_sharpe         ***
log_value_usd  -0.060445  0.011823   -5.112630      0.0      0.25  max_sharpe         ***
num_tokens      0.901075  0.007613  118.362656      0.0      0.25  max_sharpe         ***

[23:58:08]   QuantReg: max_sharpe, q=0.50 ...
[23:58:20]   QuantReg coefficients (max_sharpe, q=0.5):
                    coef   std_err      t_stat  p_value  quantile      target significant
Intercept      63.739726  0.090828  701.765450      0.0       0.5  max_sharpe         ***
month           0.353964  0.002290  154.599667      0.0       0.5  max_sharpe         ***
log_value_usd  -0.107112  0.008034  -13.332386      0.0       0.5  max_sharpe         ***
num_tokens      0.543126  0.003696  146.961708      0.0       0.5  max_sharpe         ***

[23:58:20]   QuantReg: max_sharpe, q=0.75 ...
[23:58:31]   QuantReg coefficients (max_sharpe, q=0.75):
                    coef   std_err       t_stat  p_value  quantile      target significant
Intercept      85.431042  0.026368  3239.963751      0.0      0.75  max_sharpe         ***
month           0.092434  0.000639   144.558078      0.0      0.75  max_sharpe         ***
log_value_usd  -0.063713  0.002517   -25.315545      0.0      0.75  max_sharpe         ***
num_tokens      0.298395  0.000865   345.094527      0.0      0.75  max_sharpe         ***

[23:58:31]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/quantile_regression_all.csv
[23:58:31] 
  --- 7c. Random Forest (feature importance) ---
[23:58:32]   Random Forest on 2M rows, 32 cores
[23:58:32] 
  Training RF for better_return ...
[23:58:56]   RF better_return: R²=0.1205, MAE=21.82  (24.1s)
[23:58:56]   Feature importances (better_return):
         feature  importance         target  R2_test  MAE_test
2     num_tokens      0.6253  better_return   0.1205     21.82
1  log_value_usd      0.2046  better_return   0.1205     21.82
0          month      0.1700  better_return   0.1205     21.82

[23:58:56] 
  Training RF for safer_risk ...
[23:59:21]   RF safer_risk: R²=0.1734, MAE=16.61  (25.0s)
[23:59:21]   Feature importances (safer_risk):
         feature  importance      target  R2_test  MAE_test
2     num_tokens      0.7272  safer_risk   0.1734     16.61
1  log_value_usd      0.1475  safer_risk   0.1734     16.61
0          month      0.1254  safer_risk   0.1734     16.61

[23:59:21] 
  Training RF for max_sharpe ...
[23:59:46]   RF max_sharpe: R²=0.1312, MAE=18.12  (24.1s)
[23:59:46]   Feature importances (max_sharpe):
         feature  importance      target  R2_test  MAE_test
0          month      0.4268  max_sharpe   0.1312     18.12
2     num_tokens      0.3198  max_sharpe   0.1312     18.12
1  log_value_usd      0.2535  max_sharpe   0.1312     18.12

[23:59:46]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/random_forest_importance.csv
[23:59:46] ======================================================================
[23:59:46] SECTION 8: Additional Analytics
[23:59:46] ======================================================================
[23:59:46] 
  8a. Cross-target correlation (how correlated are the 3 distances?)
[00:00:13]   Cross-target Pearson (full data):
            pearson_r
corr_br_sr     0.4744
corr_br_ms     0.5961
corr_sr_ms     0.3651

[00:00:13]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/cross_target_correlation.csv
[00:00:13] 
  8b. Multi-optimal wallets (distance=0 for all 3 objectives)
[00:00:42]   Total wallets: 32,351,085.0
[00:00:42]   opt_br_and_sr: 17 (0.00%)
[00:00:42]   opt_br_and_ms: 0 (0.00%)
[00:00:42]   opt_sr_and_ms: 2 (0.00%)
[00:00:42]   opt_all_3: 0 (0.00%)
[00:00:42]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/multi_optimal_wallets.csv
[00:00:42] 
  8c. Extended OLS with polynomial + interaction features
[00:00:45] 
  Extended OLS: better_return  R² = 0.0836 (vs base OLS R² above)
[00:00:45]   Extended OLS coefficients (better_return):
                         coef   std_err      t_stat   p_value significant
const               -0.276304  0.089805   -3.076702  0.002093          **
month                0.035864  0.000825   43.461574  0.000000         ***
log_value_usd       -0.386840  0.003814 -101.422352  0.000000         ***
num_tokens          -0.638703  0.005042 -126.683872  0.000000         ***
num_tokens_sq        0.001081  0.000012   92.825848  0.000000         ***
log_value_x_tokens  -0.003846  0.000295  -13.055093  0.000000         ***
log_num_tokens      30.055087  0.055654  540.037843  0.000000         ***

[00:00:45]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/ols_extended_coefs_better_return.csv
[00:00:46] 
  Extended OLS: safer_risk  R² = 0.1329 (vs base OLS R² above)
[00:00:46]   Extended OLS coefficients (safer_risk):
                         coef   std_err      t_stat  p_value significant
const               -7.922414  0.070450 -112.454312      0.0         ***
month                0.024223  0.000647   37.418494      0.0         ***
log_value_usd       -0.214721  0.002992  -71.762188      0.0         ***
num_tokens          -0.584069  0.003955 -147.674691      0.0         ***
num_tokens_sq        0.000963  0.000009  105.448121      0.0         ***
log_value_x_tokens  -0.003297  0.000231  -14.266736      0.0         ***
log_num_tokens      29.568926  0.043659  677.269120      0.0         ***

[00:00:46]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/ols_extended_coefs_safer_risk.csv
[00:00:48] 
  Extended OLS: max_sharpe  R² = 0.0558 (vs base OLS R² above)
[00:00:48]   Extended OLS coefficients (max_sharpe):
                         coef   std_err      t_stat  p_value significant
const               28.830757  0.080541  357.963998      0.0         ***
month                0.216386  0.000740  292.387675      0.0         ***
log_value_usd       -0.331224  0.003421  -96.829630      0.0         ***
num_tokens          -0.464592  0.004522 -102.749163      0.0         ***
num_tokens_sq        0.000732  0.000010   70.106515      0.0         ***
log_value_x_tokens   0.002069  0.000264    7.831962      0.0         ***
log_num_tokens      19.719226  0.049913  395.075696      0.0         ***

[00:00:48]   → saved /anonymized/mpt-optimization-results/analysis_results_5plus/ols_extended_coefs_max_sharpe.csv
[00:00:48] 
  8d. Smallest-portfolio wallets (num_tokens = 5)
[00:01:17]   5-token wallets: 10,587,160 (32.73% of all)
[00:01:17]   opt_better_return at dist=0: 17 (0.00% of 5-token wallets)
[00:01:17]   opt_safer_risk at dist=0: 58 (0.00% of 5-token wallets)
[00:01:17]   opt_max_sharpe at dist=0: 5 (0.00% of 5-token wallets)

[00:01:17] ======================================================================
[00:01:17] ANALYSIS COMPLETE — 14.4 minutes
[00:01:17] All results saved to: /anonymized/mpt-optimization-results/analysis_results_5plus
[00:01:17] ======================================================================
[00:01:17] 
Output files:
[00:01:17]   analysis_log.txt                                         31,072 bytes
[00:01:17]   bucket_stats_better_return.csv                              835 bytes
[00:01:17]   bucket_stats_max_sharpe.csv                                 829 bytes
[00:01:17]   bucket_stats_safer_risk.csv                                 830 bytes
[00:01:17]   bucket_token_dist_better_return.csv                      30,794 bytes
[00:01:17]   bucket_token_dist_max_sharpe.csv                         27,053 bytes
[00:01:17]   bucket_token_dist_safer_risk.csv                         32,250 bytes
[00:01:17]   correlation_sample.csv                                      383 bytes
[00:01:17]   cross_target_correlation.csv                                101 bytes
[00:01:17]   distance_distribution_count.csv                             252 bytes
[00:01:17]   distance_distribution_pct.csv                               214 bytes
[00:01:17]   low_dist_stats_better_return.csv                            327 bytes
[00:01:17]   low_dist_stats_max_sharpe.csv                               325 bytes
[00:01:17]   low_dist_stats_safer_risk.csv                               326 bytes
[00:01:17]   low_dist_tokens_dist_better_return.csv                      849 bytes
[00:01:17]   low_dist_tokens_dist_max_sharpe.csv                         606 bytes
[00:01:17]   low_dist_tokens_dist_safer_risk.csv                         777 bytes
[00:01:17]   low_dist_value_dist_better_return.csv                       171 bytes
[00:01:17]   low_dist_value_dist_max_sharpe.csv                          164 bytes
[00:01:17]   low_dist_value_dist_safer_risk.csv                          171 bytes
[00:01:17]   multi_optimal_wallets.csv                                    97 bytes
[00:01:17]   ols_coefs_better_return.csv                                 364 bytes
[00:01:17]   ols_coefs_max_sharpe.csv                                    366 bytes
[00:01:17]   ols_coefs_safer_risk.csv                                    346 bytes
[00:01:17]   ols_extended_coefs_better_return.csv                        644 bytes
[00:01:17]   ols_extended_coefs_max_sharpe.csv                           622 bytes
[00:01:17]   ols_extended_coefs_safer_risk.csv                           646 bytes
[00:01:17]   ols_summary.csv                                             152 bytes
[00:01:17]   pearson_full_data.csv                                       138 bytes
[00:01:17]   quantile_regression_all.csv                               3,691 bytes
[00:01:17]   random_forest_importance.csv                                556 bytes
[00:01:17]   zero_distance_counts.csv                                    262 bytes

Done.