[11:51:38] ======================================================================
[11:51:38] RF + SHAP ANALYSIS — RETURN DRIVERS (EXTENDED FEATURES)
[11:51:38] ======================================================================

[11:51:41] Total rows: 239,627,516

[11:51:41] ======================================================================
[11:51:41] Drawing 10M sample for RF + SHAP
[11:51:41] ======================================================================
[11:57:41]   Raw sample: 10,001,339 rows in 360.3s
[11:57:41]   holds_eth=1: 776,420  holds_btc=1: 148,228
[11:57:42]   Sample ready: 10,001,339 rows
[11:57:42]   Feature summary:
[11:57:42]     month                 mean=46.3952  std=18.5923  min=1.0000  max=72.0000
[11:57:42]     log_value_usd         mean=4.2209  std=2.9251  min=0.0000  max=24.6221
[11:57:42]     num_tokens            mean=3.1997  std=3.5826  min=2.0000  max=1168.0000
[11:57:42]     beta_baseline         mean=0.9244  std=0.9088  min=-88.3298  max=144.9675
[11:57:42]     holds_eth             mean=0.0776  std=0.2676  min=0.0000  max=1.0000
[11:57:42]     holds_btc             mean=0.0148  std=0.1208  min=0.0000  max=1.0000

[11:57:42] Winsorising returns/alpha at p5/p95 ...
[11:57:45]   Done. Rows unchanged: 10,001,339

[11:57:45] ======================================================================
[11:57:45] RANDOM FOREST — 10,001,339 rows, 6 features, 32 cores
[11:57:45] Features: ['month', 'log_value_usd', 'num_tokens', 'beta_baseline', 'holds_eth', 'holds_btc']
[11:57:45] ======================================================================
[11:57:45] 
  Training RF for ret_baseline ...
[12:06:56]   RF ret_baseline: R²=0.6476, MAE=7.61  (550.7s)
[12:06:56]   Feature importances (ret_baseline):
         feature  importance        target  R2_test  MAE_test
0          month      0.6979  ret_baseline   0.6476      7.61
3  beta_baseline      0.2163  ret_baseline   0.6476      7.61
1  log_value_usd      0.0788  ret_baseline   0.6476      7.61
4      holds_eth      0.0042  ret_baseline   0.6476      7.61
2     num_tokens      0.0022  ret_baseline   0.6476      7.61
5      holds_btc      0.0006  ret_baseline   0.6476      7.61

[12:06:56] 
  Training RF for ret_better_return ...
[12:14:30]   RF ret_better_return: R²=0.6028, MAE=8.17  (453.9s)
[12:14:30]   Feature importances (ret_better_return):
         feature  importance             target  R2_test  MAE_test
0          month      0.7458  ret_better_return   0.6028      8.17
3  beta_baseline      0.1974  ret_better_return   0.6028      8.17
1  log_value_usd      0.0462  ret_better_return   0.6028      8.17
2     num_tokens      0.0058  ret_better_return   0.6028      8.17
4      holds_eth      0.0042  ret_better_return   0.6028      8.17
5      holds_btc      0.0006  ret_better_return   0.6028      8.17

[12:14:30] 
  Training RF for ret_safer_risk ...
[12:22:11]   RF ret_safer_risk: R²=0.6478, MAE=7.40  (461.2s)
[12:22:11]   Feature importances (ret_safer_risk):
         feature  importance          target  R2_test  MAE_test
0          month      0.7139  ret_safer_risk   0.6478       7.4
3  beta_baseline      0.2036  ret_safer_risk   0.6478       7.4
1  log_value_usd      0.0717  ret_safer_risk   0.6478       7.4
2     num_tokens      0.0061  ret_safer_risk   0.6478       7.4
4      holds_eth      0.0041  ret_safer_risk   0.6478       7.4
5      holds_btc      0.0006  ret_safer_risk   0.6478       7.4

[12:22:11] 
  Training RF for ret_max_sharpe ...
[12:30:28]   RF ret_max_sharpe: R²=0.5572, MAE=9.07  (497.2s)
[12:30:28]   Feature importances (ret_max_sharpe):
         feature  importance          target  R2_test  MAE_test
0          month      0.7637  ret_max_sharpe   0.5572      9.07
3  beta_baseline      0.1596  ret_max_sharpe   0.5572      9.07
1  log_value_usd      0.0575  ret_max_sharpe   0.5572      9.07
4      holds_eth      0.0097  ret_max_sharpe   0.5572      9.07
2     num_tokens      0.0083  ret_max_sharpe   0.5572      9.07
5      holds_btc      0.0012  ret_max_sharpe   0.5572      9.07

[12:30:28] 
  Training RF for ret_equal_weight ...
[12:38:24]   RF ret_equal_weight: R²=0.6478, MAE=7.54  (475.6s)
[12:38:24]   Feature importances (ret_equal_weight):
         feature  importance            target  R2_test  MAE_test
0          month      0.7939  ret_equal_weight   0.6478      7.54
3  beta_baseline      0.1445  ret_equal_weight   0.6478      7.54
1  log_value_usd      0.0503  ret_equal_weight   0.6478      7.54
2     num_tokens      0.0072  ret_equal_weight   0.6478      7.54
4      holds_eth      0.0038  ret_equal_weight   0.6478      7.54
5      holds_btc      0.0003  ret_equal_weight   0.6478      7.54

[12:38:24] 
  Training RF for ret_mcap_weight ...
[12:46:07]   RF ret_mcap_weight: R²=0.6217, MAE=7.30  (463.7s)
[12:46:07]   Feature importances (ret_mcap_weight):
         feature  importance           target  R2_test  MAE_test
0          month      0.7060  ret_mcap_weight   0.6217       7.3
3  beta_baseline      0.2113  ret_mcap_weight   0.6217       7.3
1  log_value_usd      0.0666  ret_mcap_weight   0.6217       7.3
4      holds_eth      0.0107  ret_mcap_weight   0.6217       7.3
2     num_tokens      0.0050  ret_mcap_weight   0.6217       7.3
5      holds_btc      0.0005  ret_mcap_weight   0.6217       7.3

[12:46:07] 
  Training RF for alpha_baseline ...
[12:53:53]   RF alpha_baseline: R²=0.4767, MAE=0.08  (465.9s)
[12:53:53]   Feature importances (alpha_baseline):
         feature  importance          target  R2_test  MAE_test
0          month      0.5517  alpha_baseline   0.4767      0.08
3  beta_baseline      0.2647  alpha_baseline   0.4767      0.08
1  log_value_usd      0.1674  alpha_baseline   0.4767      0.08
4      holds_eth      0.0095  alpha_baseline   0.4767      0.08
2     num_tokens      0.0055  alpha_baseline   0.4767      0.08
5      holds_btc      0.0013  alpha_baseline   0.4767      0.08

[12:53:53] 
  Training RF for alpha_better_return ...
[13:01:42]   RF alpha_better_return: R²=0.4223, MAE=0.08  (468.3s)
[13:01:42]   Feature importances (alpha_better_return):
         feature  importance               target  R2_test  MAE_test
0          month      0.5727  alpha_better_return   0.4223      0.08
3  beta_baseline      0.3126  alpha_better_return   0.4223      0.08
1  log_value_usd      0.0927  alpha_better_return   0.4223      0.08
2     num_tokens      0.0114  alpha_better_return   0.4223      0.08
4      holds_eth      0.0092  alpha_better_return   0.4223      0.08
5      holds_btc      0.0014  alpha_better_return   0.4223      0.08

[13:01:42] 
  Training RF for alpha_safer_risk ...
[13:10:18]   RF alpha_safer_risk: R²=0.4693, MAE=0.07  (515.8s)
[13:10:18]   Feature importances (alpha_safer_risk):
         feature  importance            target  R2_test  MAE_test
0          month      0.5538  alpha_safer_risk   0.4693      0.07
3  beta_baseline      0.2915  alpha_safer_risk   0.4693      0.07
1  log_value_usd      0.1271  alpha_safer_risk   0.4693      0.07
2     num_tokens      0.0164  alpha_safer_risk   0.4693      0.07
4      holds_eth      0.0096  alpha_safer_risk   0.4693      0.07
5      holds_btc      0.0015  alpha_safer_risk   0.4693      0.07

[13:10:18] 
  Training RF for alpha_max_sharpe ...
[13:18:28]   RF alpha_max_sharpe: R²=0.3765, MAE=0.09  (490.3s)
[13:18:28]   Feature importances (alpha_max_sharpe):
         feature  importance            target  R2_test  MAE_test
0          month      0.5570  alpha_max_sharpe   0.3765      0.09
3  beta_baseline      0.2871  alpha_max_sharpe   0.3765      0.09
1  log_value_usd      0.1185  alpha_max_sharpe   0.3765      0.09
2     num_tokens      0.0183  alpha_max_sharpe   0.3765      0.09
4      holds_eth      0.0165  alpha_max_sharpe   0.3765      0.09
5      holds_btc      0.0025  alpha_max_sharpe   0.3765      0.09

[13:18:28] 
  Training RF for alpha_equal_weight ...
[13:28:00]   RF alpha_equal_weight: R²=0.4686, MAE=0.08  (572.1s)
[13:28:00]   Feature importances (alpha_equal_weight):
         feature  importance              target  R2_test  MAE_test
0          month      0.6242  alpha_equal_weight   0.4686      0.08
3  beta_baseline      0.2578  alpha_equal_weight   0.4686      0.08
1  log_value_usd      0.0964  alpha_equal_weight   0.4686      0.08
2     num_tokens      0.0139  alpha_equal_weight   0.4686      0.08
4      holds_eth      0.0070  alpha_equal_weight   0.4686      0.08
5      holds_btc      0.0008  alpha_equal_weight   0.4686      0.08

[13:28:00] 
  Training RF for alpha_mcap_weight ...
[13:36:15]   RF alpha_mcap_weight: R²=0.4612, MAE=0.07  (494.8s)
[13:36:15]   Feature importances (alpha_mcap_weight):
         feature  importance             target  R2_test  MAE_test
0          month      0.5617  alpha_mcap_weight   0.4612      0.07
3  beta_baseline      0.2531  alpha_mcap_weight   0.4612      0.07
1  log_value_usd      0.1476  alpha_mcap_weight   0.4612      0.07
4      holds_eth      0.0273  alpha_mcap_weight   0.4612      0.07
2     num_tokens      0.0091  alpha_mcap_weight   0.4612      0.07
5      holds_btc      0.0013  alpha_mcap_weight   0.4612      0.07

[13:36:15]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_shap_importance.csv

[13:36:15] ======================================================================
[13:36:15] COMPARISON: 3-feature vs 6-feature RF
[13:36:15] ======================================================================
[13:36:15] 
  Training 3-feature RF for ret_baseline ...
[13:41:58]   RF-3feat ret_baseline: R²=0.5496, MAE=9.69  (342.9s)
[13:41:58] 
  Training 3-feature RF for ret_better_return ...
[13:46:29]   RF-3feat ret_better_return: R²=0.5146, MAE=10.03  (271.1s)
[13:46:29] 
  Training 3-feature RF for ret_safer_risk ...
[13:51:07]   RF-3feat ret_safer_risk: R²=0.5577, MAE=9.23  (278.7s)
[13:51:07] 
  Training 3-feature RF for ret_max_sharpe ...
[13:56:37]   RF-3feat ret_max_sharpe: R²=0.4859, MAE=10.59  (329.8s)
[13:56:37] 
  Training 3-feature RF for ret_equal_weight ...
[14:02:14]   RF-3feat ret_equal_weight: R²=0.5809, MAE=8.84  (336.9s)
[14:02:14] 
  Training 3-feature RF for ret_mcap_weight ...
[14:06:46]   RF-3feat ret_mcap_weight: R²=0.5230, MAE=9.14  (271.5s)
[14:06:46] 
  Training 3-feature RF for alpha_baseline ...
[14:11:39]   RF-3feat alpha_baseline: R²=0.3609, MAE=0.09  (293.2s)
[14:11:39] 
  Training 3-feature RF for alpha_better_return ...
[14:16:47]   RF-3feat alpha_better_return: R²=0.3223, MAE=0.10  (308.0s)
[14:16:47] 
  Training 3-feature RF for alpha_safer_risk ...
[14:21:33]   RF-3feat alpha_safer_risk: R²=0.3680, MAE=0.09  (286.3s)
[14:21:33] 
  Training 3-feature RF for alpha_max_sharpe ...
[14:26:49]   RF-3feat alpha_max_sharpe: R²=0.2920, MAE=0.10  (315.8s)
[14:26:49] 
  Training 3-feature RF for alpha_equal_weight ...
[14:31:28]   RF-3feat alpha_equal_weight: R²=0.3877, MAE=0.09  (279.4s)
[14:31:28] 
  Training 3-feature RF for alpha_mcap_weight ...
[14:36:02]   RF-3feat alpha_mcap_weight: R²=0.3521, MAE=0.08  (273.3s)
[14:36:02]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_3feat_importance.csv
[14:36:02] 
  R² comparison (3-feature vs 6-feature):
[14:36:02]   Target                       R²(3-feat)    R²(6-feat)       ΔR²
[14:36:02]   ret_baseline                     0.5496        0.6476   +0.0980
[14:36:02]   ret_better_return                0.5146        0.6028   +0.0882
[14:36:02]   ret_safer_risk                   0.5577        0.6478   +0.0901
[14:36:02]   ret_max_sharpe                   0.4859        0.5572   +0.0713
[14:36:02]   ret_equal_weight                 0.5809        0.6478   +0.0669
[14:36:02]   ret_mcap_weight                  0.5230        0.6217   +0.0987
[14:36:02]   alpha_baseline                   0.3609        0.4767   +0.1158
[14:36:02]   alpha_better_return              0.3223        0.4223   +0.1000
[14:36:02]   alpha_safer_risk                 0.3680        0.4693   +0.1013
[14:36:02]   alpha_max_sharpe                 0.2920        0.3765   +0.0845
[14:36:02]   alpha_equal_weight               0.3877        0.4686   +0.0809
[14:36:02]   alpha_mcap_weight                0.3521        0.4612   +0.1091

[14:36:02] ======================================================================
[14:36:02] SHAP ANALYSIS — 30K subsample per model
[14:36:02] ======================================================================
[14:36:03] shap library loaded successfully
[14:36:03] 
  Computing SHAP for ret_baseline ...


[17:37:59]   SHAP computed: 30,000 samples in 10916.7s
[17:38:00]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_shap_values_baseline.csv
[17:38:00]   Mean |SHAP| (ret_baseline):
         feature      feature_display  mean_abs_shap        target
0          month          Entry month        11.3671  ret_baseline
3  beta_baseline       Portfolio beta         2.6894  ret_baseline
1  log_value_usd  Log portfolio value         0.9385  ret_baseline
4      holds_eth    Holds ETH wrapper         0.2000  ret_baseline
2     num_tokens          Token count         0.1027  ret_baseline
5      holds_btc    Holds BTC wrapper         0.0287  ret_baseline

[17:38:03]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_shap_summary_baseline.pdf
[17:38:03]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_shap_month_dep_baseline.pdf
[17:38:03]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_shap_beta_dep_baseline.pdf
[17:38:03]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_shap_value_dep_baseline.pdf
[17:38:03] 
  Computing SHAP for ret_better_return ...
[20:55:54]   SHAP computed: 30,000 samples in 11871.1s
[20:55:55]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_shap_values_better_return.csv
[20:55:55]   Mean |SHAP| (ret_better_return):
         feature      feature_display  mean_abs_shap             target
0          month          Entry month        10.9958  ret_better_return
3  beta_baseline       Portfolio beta         2.3635  ret_better_return
1  log_value_usd  Log portfolio value         0.6884  ret_better_return
4      holds_eth    Holds ETH wrapper         0.2047  ret_better_return
2     num_tokens          Token count         0.2042  ret_better_return
5      holds_btc    Holds BTC wrapper         0.0264  ret_better_return

[20:55:57]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_shap_summary_better_return.pdf
[20:55:57]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_shap_month_dep_better_return.pdf
[20:55:57]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_shap_beta_dep_better_return.pdf
[20:55:57]   → saved /anonymized/mpt-optimization-results/rf_shap_results_extended/rf_shap_value_dep_better_return.pdf
[20:55:57] 