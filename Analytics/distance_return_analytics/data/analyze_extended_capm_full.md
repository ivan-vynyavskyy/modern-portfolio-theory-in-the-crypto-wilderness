(erc_20_transfers_env) Apptainer> python analyze_extended_capm.py 

================================================================================
  LOADING DATA
================================================================================

Total rows: 235,868,007
Output directory: ./camp_analysis_results_full

================================================================================
  DATA COVERAGE
================================================================================


--- Data Coverage ---
                             value
total_rows        235868007.000000
n_blocks                 72.000000
n_with_mcap       235126564.000000
n_without_mcap       741443.000000
mcap_coverage_pct        99.690000
mean_num_tokens           3.142781
median_num_tokens         2.000000
mean_value_usd        50981.934239
median_value_usd         54.135055


================================================================================
  1 & 2. L1 Distance Statistics
================================================================================


--- L1 Distances — Overall (all blocks) ---
                    l1_metric  mean (%)  median (%)   std (%)
0        l1_gap_better_return 27.542532   12.379713 31.714281
1           l1_gap_safer_risk 13.271790    0.000000 22.769096
2           l1_gap_max_sharpe 46.528550   50.053334 33.903762
3        l1_baseline_vs_equal 37.707924   40.000000 15.907676
4         l1_baseline_vs_mcap 38.431641   31.155154 30.306228
5   l1_better_return_vs_equal 35.613722   39.999999 17.887974
6      l1_safer_risk_vs_equal 34.014474   39.999999 14.271763
7      l1_max_sharpe_vs_equal 46.356928   40.000000 14.622327
8    l1_better_return_vs_mcap 44.944037   42.626549 29.451369
9       l1_safer_risk_vs_mcap 38.046116   33.275716 28.441808
10      l1_max_sharpe_vs_mcap 48.662518   49.947501 34.172330

  [saved] L1 distances per block -> ./camp_analysis_results_full/l1_distances_per_block.csv

================================================================================
  3. Return Statistics (20-day forward)
================================================================================


--- Returns — Overall (all blocks) ---
       return_metric  median (%)
0       ret_baseline   -1.867986
1  ret_better_return   -2.326504
2     ret_safer_risk   -1.998242
3     ret_max_sharpe   -1.916618
4   ret_equal_weight   -2.820286
5    ret_mcap_weight   -0.238138
6      market_return   -0.519686

  [saved] Returns per block -> ./camp_analysis_results_full/returns_per_block.csv

================================================================================
  4. Beta & Alpha Statistics
================================================================================

--- Betas — Overall ---
               metric     mean   median      std
0       beta_baseline 0.925186 0.988795 0.921921
1  beta_better_return 0.893539 0.948707 1.231020
2     beta_safer_risk 0.875218 0.926542 0.887251
3     beta_max_sharpe 0.894067 0.971288 0.866851
4   beta_equal_weight 0.895557 0.915591 0.866944
5    beta_mcap_weight 0.823828 0.933793 0.844155

--- Alphas — Overall ---
                metric                                 mean (%)  median (%)                                     std (%)
0       alpha_baseline                         283316272.900127   -1.914306                        3289320873062.762207
1  alpha_better_return  89625875094297916482056475377664.000000   -2.314412  98350393449820581888425293701447680.000000
2     alpha_safer_risk  82306480027767326946909220765696.000000   -1.847118  95041178580588258022628506063601664.000000
3     alpha_max_sharpe 113877889326455071497353613541376.000000   -1.918625 143776163475434095632946547333791744.000000
4   alpha_equal_weight 116604672694984144213508767088640.000000   -2.775545 106511348886251823376360078198177792.000000
5    alpha_mcap_weight                                 3.039968   -0.298615                                  885.442748

  [saved] Betas & Alphas per block -> ./camp_analysis_results_full/betas_alphas_per_block.csv

================================================================================
  5. Alpha Distribution — Do Wallets Outperform On a Risk-Adjusted Basis?
================================================================================


--- Alpha Summary by Strategy ---
        strategy                                 mean (%)  median (%)                                     std (%)     p5 (%)    p25 (%)  p75 (%)   p95 (%)  frac_positive
0       baseline                         283316272.900234   -1.914306                        3289320873062.961914 -29.933588 -11.650101 3.879236 39.156822       0.389068
1  better_return  89625875094297916482056475377664.000000   -2.314412  98350393449819862465406419028934656.000000 -29.018153 -11.795166 3.900947 39.643845       0.377730
2     safer_risk  82306480027767344961307730247680.000000   -1.847118  95041178580588018214955547839430656.000000 -27.777434 -10.765792 3.809541 37.627612       0.384538
3     max_sharpe 113877889326455107526150632505344.000000   -1.918625 143776163475433634464344704595001344.000000 -31.257262 -12.397172 3.479191 38.190348       0.379890
4   equal_weight 116604672694984126199110257606656.000000   -2.775545 106511348886251768036127857069522944.000000 -25.287561 -10.877716 4.603697 39.193588       0.379239
5    mcap_weight                                 3.039968   -0.298615                                  885.442748 -26.688247  -8.801029 2.269123 33.191530       0.397143

  [saved] Alpha distribution (binned) -> ./camp_analysis_results_full/alpha_distribution_binned.csv
  [saved] Alpha stats per block -> ./camp_analysis_results_full/alpha_per_block.csv

================================================================================
  6. Beta Distribution — How Much Market Risk Do Wallets Take?
================================================================================


--- Beta Summary by Strategy ---
        strategy     mean   median      std       p5      p25      p75      p95  frac_above_1  frac_negative
0       baseline 0.925186 0.988795 0.921921 0.005243 0.580460 1.283587 1.751761      0.491096       0.030847
1  better_return 0.893539 0.948707 1.231020 0.003831 0.530170 1.249101 1.737966      0.459469       0.033680
2     safer_risk 0.875218 0.926542 0.887251 0.005814 0.512791 1.223284 1.674892      0.438581       0.029421
3     max_sharpe 0.894067 0.971288 0.866851 0.002921 0.530600 1.263101 1.763088      0.475037       0.038606
4   equal_weight 0.895557 0.915591 0.866944 0.007456 0.599983 1.195160 1.627508      0.421178       0.027315
5    mcap_weight 0.823828 0.933793 0.844155 0.000598 0.209317 1.256640 1.714574      0.452991       0.042725

  [saved] Beta distribution (binned) -> ./camp_analysis_results_full/beta_distribution_binned.csv
  [saved] Beta stats per block -> ./camp_analysis_results_full/beta_per_block.csv

================================================================================
  7. Does L1 Gap Predict Alpha?
================================================================================

--- Pearson Correlations (full data) ---

  L1 better_ret vs α_baseline                         r = +0.0000
  L1 safer_risk vs α_baseline                         r = -0.0001
  L1 max_sharpe vs α_baseline                         r = -0.0001
  L1 base_vs_equal vs α_baseline                      r = +0.0000
  L1 base_vs_mcap vs α_baseline                       r = -0.0001
  num_tokens vs α_baseline                            r = -0.0000
  β_baseline vs α_baseline                            r = +0.0010
  L1 safer_risk vs α_safer_risk                       r = -0.0004
  L1 max_sharpe vs α_max_sharpe                       r = -0.0001
  L1 base_vs_equal vs α_equal_weight                  r = -0.0013
  L1 base_vs_mcap vs α_mcap_weight                    r = +0.0034

--- Spearman Correlations (sample n=5,000,000) ---

  l1_gap_safer_risk vs alpha_baseline                      ρ = +0.0264  (p = 0.00e+00)
  l1_gap_better_return vs alpha_baseline                   ρ = +0.0681  (p = 0.00e+00)
  l1_gap_max_sharpe vs alpha_baseline                      ρ = +0.1012  (p = 0.00e+00)
  l1_baseline_vs_equal vs alpha_baseline                   ρ = -0.0037  (p = 5.43e-17)
  l1_baseline_vs_mcap vs alpha_baseline                    ρ = +0.0174  (p = 0.00e+00)
  num_tokens vs alpha_baseline                             ρ = -0.0209  (p = 0.00e+00)
  beta_baseline vs alpha_baseline                          ρ = -0.1431  (p = 0.00e+00)
  l1_baseline_vs_equal vs alpha_equal_weight               ρ = +0.0196  (p = 0.00e+00)
  l1_baseline_vs_mcap vs alpha_mcap_weight                 ρ = +0.0288  (p = 0.00e+00)

  [saved] Spearman L1 vs Alpha -> ./camp_analysis_results_full/spearman_l1_vs_alpha.csv
  [saved] L1 vs Alpha (binned ventiles) -> ./camp_analysis_results_full/l1_vs_alpha_binned.csv

================================================================================
  8. Did the Optimizer Improve Risk-Adjusted Returns?
================================================================================


--- Strategy Improvement Over Baseline — Overall ---
        strategy                         mean_ret_diff_pp  median_ret_diff_pp                       mean_alpha_diff_pp  median_alpha_diff_pp  ret_hit_rate_pct  alpha_hit_rate_pct
0  better_return  89625875094297898467657965895680.000000           -0.000000  89625875094297898467657965895680.000000             -0.000000         48.533542           48.612278
1     safer_risk  82306480027767344961307730247680.000000            0.000000  82306480027767344961307730247680.000000              0.000000         51.006960           51.051860
2     max_sharpe 113877889326455107526150632505344.000000           -0.001249 113877889326455107526150632505344.000000             -0.009112         49.773607           49.115918
3   equal_weight 116604672694984144213508767088640.000000            0.001478 116604672694984144213508767088640.000000              0.000089         50.374229           50.155005
4    mcap_weight                        -284209673.294453            0.022026                        -284209673.214631              0.083681         53.348880           53.490530

  [saved] Optimizer improvement per block -> ./camp_analysis_results_full/optimizer_improvement_per_block.csv
  [saved] Alpha diff distribution (binned) -> ./camp_analysis_results_full/alpha_diff_distribution_binned.csv

================================================================================
  9. Does the Optimizer Increase or Decrease Market Risk?
================================================================================


--- Beta Change (strategy − baseline) — Overall ---
        strategy   mean_Δβ  median_Δβ   std_Δβ  pct_increased
0  better_return -0.031647   0.000000 0.840996      54.637244
1     safer_risk -0.049968  -0.000000 0.275299      39.159054
2     max_sharpe -0.031120   0.000000 0.857994      44.226970
3   equal_weight -0.029629  -0.012096 0.645790      44.300750
4    mcap_weight -0.101682  -0.007550 0.819978      43.006450

  [saved] Beta change by num_tokens -> ./camp_analysis_results_full/beta_change_by_num_tokens.csv
  [saved] Beta change per block -> ./camp_analysis_results_full/beta_change_per_block.csv

================================================================================
  EXTRA: Cross-Strategy Return & Alpha Comparison
================================================================================

--- How often is each strategy the BEST (highest return)? ---
                        pct_best
pct_best_better_return  9.814595
pct_best_safer_risk     5.581400
pct_best_max_sharpe    23.279969
pct_best_equal_weight  13.451929
pct_best_baseline      21.340863

--- How far is baseline from each strategy (mean L1 %)? ---
                                      value
mean_l1_baseline_vs_equal         37.707924
mean_l1_baseline_vs_mcap          38.431641
mean_l1_baseline_vs_better_return 27.542532
mean_l1_baseline_vs_safer_risk    13.271790
mean_l1_baseline_vs_max_sharpe    46.528550
pct_closer_to_equal_than_safer    12.424858
pct_closer_to_mcap_than_safer     11.880944

  [saved] Cross-strategy returns/alpha per block -> ./camp_analysis_results_full/cross_strategy_per_block.csv

================================================================================
  EXTRA: Does MPT Optimization Move Wallets Toward or Away From Naive Strategies?
================================================================================


--- MPT vs Naive Direction — Overall ---
    mpt_strategy naive_strategy  mean_delta_l1  median_delta_l1  pct_moved_closer  pct_moved_farther  pct_unchanged
0  better_return   equal_weight      -0.041884         0.000000         32.947151          20.107545      46.945304
1  better_return    mcap_weight       0.130248         0.000000         22.693291          37.082725      39.909637
2     safer_risk   equal_weight      -0.073869        -0.000000         27.402573           9.132983      63.464444
3     safer_risk    mcap_weight      -0.007711         0.000000         18.768491          17.871657      63.045506
4     max_sharpe   equal_weight       0.172980         0.011087         10.155514          50.785693      39.058793
5     max_sharpe    mcap_weight       0.204618         0.033665         29.754668          52.022595      17.908391

  [saved] MPT vs Naive direction per block -> ./camp_analysis_results_full/mpt_vs_naive_direction_per_block.csv

================================================================================
  DONE
================================================================================

All CSV files saved to: ./camp_analysis_results_full/
Files:
  alpha_diff_distribution_binned.csv                     0.05 MB
  alpha_distribution_binned.csv                          0.05 MB
  alpha_per_block.csv                                    0.03 MB
  beta_change_by_num_tokens.csv                          0.00 MB
  beta_change_per_block.csv                              0.01 MB
  beta_distribution_binned.csv                           0.02 MB
  beta_per_block.csv                                     0.02 MB
  betas_alphas_per_block.csv                             0.05 MB
  cross_strategy_per_block.csv                           0.02 MB
  l1_distances_per_block.csv                             0.04 MB
  l1_vs_alpha_binned.csv                                 0.01 MB
  mpt_vs_naive_direction_per_block.csv                   0.04 MB
  optimizer_improvement_per_block.csv                    0.02 MB
  returns_per_block.csv                                  0.03 MB
  spearman_l1_vs_alpha.csv                               0.00 MB