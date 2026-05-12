#!/usr/bin/env Rscript
# =============================================================================
# plot_beat_market_quantile_box.R
#
# Description:
#   Custom quantile boxplot, one box per strategy on a single panel.
#   Box layers (drawn outermost first):
#     - thin line:    Q01  -- Q99   (full extent shown)
#     - thick line:   Q05  -- Q95
#     - filled box:   Q25  -- Q75   (interquartile range)
#     - inner line:   Q50          (median across wallets)
#   Quantiles are first aggregated across blocks (default: median).
#
# Input:
#   Long quantile CSV from analyze_pct_beat_market.py with columns:
#     block_number, block_date, n_wallets, strategy, metric,
#     quantile, value
#
# Output:
#   Two PDFs in --out-dir:
#     quantile_box_excess_return.pdf
#     quantile_box_alpha.pdf
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(optparse)
})

option_list <- list(
  make_option("--in-csv",  type = "character",
              default = "../data/excess_return_quantiles_per_block.csv"),
  make_option("--out-dir", type = "character",
              default = "../graphics"),
  make_option("--agg",     type = "character", default = "median",
              help = "Across-block aggregation: 'median' (default) or 'mean'"),
  make_option("--width",   type = "double", default = 12),
  make_option("--height",  type = "double", default = 6)
)
opts <- parse_args(OptionParser(option_list = option_list))


script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grep("^--file=", args)]
  if (length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg)))
  } else {
    getwd()
  }
})()
source(file.path(script_dir, "..", "..", "shared", "plot_theme.R"))
opts$width <- apply_plot_theme_fonts(opts$width)

stopifnot(opts$agg %in% c("median", "mean"))
agg_fn <- if (opts$agg == "median") median else mean

dir.create(opts$`out-dir`, showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# STRATEGY ORDER + COLOUR PALETTE
# Mirrors plot_panel_delta_and_market.R for paper-wide consistency.
STRATEGY_LEVELS <- c("baseline", "better_return", "safer_risk",
                     "max_sharpe", "equal_weight", "mcap_weight")

STRATEGY_LABELS <- c(
  baseline      = "Baseline (actual)",
  better_return = "MaxRet",
  safer_risk    = "MinVar",
  max_sharpe    = "MaxSR",
  equal_weight  = "Equal-weight",
  mcap_weight   = "MCap-weight"
)

STRATEGY_PAL <- c(
  "Baseline (actual)" = "#E31A1C",
  "MaxRet"            = "#1F78B4",
  "MinVar"            = "#33A02C",
  "MaxSR"             = "#FF7F00",
  "Equal-weight"      = "#6A3D9A",
  "MCap-weight"       = "#B15928"
)

# -----------------------------------------------------------------------------
# LOAD + AGGREGATE + PIVOT WIDE (one row per strategy x metric)
df_raw <- read.csv(opts$`in-csv`)

df_box <- df_raw %>%
  group_by(strategy, metric, quantile) %>%
  summarise(value_pct = agg_fn(value, na.rm = TRUE) * 100, .groups = "drop") %>%
  mutate(qkey = paste0("q", sprintf("%02d", round(quantile * 100)))) %>%
  select(-quantile) %>%
  pivot_wider(names_from = qkey, values_from = value_pct) %>%
  mutate(
    strategy_f = factor(strategy,
                        levels = STRATEGY_LEVELS,
                        labels = STRATEGY_LABELS[STRATEGY_LEVELS])
  )

# -----------------------------------------------------------------------------
# THEME
base_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    panel.grid.major.y = element_line(colour = "grey85", linetype = "dashed"),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    legend.position    = "none",
    axis.text.x        = element_text(size = FONT_BASE,
                                      angle = 18, hjust = 1)
  )

# -----------------------------------------------------------------------------
# PER-METRIC RENDER
METRIC_CFG <- list(
  excess_return = list(
    out_name = "quantile_box_excess_return.pdf",
    y_label  = "Excess return, wallet minus market (%)"
  ),
  alpha = list(
      out_name = "quantile_box_alpha.pdf",
      y_label  = "CAPM alpha (%)"
  )
)

plot_metric <- function(df_box_all, m, cfg) {
  df_m <- df_box_all %>% filter(metric == m)
  
  p <- ggplot(df_m, aes(x = strategy_f)) +
    geom_hline(yintercept = 0, colour = "grey40",
               linetype = "dashed", linewidth = 0.4) +
    # Q01 -- Q99: thin outer line
    geom_linerange(aes(ymin = q01, ymax = q99, colour = strategy_f),
                   linewidth = 0.45, alpha = 0.7) +
    # Q05 -- Q95: thick inner whisker
    geom_linerange(aes(ymin = q05, ymax = q95, colour = strategy_f),
                   linewidth = 1.8, alpha = 0.85) +
    # Q25 -- Q75 box, with Q50 line
    geom_crossbar(aes(y = q50, ymin = q25, ymax = q75,
                      fill = strategy_f, colour = strategy_f),
                  width = 0.55, alpha = 0.4,
                  linewidth = 0.5, fatten = 2) +
    scale_colour_manual(values = STRATEGY_PAL, guide = "none") +
    scale_fill_manual(values   = STRATEGY_PAL, guide = "none") +
    scale_y_continuous(labels = label_number(accuracy = 1, suffix = "%")) +
    labs(x = NULL, y = cfg$y_label) +
    base_theme
  
  out_path <- file.path(opts$`out-dir`, cfg$out_name)
  ggsave(out_path, p, width = opts$width, height = opts$height)
  cat("[saved]", out_path, "\n")
}

for (m in names(METRIC_CFG)) {
  plot_metric(df_box, m, METRIC_CFG[[m]])
}
