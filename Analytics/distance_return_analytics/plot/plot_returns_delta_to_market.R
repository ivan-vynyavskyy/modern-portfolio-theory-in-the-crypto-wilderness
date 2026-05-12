#!/usr/bin/env Rscript
# =============================================================================
# plot_returns_delta_to_market.R
#
# Description:
#   Delta-to-market plot: median return of each strategy minus the
#   market median, per monthly snapshot. If timing dominates, all
#   deltas cluster together while the level varies with the cycle.
#
# Input:
#   CSV with per-block return statistics.
#
# Output:
#   Single PDF (default: ../graphics/returns_delta_to_market.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(optparse)
  library(scales)
})

option_list <- list(
  make_option("--in-csv",    type = "character", default = "../data/camp_analysis_results_full/returns_per_block.csv"),
  make_option("--out-pdf",   type = "character", default = "../graphics/returns_delta_to_market.pdf"),
  make_option("--width",     type = "double", default = 10),
  make_option("--height",    type = "double", default = 5.5)
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
df <- read.csv(opts$`in-csv`)
df$block_date <- as.Date(df$block_date)

# Compute delta: strategy median return minus market median return
long <- df %>%
  mutate(
    delta_baseline      = (median_ret_baseline - median_market_return) * 100,
    delta_better_return  = (median_ret_better_return - median_market_return) * 100,
    delta_safer_risk     = (median_ret_safer_risk - median_market_return) * 100,
    delta_max_sharpe     = (median_ret_max_sharpe - median_market_return) * 100,
    delta_equal_weight   = (median_ret_equal_weight - median_market_return) * 100,
    delta_mcap_weight    = (median_ret_mcap_weight - median_market_return) * 100
  ) %>%
  select(block_date, starts_with("delta_")) %>%
  pivot_longer(
    cols      = -block_date,
    names_to  = "strategy",
    values_to = "delta_pct"
  ) %>%
  mutate(
    strategy = recode(strategy,
      "delta_baseline"       = "Baseline (actual)",
      "delta_better_return"  = "MaxRet",
      "delta_safer_risk"     = "MinVar",
      "delta_max_sharpe"     = "MaxSR",
      "delta_equal_weight"   = "Equal-weight",
      "delta_mcap_weight"    = "MCap-weight"
    ),
    strategy = factor(strategy, levels = c(
      "Baseline (actual)", "MaxRet", "MinVar",
      "MaxSR", "Equal-weight", "MCap-weight"
    ))
  )

pal <- c(
  "Baseline (actual)" = "#E31A1C",
  "MaxRet"            = "#1F78B4",
  "MinVar"            = "#33A02C",
  "MaxSR"             = "#FF7F00",
  "Equal-weight"      = "#6A3D9A",
  "MCap-weight"       = "#B15928"
)

lty <- c(
  "Baseline (actual)" = "solid",
  "MaxRet"            = "solid",
  "MinVar"            = "solid",
  "MaxSR"             = "solid",
  "Equal-weight"      = "dotted",
  "MCap-weight"       = "dotted"
)

shp <- c(
  "Baseline (actual)" = 16,
  "MaxRet"            = 17,
  "MinVar"            = 15,
  "MaxSR"             = 18,
  "Equal-weight"      = 1,
  "MCap-weight"       = 2
)

p <- ggplot(long, aes(x = block_date, y = delta_pct,
                       colour = strategy, linetype = strategy,
                       shape = strategy)) +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4, linetype = "dashed") +
  geom_line(linewidth = 0.55) +
  geom_point(size = 1.4, alpha = 0.8) +
  scale_colour_manual(values = pal) +
  scale_linetype_manual(values = lty) +
  scale_shape_manual(values = shp) +
  scale_x_date(
    date_breaks = "6 months",
    date_labels = "%b %Y",
    expand      = expansion(mult = 0.02)
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    x      = NULL,
    y      = "Median return minus market benchmark (%)",
    colour = NULL, linetype = NULL, shape = NULL
  ) +
  theme_minimal(base_size = FONT_BASE) +
  theme(
    panel.grid.major.y = element_line(colour = "grey85", linetype = "dashed"),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    legend.position    = "bottom",
    legend.text        = element_text(size = FONT_LEGEND_TEXT),
    axis.text.x        = element_text(angle = 45, hjust = 1)
  ) +
  guides(
    colour   = guide_legend(nrow = 1),
    linetype = guide_legend(nrow = 1),
    shape    = guide_legend(nrow = 1)
  )

ggsave(opts$`out-pdf`, p, width = opts$width, height = opts$height)
cat("[saved]", opts$`out-pdf`, "\n")