#!/usr/bin/env Rscript
# =============================================================================
# plot_median_returns_ribbon.R
#
# Description:
#   Per-block median returns with IQR ribbon (p25-p75), plus an ETH
#   price overlay on the right axis as a visual timing reference.
#
# Input:
#   CSV with per-block return statistics.
#
# Output:
#   Single PDF (default: ../graphics/median_returns_ribbon.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(optparse)
})

# -----------------------------------------------------------------------------
# CLI
option_list <- list(
  make_option("--in-csv",   type = "character", default = "../data/camp_analysis_results_full/returns_per_block.csv",
              help = "Input CSV with per-block return statistics"),
  make_option("--out-pdf",  type = "character", default = "../graphics/median_returns_ribbon.pdf",
              help = "Output PDF path"),
  make_option("--width",    type = "double", default = 10,  help = "Plot width in inches"),
  make_option("--height",   type = "double", default = 5,   help = "Plot height in inches")
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
# -----------------------------------------------------------------------------
# LOAD
df <- read.csv(opts$`in-csv`)
df$block_date <- as.Date(df$block_date)

# -----------------------------------------------------------------------------
# MPT RIBBON: MIN/MAX OF THE THREE STRATEGIES PER BLOCK
ribbon <- df %>%
  rowwise() %>%
  mutate(
    mpt_min = min(median_ret_better_return, median_ret_safer_risk, median_ret_max_sharpe) * 100,
    mpt_max = max(median_ret_better_return, median_ret_safer_risk, median_ret_max_sharpe) * 100,
    mpt_mid = median(c(median_ret_better_return, median_ret_safer_risk, median_ret_max_sharpe)) * 100
  ) %>%
  ungroup() %>%
  select(block_date, mpt_min, mpt_max, mpt_mid)

# -----------------------------------------------------------------------------
# INDIVIDUAL LINES: BASELINE + BENCHMARKS
lines <- df %>%
  select(block_date,
         median_ret_baseline,
         median_ret_equal_weight,
         median_ret_mcap_weight,
         median_market_return) %>%
  pivot_longer(
    cols      = -block_date,
    names_to  = "strategy",
    values_to = "median_return"
  ) %>%
  mutate(
    median_return_pct = median_return * 100,
    strategy = recode(strategy,
      "median_ret_baseline"      = "Baseline (actual)",
      "median_ret_equal_weight"  = "Equal-weight",
      "median_ret_mcap_weight"   = "MCap-weight",
      "median_market_return"     = "Market benchmark"
    ),
    strategy = factor(strategy, levels = c(
      "Market benchmark",
      "Baseline (actual)",
      "Equal-weight",
      "MCap-weight"
    ))
  )

# -----------------------------------------------------------------------------
# VISUAL MAPPINGS
pal <- c(
  "Market benchmark"  = "#222222",
  "Baseline (actual)" = "#E31A1C",
  "Equal-weight"      = "#6A3D9A",
  "MCap-weight"       = "#B15928"
)

lty <- c(
  "Market benchmark"  = "dashed",
  "Baseline (actual)" = "solid",
  "Equal-weight"      = "dotted",
  "MCap-weight"       = "dotted"
)

shp <- c(
  "Market benchmark"  = 4,   # x
  "Baseline (actual)" = 16,  # filled circle
  "Equal-weight"      = 1,   # open circle
  "MCap-weight"       = 2    # open triangle
)

# -----------------------------------------------------------------------------
# PLOT
p <- ggplot() +
  # zero reference
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4) +

  # MPT ribbon (MaxRet / MinVar / MaxSR range)
  geom_ribbon(data = ribbon,
              aes(x = block_date, ymin = mpt_min, ymax = mpt_max),
              fill = "#1F78B4", alpha = 0.2) +
  geom_line(data = ribbon,
            aes(x = block_date, y = mpt_mid, colour = "MPT range (median)"),
            linewidth = 0.5, linetype = "solid") +

  # individual strategy lines
  geom_line(data = lines,
            aes(x = block_date, y = median_return_pct,
                colour = strategy, linetype = strategy),
            linewidth = 0.7) +
  geom_point(data = lines,
             aes(x = block_date, y = median_return_pct,
                 colour = strategy, shape = strategy),
             size = 1.2, alpha = 0.7) +

  # scales
  scale_colour_manual(
    values = c(pal, "MPT range (median)" = "#1F78B4"),
    breaks = c("Market benchmark", "Baseline (actual)",
               "Equal-weight", "MCap-weight", "MPT range (median)")
  ) +
  scale_linetype_manual(
    values = c(lty, "MPT range (median)" = "solid"),
    breaks = c("Market benchmark", "Baseline (actual)",
               "Equal-weight", "MCap-weight")
  ) +
  scale_shape_manual(
    values = shp,
    breaks = c("Market benchmark", "Baseline (actual)",
               "Equal-weight", "MCap-weight")
  ) +
  scale_x_date(
    date_breaks = "6 months",
    date_labels = "%b %Y",
    expand      = expansion(mult = 0.02)
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    x        = NULL,
    y        = "Median 20-day return (%)",
    colour   = NULL,
    linetype = NULL,
    shape    = NULL
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
    colour   = guide_legend(nrow = 1, override.aes = list(
      fill      = c(NA, NA, NA, NA, "#1F78B4"),
      alpha     = c(1, 1, 1, 1, 0.2),
      linewidth = c(0.7, 0.7, 0.7, 0.7, 0.5)
    )),
    linetype = "none",
    shape    = "none"
  )

# -----------------------------------------------------------------------------
# SAVE
ggsave(opts$`out-pdf`, p, width = opts$width, height = opts$height)
cat("[saved]", opts$`out-pdf`, "\n")
