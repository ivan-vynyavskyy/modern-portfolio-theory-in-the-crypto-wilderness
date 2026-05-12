#!/usr/bin/env Rscript
# =============================================================================
# plot_median_alpha_per_block.R
#
# Description:
#   Per-block median alpha for each strategy, as a line/point chart
#   with distinct shapes and linetypes per strategy.
#
# Input:
#   CSV with per-block alpha statistics.
#
# Output:
#   Single PDF (default: ../graphics/median_alpha_per_block.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(optparse)
  library(scales)
})

# -----------------------------------------------------------------------------
# CLI
option_list <- list(
  make_option("--in-csv",   type = "character", default = "../data/camp_analysis_results_full/alpha_per_block.csv",
              help = "Input CSV with per-block alpha statistics"),
  make_option("--out-pdf",  type = "character", default = "../graphics/median_alpha_per_block.pdf",
              help = "Output PDF path"),
  make_option("--width",    type = "double", default = 10, help = "Plot width in inches"),
  make_option("--height",   type = "double", default = 5.5, help = "Plot height in inches")
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
# LOAD & RESHAPE
df <- read.csv(opts$`in-csv`)
df$block_date <- as.Date(df$block_date)

long <- df %>%
  select(block_date,
         median_alpha_baseline,
         median_alpha_better_return,
         median_alpha_safer_risk,
         median_alpha_max_sharpe,
         median_alpha_equal_weight,
         median_alpha_mcap_weight) %>%
  pivot_longer(
    cols      = -block_date,
    names_to  = "strategy",
    values_to = "median_alpha"
  ) %>%
  mutate(
    median_alpha_pct = median_alpha * 100,
    strategy = recode(strategy,
      "median_alpha_baseline"       = "Baseline (actual)",
      "median_alpha_better_return"  = "MaxRet",
      "median_alpha_safer_risk"     = "MinVar",
      "median_alpha_max_sharpe"     = "MaxSR",
      "median_alpha_equal_weight"   = "Equal-weight",
      "median_alpha_mcap_weight"    = "MCap-weight"
    ),
    strategy = factor(strategy, levels = c(
      "Baseline (actual)",
      "MaxRet",
      "MinVar",
      "MaxSR",
      "Equal-weight",
      "MCap-weight"
    ))
  )

# -----------------------------------------------------------------------------
# VISUAL MAPPINGS
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

lwd <- c(
  "Baseline (actual)" = 0.85,
  "MaxRet"            = 0.55,
  "MinVar"            = 0.55,
  "MaxSR"             = 0.55,
  "Equal-weight"      = 0.65,
  "MCap-weight"       = 0.65
)

shp <- c(
  "Baseline (actual)" = 16,  # filled circle
  "MaxRet"            = 17,  # filled triangle
  "MinVar"            = 15,  # filled square
  "MaxSR"             = 18,  # filled diamond
  "Equal-weight"      = 1,   # open circle
  "MCap-weight"       = 2    # open triangle
)

# -----------------------------------------------------------------------------
# PLOT
p <- ggplot(long, aes(x = block_date, y = median_alpha_pct,
                       colour = strategy, linetype = strategy,
                       shape = strategy, linewidth = strategy)) +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4) +
  geom_line() +
  geom_point(size = 1.4, alpha = 0.8) +
  scale_colour_manual(values = pal) +
  scale_linetype_manual(values = lty) +
  scale_linewidth_manual(values = lwd) +
  scale_shape_manual(values = shp) +
  scale_x_date(
    date_breaks = "6 months",
    date_labels = "%b %Y",
    expand      = expansion(mult = 0.02)
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    x      = NULL,
    y      = expression(paste("Median 20-day CAPM ", alpha, " (%)")),
    colour = NULL, linetype = NULL, shape = NULL, linewidth = NULL
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
    colour    = guide_legend(nrow = 1),
    linetype  = guide_legend(nrow = 1),
    shape     = guide_legend(nrow = 1),
    linewidth = guide_legend(nrow = 1)
  )

# -----------------------------------------------------------------------------
# SAVE
ggsave(opts$`out-pdf`, p, width = opts$width, height = opts$height)
cat("[saved]", opts$`out-pdf`, "\n")
