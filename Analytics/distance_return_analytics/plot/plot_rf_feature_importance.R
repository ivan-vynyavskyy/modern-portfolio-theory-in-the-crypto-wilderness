#!/usr/bin/env Rscript
# =============================================================================
# plot_rf_feature_importance.R
#
# Description:
#   Grouped horizontal bar chart of Random Forest feature importances
#   for the three MPT optimisation strategies. Data is embedded from
#   SLURM output (no input file needed).
#
# Output:
#   Single PDF (default: ../graphics/rf_feature_importance.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_OUT_PDF <- "../graphics/rf_feature_importance.pdf"

# Strategy colours (Brewer Paired subset)
STRATEGY_COLORS <- c(
  "Max Return (Same Vol)"  = "#1F78B4",
  "Max Sharpe Ratio"       = "#E31A1C",
  "Min Variance (Same Ret)" = "#33A02C"
)

# Embedded data from SLURM output
IMPORTANCE_DATA <- tribble(
  ~strategy,                  ~feature,          ~importance,
  "Max Return (Same Vol)",    "log_value_usd",   0.4492,
  "Max Return (Same Vol)",    "month",           0.2813,
  "Max Return (Same Vol)",    "num_tokens",      0.2695,
  "Min Variance (Same Ret)",  "num_tokens",      0.9547,
  "Min Variance (Same Ret)",  "log_value_usd",   0.0282,
  "Min Variance (Same Ret)",  "month",           0.0170,
  "Max Sharpe Ratio",         "log_value_usd",   0.4446,
  "Max Sharpe Ratio",         "num_tokens",      0.3080,
  "Max Sharpe Ratio",         "month",           0.2474
)

# Human-readable feature labels
FEATURE_LABELS <- c(
  "num_tokens"    = "Token count",
  "log_value_usd" = "Log portfolio value",
  "month"         = "Calendar month"
)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("-o", "--out-pdf"), type = "character", default = DEFAULT_OUT_PDF,
              help = "Output PDF path [default: %default]"),
  make_option(c("--width"),  type = "double", default = 10,
              help = "Plot width in inches [default: %default]"),
  make_option(c("--height"), type = "double", default = 4.5,
              help = "Plot height in inches [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))


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
opt$width <- apply_plot_theme_fonts(opt$width)
# -----------------------------------------------------------------------------
# DATA PROCESSING
# -----------------------------------------------------------------------------
cat("[info] Preparing data...\n")

df <- IMPORTANCE_DATA %>%
  mutate(
    feature_label = factor(FEATURE_LABELS[feature],
                           levels = rev(FEATURE_LABELS)),
    strategy      = factor(strategy, levels = names(STRATEGY_COLORS)),
    pct           = importance * 100
  )

# -----------------------------------------------------------------------------
# THEME
# -----------------------------------------------------------------------------
fintech_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    plot.title    = element_blank(),
    plot.subtitle = element_blank(),

    axis.title   = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text    = element_text(color = "grey30", size = FONT_AXIS_TEXT),

    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_line(color = "grey92", linetype = "dashed"),

    legend.position  = "bottom",
    legend.title     = element_blank(),
    legend.text      = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),

    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),

    plot.margin = margin(10, 15, 10, 10)
  )

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
cat("[info] Building plot...\n")

p <- ggplot(df, aes(x = pct, y = feature_label, fill = strategy)) +

  geom_col(
    position = position_dodge(width = 0.75),
    width    = 0.68
  ) +

  # Percentage labels
  geom_text(
    aes(label = paste0(sprintf("%.1f", pct), "%")),
    position = position_dodge(width = 0.75),
    hjust    = -0.12,
    size     = 3.0,
    color    = "grey30"
  ) +

  scale_fill_manual(values = STRATEGY_COLORS) +

  scale_x_continuous(
    labels = function(x) paste0(round(x, 0), "%"),
    limits = c(0, 108),
    breaks = seq(0, 100, by = 25),
    expand = expansion(mult = c(0, 0))
  ) +

  labs(
    x = "Feature importance (%)",
    y = NULL
  ) +

  fintech_theme

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
out_file <- opt$`out-pdf`
if (!grepl("\\.pdf$", out_file, ignore.case = TRUE)) out_file <- paste0(out_file, ".pdf")
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

ggsave(out_file, plot = p, width = opt$width, height = opt$height, device = "pdf")
cat(sprintf("[ok] Saved PDF: %s\n", out_file))