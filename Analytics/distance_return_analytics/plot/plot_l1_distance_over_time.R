#!/usr/bin/env Rscript
# =============================================================================
# plot_l1_distance_over_time.R
#
# 3×3 facet grid:
#   Columns = strategy (Min Variance, Max Return, Max Sharpe)
#   Rows    = token group (All wallets, Tokens < 5, Tokens ≥ 5)
#
# Each panel shows:
#   - Mean L1 distance (solid line)
#   - ±1 Std deviation (shaded ribbon)
#   - Median L1 distance (dashed line)
#
# Style: Fintech-minimal, PDF output, no titles (for clean LaTeX embedding).
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(lubridate)
})

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_IN_CSV  <- "../data/per_block_distance_stats.csv"
DEFAULT_OUT_PDF <- "../graphics/l1_distance_over_time.pdf"

STRATEGY_COLORS <- c(
  "safer_risk"    = "#33A02C",
  "better_return" = "#1F78B4",
  "max_sharpe"    = "#E31A1C"
)

STRATEGY_LABELS <- c(
  "safer_risk"    = "Min Variance (Same Ret)",
  "better_return" = "Max Return (Same Vol)",
  "max_sharpe"    = "Max Sharpe Ratio"
)

GROUP_LABELS <- c(
  "all"  = "All wallets",
  "lt5"  = "Tokens < 5",
  "gte5" = "Tokens >= 5"
)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("-i", "--in-csv"), type = "character", default = DEFAULT_IN_CSV,
              help = "Input CSV path [default: %default]"),
  make_option(c("-o", "--out-pdf"), type = "character", default = DEFAULT_OUT_PDF,
              help = "Output PDF path [default: %default]"),
  make_option(c("--width"),  type = "double", default = 12,
              help = "Plot width in inches [default: %default]"),
  make_option(c("--height"), type = "double", default = 8,
              help = "Plot height in inches [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))
if (!file.exists(opt$`in-csv`)) stop("Input file not found: ", opt$`in-csv`)


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
# LOAD & PROCESS
# -----------------------------------------------------------------------------
cat("[info] Loading data...\n")
df <- read_csv(opt$`in-csv`, show_col_types = FALSE)

df <- df %>%
  mutate(
    date         = as.Date(date),
    optimisation = as.character(optimisation),
    token_group  = as.character(token_group),
    mean_dist    = as.numeric(mean_dist),
    median_dist  = as.numeric(median_dist),
    std_dist     = as.numeric(std_dist)
  ) %>%
  filter(!is.na(date), !is.na(mean_dist))

# Ribbon bounds (clamp to 0-100)
df <- df %>%
  mutate(
    ymin = pmax(mean_dist - std_dist, 0),
    ymax = pmin(mean_dist + std_dist, 100)
  )

# Factors with nice labels
df$optimisation <- factor(df$optimisation,
                          levels = names(STRATEGY_LABELS),
                          labels = STRATEGY_LABELS)

df$token_group <- factor(GROUP_LABELS[df$token_group],
                         levels = GROUP_LABELS)

cat(sprintf("[info] %d rows across %d blocks, %d strategies, %d groups\n",
            nrow(df), n_distinct(df$date),
            n_distinct(df$optimisation), n_distinct(df$token_group)))

# -----------------------------------------------------------------------------
# THEME
# -----------------------------------------------------------------------------
fintech_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    plot.title    = element_blank(),
    plot.subtitle = element_blank(),

    axis.title   = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text    = element_text(color = "grey30"),

    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "grey90", linetype = "dashed"),

    strip.text       = element_text(face = "bold", size = FONT_STRIP, color = "grey20"),
    strip.background = element_blank(),

    legend.position  = "bottom",
    legend.title     = element_blank(),
    legend.text      = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),
    legend.key.width = unit(1.8, "cm"),

    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),

    plot.margin = margin(10, 15, 10, 10)
  )

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
cat("[info] Building plot...\n")

# Map nice strategy labels to colours
ribbon_cols <- setNames(
  unname(STRATEGY_COLORS[names(STRATEGY_LABELS)]),
  STRATEGY_LABELS
)

p <- ggplot(df, aes(x = date)) +

  # ±1 Std ribbon
  geom_ribbon(
    aes(ymin = ymin, ymax = ymax, fill = optimisation),
    alpha = 0.18
  ) +

  # Median (dashed)
  geom_line(
    aes(y = median_dist, color = optimisation, linetype = "Median"),
    linewidth = 0.55
  ) +

  # Mean (solid)
  geom_line(
    aes(y = mean_dist, color = optimisation, linetype = "Mean"),
    linewidth = 0.9
  ) +

  # 3×3 grid: rows = token group, columns = strategy
  facet_grid(token_group ~ optimisation) +

  # Scales
  scale_color_manual(values = ribbon_cols, guide = "none") +
  scale_fill_manual(values = ribbon_cols, guide = "none") +

  scale_linetype_manual(
    name   = NULL,
    values = c("Mean" = "solid", "Median" = "dashed"),
    guide  = guide_legend(override.aes = list(
      linewidth = c(0.9, 0.55),
      color     = c("grey20", "grey40")
    ))
  ) +

  scale_x_date(
    breaks = seq.Date(as.Date("2020-01-01"), as.Date("2025-01-01"), by = "1 year"),
    date_labels = "'%y",
    expand = expansion(mult = c(0.01, 0.03))
  ) +

  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 80, by = 20),
    labels = function(x) paste0(round(x, 0), "%"),
    expand = expansion(mult = c(0, 0.02))
  ) +

  labs(
    x = NULL,
    y = "L1 distance (%)"
  ) +

  fintech_theme

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
out_file <- opt$`out-pdf`
if (!grepl("\\.pdf$", out_file, ignore.case = TRUE)) {
  out_file <- paste0(out_file, ".pdf")
}
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

ggsave(out_file, plot = p, width = opt$width, height = opt$height, device = "pdf")
cat(sprintf("[ok] Saved PDF: %s\n", out_file))