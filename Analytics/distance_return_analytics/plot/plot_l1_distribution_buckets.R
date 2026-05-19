#!/usr/bin/env Rscript
# =============================================================================
# plot_l1_distribution_buckets.R
#
# Description:
#   Two publication-ready plots for L1 distance distribution across
#   strategies:
#     A - Grouped bar chart (mean L1 by strategy and bucket).
#     B - Tile heatmap with percentage labels.
#
# Input:
#   CSV with columns: strategy, l1_bucket, count, pct (or similar
#   aggregation).
#
# Output:
#   Two PDFs (default: ../graphics/l1_distribution_grouped.pdf and
#   ../graphics/l1_distribution_heatmap.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_OUT_A <- "../graphics/l1_distribution_grouped.pdf"
DEFAULT_OUT_B <- "../graphics/l1_distribution_heatmap.pdf"

# Strategy colours (Brewer Paired subset)
STRATEGY_COLORS <- c(
  "Max Return\n(Same Vol)"    = "#1F78B4",
  "Max Sharpe\nRatio"         = "#E31A1C",
  "Min Variance\n(Same Ret)"  = "#33A02C"
)

# Embedded data — from SLURM output
RAW_DATA <- tribble(
  ~bucket,    ~better_return, ~max_sharpe, ~safer_risk,
  "0",                  0.57,       0.80,       1.32,
  "(0,1]",             39.52,      17.07,      62.03,
  "(1,20]",            14.98,      13.41,      11.24,
  "(20,40]",           12.02,      12.11,       9.74,
  "(40,60]",           10.61,      12.46,       8.60,
  "(60,80]",           11.16,      24.66,       5.06,
  "(80,100]",          11.15,      19.49,       2.00
)

# Readable bucket labels
BUCKET_LABELS <- c(
  "0"        = "0",
  "(0,1]"    = "(0, 1]",
  "(1,20]"   = "(1, 20]",
  "(20,40]"  = "(20, 40]",
  "(40,60]"  = "(40, 60]",
  "(60,80]"  = "(60, 80]",
  "(80,100]" = "(80, 100]"
)

# Smart % formatter: 1 decimal for sub-1% values (so 0.57 stays "0.6%"
# instead of rounding to "1%"), integer for everything else.
format_pct_label <- function(x) {
  ifelse(x < 1,
         paste0(sprintf("%.1f", x), "%"),
         paste0(sprintf("%.0f", x), "%"))
}

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("--out-a"), type = "character", default = DEFAULT_OUT_A,
              help = "Output PDF for variant A (grouped bars) [default: %default]"),
  make_option(c("--out-b"), type = "character", default = DEFAULT_OUT_B,
              help = "Output PDF for variant B (heatmap) [default: %default]"),
  make_option(c("--width"),  type = "double", default = 10,
              help = "Plot width in inches [default: %default]"),
  make_option(c("--height"), type = "double", default = 3.8,
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

df_long <- RAW_DATA %>%
  pivot_longer(
    cols      = c(better_return, max_sharpe, safer_risk),
    names_to  = "strategy",
    values_to = "pct"
  ) %>%
  mutate(
    bucket_label = factor(BUCKET_LABELS[bucket], levels = BUCKET_LABELS),
    strategy_label = factor(
      case_when(
        strategy == "better_return" ~ "Max Return\n(Same Vol)",
        strategy == "max_sharpe"    ~ "Max Sharpe\nRatio",
        strategy == "safer_risk"    ~ "Min Variance\n(Same Ret)"
      ),
      levels = names(STRATEGY_COLORS)
    )
  )

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
    panel.grid.major.y = element_line(color = "grey92", linetype = "dashed"),

    legend.position  = "bottom",
    legend.title     = element_blank(),
    legend.text      = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),

    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),

    plot.margin = margin(10, 15, 10, 10)
  )

# =============================================================================
# VARIANT A — Grouped Bar Chart
# =============================================================================
cat("[info] Building variant A (grouped bars)...\n")

p_a <- ggplot(df_long, aes(x = bucket_label, y = pct, fill = strategy_label)) +

  geom_col(
    position = position_dodge(width = 0.78),
    width    = 0.72
  ) +

  # Percentage labels on every bar (smart format keeps sub-1% values
  # accurate without bloating the major-bar labels).
  geom_text(
    aes(label = format_pct_label(pct)),
    position = position_dodge(width = 0.78),
    vjust    = -0.4,
    size     = 2.8,
    color    = "grey30"
  ) +

  scale_fill_manual(values = STRATEGY_COLORS) +

  scale_y_continuous(
    labels = function(x) paste0(round(x, 0), "%"),
    expand = expansion(mult = c(0, 0.12))
  ) +

  labs(
    x = "L1 distance bucket (%)",
    y = "Share of accounts (%)",
  ) +

  fintech_theme

# Save A
out_a <- opt$`out-a`
if (!grepl("\\.pdf$", out_a, ignore.case = TRUE)) out_a <- paste0(out_a, ".pdf")
dir.create(dirname(out_a), showWarnings = FALSE, recursive = TRUE)
ggsave(out_a, plot = p_a, width = opt$width, height = opt$height, device = "pdf")
cat(sprintf("[ok] Variant A saved: %s\n", out_a))

# =============================================================================
# VARIANT B — Tile Heatmap
# =============================================================================
cat("[info] Building variant B (heatmap)...\n")

# Single-line labels for heatmap y-axis
df_heat <- RAW_DATA %>%
  pivot_longer(
    cols      = c(better_return, max_sharpe, safer_risk),
    names_to  = "strategy",
    values_to = "pct"
  ) %>%
  mutate(
    bucket_label = factor(BUCKET_LABELS[bucket], levels = BUCKET_LABELS),
    strategy_label = factor(
      case_when(
        strategy == "better_return" ~ "Max Return (Same Vol)",
        strategy == "max_sharpe"    ~ "Max Sharpe Ratio",
        strategy == "safer_risk"    ~ "Min Variance (Same Ret)"
      ),
      levels = c("Min Variance (Same Ret)", "Max Return (Same Vol)", "Max Sharpe Ratio")
    )
  )

# Colour for text: dark on light tiles, white on dark tiles
df_heat <- df_heat %>%
  mutate(text_color = ifelse(pct > 30, "white", "grey20"))

p_b <- ggplot(df_heat, aes(x = bucket_label, y = strategy_label)) +

  geom_tile(
    aes(fill = pct),
    color = "white", linewidth = 1.2
  ) +

  # Percentage labels inside tiles
  geom_text(
    aes(label = paste0(sprintf("%.1f", pct), "%"), color = text_color),
    size = 4, fontface = "bold"
  ) +
  scale_color_identity() +

  # Sequential blue scale (matches the analytical tone)
  scale_fill_gradient(
    low  = "#EFF3FF",
    high = "#2171B5",
    name = "% of wallets",
    labels = function(x) paste0(round(x, 0), "%")
  ) +

  labs(x = "L1 distance bucket (%)", y = NULL) +

  theme_minimal(base_size = FONT_BASE) +
  theme(
    plot.title    = element_blank(),
    plot.subtitle = element_blank(),

    axis.title.x = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text    = element_text(color = "grey30", size = FONT_AXIS_TEXT),
    axis.text.y  = element_text(face = "bold"),

    panel.grid   = element_blank(),

    legend.position  = "right",
    legend.title     = element_text(size = FONT_LEGEND_TITLE, color = "grey30", face = "bold"),
    legend.text      = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),

    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),

    plot.margin = margin(10, 10, 10, 10)
  )

# Save B
out_b <- opt$`out-b`
if (!grepl("\\.pdf$", out_b, ignore.case = TRUE)) out_b <- paste0(out_b, ".pdf")
dir.create(dirname(out_b), showWarnings = FALSE, recursive = TRUE)
ggsave(out_b, plot = p_b, width = opt$width, height = 4, device = "pdf")
cat(sprintf("[ok] Variant B saved: %s\n", out_b))