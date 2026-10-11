#!/usr/bin/env Rscript
# =============================================================================
# plot_mpt_vs_naive.R
#
# Description:
#   Two-panel figure comparing MPT-optimised vs naive portfolios:
#     Left  - Stacked horizontal bar (closer / farther / unchanged %).
#     Right - Lollipop chart of the mean change in distance, in pp of the
#             0-100 distance scale (raw L1 delta * 50; green = closer,
#             orange = farther).
#
# Input:
#   mpt_vs_naive_direction_overall.csv of analyze_extended_capm.py
#   (mpt_strategy, naive_strategy, mean_delta_l1, pct_moved_closer,
#   pct_moved_farther, pct_unchanged).
#
# Output:
#   Single PDF (default: ../graphics/mpt_vs_naive.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
})

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("-i", "--in-csv"), type = "character",
              default = "../data/camp_analysis_results_full/mpt_vs_naive_direction_overall.csv",
              help = "Input CSV [default: %default]"),
  make_option(c("-o", "--out-pdf"), type = "character",
              default = "../graphics/mpt_vs_naive.pdf",
              help = "Output PDF path [default: %default]"),
  make_option(c("--width"),  type = "double", default = 12,
              help = "Plot width in inches [default: %default]"),
  make_option(c("--height"), type = "double", default = 5,
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
# DATA (overall direction table, same numbers as the paper's MPT-vs-naive table, tab:mpt_vs_naive)
# -----------------------------------------------------------------------------
MPT_LABELS   <- c(better_return = "MaxRet", safer_risk = "MinVar", max_sharpe = "MaxSR")
NAIVE_LABELS <- c(equal_weight = "Equal-wt", mcap_weight = "MCap-wt")
raw <- read.csv(opt$`in-csv`, stringsAsFactors = FALSE)
df <- data.frame(
  mpt       = unname(MPT_LABELS[raw$mpt_strategy]),
  naive     = unname(NAIVE_LABELS[raw$naive_strategy]),
  closer    = raw$pct_moved_closer,
  farther   = raw$pct_moved_farther,
  unchanged = raw$pct_unchanged,
  # raw L1 (0-2) delta -> pp of the 0-100 distance scale d = L1 / 2 * 100
  mean_dl1  = raw$mean_delta_l1 * 50,
  stringsAsFactors = FALSE
)
# Equal-weight block first, MaxRet / MinVar / MaxSR within each block
df <- df[order(match(df$naive, NAIVE_LABELS), match(df$mpt, MPT_LABELS)), ]

# Row label: "MaxRet · Equal-wt" etc
df$label <- paste0(df$mpt, " \u00b7 ", df$naive)

# Preserve order (top to bottom matches the image)
row_order <- rev(df$label)
df$label <- factor(df$label, levels = row_order)

# Direction colour for lollipop
df$direction <- ifelse(df$mean_dl1 <= 0, "Improves", "Worsens")

# -----------------------------------------------------------------------------
# THEME
# -----------------------------------------------------------------------------
fintech_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    plot.title       = element_text(size = 11, face = "bold", color = "grey25",
                                    margin = margin(b = 8)),
    axis.title       = element_blank(),
    axis.text.y      = element_text(size = FONT_AXIS_TEXT, color = "grey30"),
    axis.text.x      = element_text(size = FONT_AXIS_TEXT, color = "grey40"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_line(color = "grey92", linetype = "dashed"),
    legend.position  = "bottom",
    legend.title     = element_blank(),
    legend.text      = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin      = margin(10, 10, 10, 10)
  )

# Colours
BAR_COLORS <- c(
  "Closer"    = "#33A02C",
  "Farther"   = "#E25822",
  "Unchanged" = "#CCCCCC"
)

DOT_COLORS <- c(
  "Improves" = "#33A02C",
  "Worsens"  = "#E25822"
)

# -----------------------------------------------------------------------------
# LEFT PANEL: Stacked horizontal bars
# -----------------------------------------------------------------------------
bar_df <- df %>%
  select(label, Closer = closer, Farther = farther, Unchanged = unchanged) %>%
  pivot_longer(cols = c("Closer", "Farther", "Unchanged"),
               names_to = "category", values_to = "pct")

bar_df$category <- factor(bar_df$category, levels = c("Closer", "Farther", "Unchanged"))

p_left <- ggplot(bar_df, aes(x = pct, y = label, fill = category)) +
  geom_bar(stat = "identity", position = "stack", width = 0.65) +
  scale_fill_manual(values = BAR_COLORS) +
  scale_x_continuous(
    limits = c(0, 100),
    oob    = scales::squish,
    breaks = seq(0, 100, by = 20),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(title = "Period distribution (%)") +
  fintech_theme +
  theme(
    legend.position = "bottom",
    legend.key.size = unit(0.4, "cm")
  )

# -----------------------------------------------------------------------------
# RIGHT PANEL: Lollipop chart
# -----------------------------------------------------------------------------

# Separator line position (between equal-wt and mcap-wt groups)
sep_y <- 3.5

# Symmetric x axis with room for the value labels
x_lim <- max(abs(df$mean_dl1)) * 1.6

p_right <- ggplot(df, aes(x = mean_dl1, y = label)) +

  # Dashed separator between benchmark groups
  geom_hline(yintercept = sep_y, linetype = "dashed", color = "grey75",
             linewidth = 0.4) +

  # Zero line
  geom_vline(xintercept = 0, color = "grey60", linewidth = 0.4) +

  # Lollipop stems
  geom_segment(
    aes(x = 0, xend = mean_dl1, y = label, yend = label, color = direction),
    linewidth = 0.8
  ) +

  # Dots
  geom_point(aes(color = direction), size = 3.5) +

  # Value labels
  geom_text(
    aes(label = sprintf("%+.2f", mean_dl1),
        hjust = ifelse(mean_dl1 >= 0, -0.3, 1.3)),
    size = 3.2, color = "grey30", fontface = "bold"
  ) +

  scale_color_manual(values = DOT_COLORS) +

  scale_x_continuous(
    limits = c(-1, 1) * x_lim,
    breaks = scales::breaks_pretty(n = 5),
    labels = function(x) ifelse(x == 0, "0", sprintf("%+.0f", x)),
    expand = expansion(mult = c(0.05, 0.05))
  ) +

  labs(title = "Mean change in distance (pp)") +

  fintech_theme +
  theme(
    axis.text.y  = element_blank(),
    legend.position = "bottom",
    legend.key.size = unit(0.4, "cm")
  )

# -----------------------------------------------------------------------------
# COMBINE & SAVE
# -----------------------------------------------------------------------------
p <- p_left + p_right +
  plot_layout(widths = c(1, 1))

out_file <- opt$`out-pdf`
if (!grepl("\\.pdf$", out_file, ignore.case = TRUE)) out_file <- paste0(out_file, ".pdf")
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

ggsave(out_file, plot = p, width = opt$width, height = opt$height, device = "pdf")
cat(sprintf("[ok] Saved PDF: %s\n", out_file))