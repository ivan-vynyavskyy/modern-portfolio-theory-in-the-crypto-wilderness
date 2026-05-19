#!/usr/bin/env Rscript
# =============================================================================
# plot_size_stats_by_wealth.R
#
# Description:
#   Stacked bar chart showing the distribution of portfolio sizes (number
#   of distinct assets held) across wealth categories. The "0" size bucket
#   is filtered out. Percentages are averaged across blocks per wealth
#   category.
#
# Input:
#   CSV with columns: wealth_bucket, size_bucket, wallet_pct
#   (default: ../data/portfolio_size_hist_by_wealth.csv).
#
# Output:
#   Single PDF (default: ../graphics/portfolio_size_stacked_avg.pdf).
#
# Usage:
#   RScript plot_size_stats_by_wealth.R \
#     --input ../data/portfolio_size_hist_by_wealth.csv \
#     --out_pdf ../graphics/portfolio_size_stacked_avg.pdf
# =============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(viridis)
  library(optparse)
})

# -----------------------------------------------------------------------------
# CLI OPTIONS
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../data/portfolio_size_hist_by_wealth.csv",
              help="Input CSV path [default %default]"),
  make_option(c("-o", "--out_pdf"), type="character", default="../graphics/portfolio_size_stacked_avg.pdf",
              help="Output PDF path [default %default]"),
  make_option(c("--width"), type="double", default=10, help="Plot width (in)"),
  make_option(c("--height"), type="double", default=4, help="Plot height (in)")
)

opt <- parse_args(OptionParser(option_list=option_list))

# -----------------------------------------------------------------------------
# LOAD & PROCESS
# -----------------------------------------------------------------------------
if (!file.exists(opt$input)) stop(paste("Input file not found:", opt$input))

df_hist <- read_csv(opt$input, show_col_types = FALSE)

wealth_levels <- c("0–1","1–100","100–1,000","1,000–10,000","10,000–100,000",">100,000")
wealth_labels <- c(
  "0–1"            = "$0–$1",
  "1–100"          = "$1–$100",
  "100–1,000"      = "$100–$1,000",
  "1,000–10,000"   = "$1,000–$10,000",
  "10,000–100,000" = "$10,000–$100,000",
  ">100,000"       = ">$100,000"
)

size_levels <- c("1","2","3","4","5","6–10","11–20","21–50","51–100",">100")

df_hist <- df_hist %>%
  mutate(
    wealth_bucket = factor(wealth_bucket, levels = wealth_levels),
    wealth_label  = factor(wealth_labels[as.character(wealth_bucket)],
                           levels = wealth_labels[wealth_levels]),
    size_bucket   = factor(size_bucket, levels = c("0", size_levels))
  ) %>%
  filter(size_bucket != "0") %>%
  mutate(size_bucket = factor(as.character(size_bucket), levels = size_levels))

df_hist_avg <- df_hist %>%
  group_by(wealth_label, size_bucket) %>%
  summarise(wallet_pct = mean(wallet_pct, na.rm = TRUE), .groups = "drop")

# -----------------------------------------------------------------------------
# THEME
# -----------------------------------------------------------------------------
script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grep("^--file=", args)]
  if (length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg)))
  } else {
    getwd()
  }
})()
source(file.path(script_dir, "..", "..", "..", "shared", "plot_theme.R"))
opt$width <- apply_plot_theme_fonts(opt$width)

fintech_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    axis.title    = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text     = element_text(size = FONT_AXIS_TEXT,  color = "grey30"),
    axis.line     = element_blank(),

    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "grey90", linetype = "dashed"),

    plot.background   = element_rect(fill = "white", color = NA),
    panel.background  = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA),

    legend.title = element_text(
      size = FONT_LEGEND_TITLE, color = "grey30", face = "bold"
    ),
    legend.text = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),
    legend.position = "right",

    plot.margin = margin(10, 10, 10, 10),
    axis.text.x = element_text(margin = margin(t = 6))
  )

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
p_stack_avg <- ggplot(df_hist_avg, aes(x = wealth_label, y = wallet_pct, fill = size_bucket)) +
  geom_col(width = 0.70) +

  scale_y_continuous(
    labels = function(x) paste0(round(x, 0), "%"),
    expand = expansion(mult = c(0.01, 0.05))
  ) +

  # Viridis option C (plasma) gives good contrast across the 10 levels
  scale_fill_viridis_d(option = "C", end = 0.95, direction = -1) +

  scale_x_discrete(guide = guide_axis(n.dodge = 2)) +

  labs(
    x = "Wealth category (USD)",
    y = "Avg % of accounts",
    fill = "Portfolio size\n(# assets)"
  ) +
  fintech_theme

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
out_file <- opt$out_pdf
if (!grepl("\\.pdf$", out_file, ignore.case = TRUE)) {
  out_file <- paste0(out_file, ".pdf")
}

dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

ggsave(
  filename = out_file,
  plot = p_stack_avg,
  width = opt$width,
  height = opt$height,
  device = "pdf"
)

cat(sprintf("[ok] Saved PDF: %s\n", out_file))