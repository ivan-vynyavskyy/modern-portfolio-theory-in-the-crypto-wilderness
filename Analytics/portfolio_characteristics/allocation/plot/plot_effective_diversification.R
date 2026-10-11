#!/usr/bin/env Rscript
# =============================================================================
# plot_effective_diversification.R
#
# Description:
#   Two stacked bar charts of multi-asset portfolios by wealth category,
#   averaged over the monthly snapshots: (a) the largest portfolio weight,
#   (b) the effective number of assets (1 / HHI of the weights). Same layout
#   and colours as the portfolio-size figure (plot_size_stats_by_wealth.R):
#   the most concentrated category is yellow and on top.
#
# Input:
#   ../data/max_weight_hist.csv and ../data/effective_n_hist.csv
#   (weight_concentration.py); rows with wallet_type = ALL and
#   n_bucket = 2+ (all multi-asset accounts) are used.
#
# Output:
#   ../graphics/effective_diversification_by_wealth.pdf
#
# Usage:
#   Rscript plot_effective_diversification.R
#   Rscript plot_effective_diversification.R --out_pdf ../graphics/x.pdf --width 10 --height 7
# =============================================================================

if (!grepl("UTF-8", Sys.getlocale("LC_CTYPE"), ignore.case = TRUE)) {
  invisible(try(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"), silent = TRUE))
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
  library(viridis)
  library(patchwork)
  library(optparse)
})

option_list <- list(
  make_option(c("--max_weight"), type = "character", default = "../data/max_weight_hist.csv",
              help = "Largest-weight histogram CSV [default %default]"),
  make_option(c("--effective_n"), type = "character", default = "../data/effective_n_hist.csv",
              help = "Effective-N histogram CSV [default %default]"),
  make_option(c("-o", "--out_pdf"), type = "character",
              default = "../graphics/effective_diversification_by_wealth.pdf",
              help = "Output PDF [default %default]"),
  make_option(c("--width"), type = "double", default = 10, help = "Plot width (in)"),
  make_option(c("--height"), type = "double", default = 7, help = "Plot height (in)")
)
opt <- parse_args(OptionParser(option_list = option_list))

script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grep("^--file=", args)]
  if (length(file_arg) > 0) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
})()
source(file.path(script_dir, "..", "..", "..", "shared", "plot_theme.R"))
opt$width <- apply_plot_theme_fonts(opt$width)

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
wealth_levels <- c("0–1", "1–100", "100–1K", "1K–10K", "10K–100K", ">100K")
wealth_labels <- c("$0–$1", "$1–$100", "$100–$1K", "$1K–$10K", "$10K–$100K", ">$100K")

# Most concentrated first: it gets the first viridis colour (yellow) and sits on top.
maxw_bins <- c("(0.99,1]", "(0.95,0.99]", "(0.9,0.95]", "[0.8,0.9]", "[0.7,0.8)",
               "[0.6,0.7)", "[0.5,0.6)", "[0.25,0.5)", "[0,0.25)")
maxw_labels <- c("> 0.99", "0.95–0.99", "0.90–0.95", "0.80–0.90", "0.70–0.80",
                 "0.60–0.70", "0.50–0.60", "0.25–0.50", "< 0.25")
effn_bins <- c("[1,1.1)", "[1.1,1.5)", "[1.5,2)", "[2,3)", "[3,5)", "[5,10)", "10+")
effn_labels <- c("1.0–1.1", "1.1–1.5", "1.5–2", "2–3", "3–5", "5–10", "10+")

average_hist <- function(path, bin_col, bins, labels) {
  if (!file.exists(path)) stop(paste("Input file not found:", path))
  read_csv(path, show_col_types = FALSE) %>%
    filter(wallet_type == "ALL", n_bucket == "2+", wealth_bucket %in% wealth_levels) %>%
    select(block, wealth_bucket, bin = all_of(bin_col), pct_accounts) %>%
    # a bin without accounts has no row; count it as 0 % before averaging
    complete(block, wealth_bucket, bin = bins, fill = list(pct_accounts = 0)) %>%
    group_by(wealth_bucket, bin) %>%
    summarise(pct = mean(pct_accounts), .groups = "drop") %>%
    mutate(
      wealth = factor(wealth_labels[match(wealth_bucket, wealth_levels)], levels = wealth_labels),
      bin = factor(labels[match(bin, bins)], levels = labels)
    )
}

maxw <- average_hist(opt$max_weight, "max_weight_bin", maxw_bins, maxw_labels)
effn <- average_hist(opt$effective_n, "effective_n_bin", effn_bins, effn_labels)

# -----------------------------------------------------------------------------
# THEME (as in the portfolio-size figure)
# -----------------------------------------------------------------------------
fintech_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    axis.title = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text = element_text(size = FONT_AXIS_TEXT, color = "grey30"),
    axis.line = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "grey90", linetype = "dashed"),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA),
    legend.title = element_text(size = FONT_LEGEND_TITLE, color = "grey30", face = "bold"),
    legend.text = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),
    legend.position = "right",
    plot.tag = element_text(size = FONT_STRIP, face = "bold", color = "grey20"),
    plot.margin = margin(6, 10, 6, 10),
    axis.text.x = element_text(margin = margin(t = 6))
  )

stacked <- function(df, legend_title) {
  ggplot(df, aes(x = wealth, y = pct, fill = bin)) +
    geom_col(width = 0.70) +
    scale_y_continuous(labels = function(x) paste0(round(x, 0), "%"),
                       expand = expansion(mult = c(0.01, 0.05))) +
    scale_fill_viridis_d(option = "C", end = 0.95, direction = -1) +
    labs(x = "Wealth category (USD)", y = "Avg % of multi-asset accounts", fill = legend_title) +
    fintech_theme
}

p <- (stacked(maxw, "Largest\nweight") / stacked(effn, "Effective\nnumber of\nassets")) +
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")")

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
dir.create(dirname(opt$out_pdf), showWarnings = FALSE, recursive = TRUE)
ggsave(filename = opt$out_pdf, plot = p, width = opt$width, height = opt$height,
       device = "pdf", encoding = "WinAnsi")  # WinAnsi has the en-dash
cat(sprintf("[ok] Saved PDF: %s\n", opt$out_pdf))
