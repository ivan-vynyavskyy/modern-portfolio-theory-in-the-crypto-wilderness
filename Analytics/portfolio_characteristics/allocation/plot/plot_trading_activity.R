#!/usr/bin/env Rscript
# =============================================================================
# plot_trading_activity.R
#
# Description:
#   Month-to-month trading activity, 2020-2025, in three panels:
#   (a) accounts that keep a balance and change none of their holdings, all
#       accounts and multi-asset accounts;
#   (b) multi-asset accounts that reallocate more than 25 % of their
#       portfolio (turnover > 0.25);
#   (c) mean turnover (active reallocation) and mean drift (weight change from
#       prices alone) of multi-asset accounts.
#   Each point covers the month before the snapshot on the x axis.
#
# Input:
#   ../data/trading_drift_summary.csv (trading_vs_drift.py); block dates from
#   ../../distribution_of_wealth/plot/block_date_mapping.R
#
# Output:
#   ../graphics/trading_activity_monthly.pdf
#
# Usage:
#   Rscript plot_trading_activity.R
#   Rscript plot_trading_activity.R --out_pdf ../graphics/x.pdf --width 10 --height 3.8
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
  library(optparse)
})

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = "../data/trading_drift_summary.csv",
              help = "Trading summary CSV [default %default]"),
  make_option(c("-o", "--out_pdf"), type = "character", default = "../graphics/trading_activity_monthly.pdf",
              help = "Output PDF [default %default]"),
  make_option(c("--width"), type = "double", default = 10, help = "Plot width (in)"),
  make_option(c("--height"), type = "double", default = 3.8, help = "Plot height (in)")
)
opt <- parse_args(OptionParser(option_list = option_list))

script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grep("^--file=", args)]
  if (length(file_arg) > 0) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
})()
source(file.path(script_dir, "..", "..", "..", "shared", "plot_theme.R"))
source(file.path(script_dir, "..", "..", "distribution_of_wealth", "plot", "block_date_mapping.R"))
opt$width <- apply_plot_theme_fonts(opt$width)

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
if (!file.exists(opt$input)) stop(paste("Input file not found:", opt$input))
s <- read_csv(opt$input, show_col_types = FALSE)
multi_buckets <- c("2", "3–4", "5–9", "10+")

all_acc <- s %>%
  filter(wallet_type == "ALL", wealth_bucket == "ALL", n_bucket == "ALL") %>%
  transmute(block_t1, no_change_all = pct_units_unchanged)

multi <- s %>%
  filter(wallet_type == "ALL", wealth_bucket == "ALL", n_bucket %in% multi_buckets) %>%
  group_by(block_t1) %>%
  summarise(
    no_change_multi = 100 * sum(n_units_unchanged) / sum(n_stay),
    quarter = 100 - 100 * sum(n_turnover_le_25pct) / sum(n_measured),
    turnover = 100 * sum(mean_turnover * n_measured, na.rm = TRUE) / sum(n_measured),
    drift = 100 * sum(mean_drift * n_measured, na.rm = TRUE) / sum(n_measured),
    .groups = "drop"
  )

panel_a <- "(a) No balance change"
panel_b <- "(b) More than 25% reallocated"
panel_c <- "(c) Turnover and drift"
series_levels <- c("All accounts", "Multi-asset accounts",
                   "Mean turnover (multi-asset)", "Mean drift (multi-asset)")

df <- all_acc %>%
  inner_join(multi, by = "block_t1") %>%
  inner_join(b2d, by = c("block_t1" = "block")) %>%
  transmute(
    date,
    `All accounts|a` = no_change_all,
    `Multi-asset accounts|a` = no_change_multi,
    `Multi-asset accounts|b` = quarter,
    `Mean turnover (multi-asset)|c` = turnover,
    `Mean drift (multi-asset)|c` = drift
  ) %>%
  pivot_longer(-date, names_to = c("series", "panel"), names_sep = "\\|", values_to = "value") %>%
  mutate(
    panel = factor(c(a = panel_a, b = panel_b, c = panel_c)[panel], levels = c(panel_a, panel_b, panel_c)),
    series = factor(series, levels = series_levels)
  )

if (nrow(df) == 0) stop("No rows after joining the block dates; check the block numbers.")

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
colors <- c("All accounts" = "grey55", "Multi-asset accounts" = "#0D0887",
            "Mean turnover (multi-asset)" = "#CC4778", "Mean drift (multi-asset)" = "#F89540")

p <- ggplot(df, aes(x = date, y = value, color = series)) +
  geom_line(linewidth = 0.9) +
  facet_wrap(~panel, nrow = 1, scales = "free_y") +
  # data run from Feb 2020 to Dec 2025: label the year starts inside that range only
  scale_x_date(breaks = as.Date(sprintf("%d-01-01", 2021:2025)), date_labels = "%Y",
               expand = expansion(mult = c(0.01, 0.02))) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_manual(values = colors, breaks = series_levels) +
  labs(x = NULL, y = NULL, color = NULL) +
  theme_minimal(base_size = FONT_BASE) +
  theme(
    axis.text = element_text(size = FONT_AXIS_TEXT, color = "grey30"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "grey90", linetype = "dashed"),
    plot.background = element_rect(fill = "white", color = NA),
    strip.text = element_text(size = FONT_STRIP, face = "bold", color = "grey20", hjust = 0),
    legend.position = "bottom",
    legend.text = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),
    panel.spacing = unit(1.2, "lines"),
    plot.margin = margin(6, 10, 6, 6)
  )

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
dir.create(dirname(opt$out_pdf), showWarnings = FALSE, recursive = TRUE)
ggsave(filename = opt$out_pdf, plot = p, width = opt$width, height = opt$height,
       device = "pdf", encoding = "WinAnsi")
cat(sprintf("[ok] Saved PDF: %s\n", opt$out_pdf))
