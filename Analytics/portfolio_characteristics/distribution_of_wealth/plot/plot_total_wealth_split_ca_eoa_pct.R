#!/usr/bin/env Rscript
# =============================================================================
# plot_total_wealth_split_ca_eoa_pct.R
#
# Description:
#   Two-group stacked-area plot showing the share of total ERC-20 wealth
#   held by EOAs vs contract accounts (CAs) over time. Alongside the plot,
#   summary statistics (mean, median, sd, min/max, quantiles) for each
#   group are printed to stdout.
#
# Input:
#   CSV with columns: block, wallet_type, bucket, wallet_count, wallet_pct,
#   value_sum_usd, value_pct.
#
# Output:
#   Single PDF with the stacked-area chart.
#
# Usage:
#   Rscript plot_total_wealth_split_ca_eoa_pct.R \
#     --in-csv ../data/value_buckets_minimal.csv \
#     --out-pdf total_wealth_split_ca_eoa_pct.pdf
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(lubridate)
  library(scales)
  library(tidyr)
})

# -----------------------------------------------------------------------------
# COLOR PALETTE (Brewer Paired)
# -----------------------------------------------------------------------------
PAIRED_COLORS <- c(
  "#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
  "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928"
)

COLOR_MAP <- c(
  "EOA" = PAIRED_COLORS[1],
  "CA"  = PAIRED_COLORS[2]
)

# -----------------------------------------------------------------------------
# DATA MAPPING (Block -> Date) — sourced from shared file
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
source(file.path(script_dir, "block_date_mapping.R"))
source(file.path(script_dir, "..", "..", "..", "shared", "plot_theme.R"))

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("--in-csv"),  dest = "in_csv",  type = "character", help = "Input CSV path", metavar = "PATH"),
  make_option(c("--out-pdf"), dest = "out_pdf", type = "character", default = "total_wealth_split_ca_eoa_pct.pdf", help = "Output PDF path", metavar = "PATH"),
  make_option(c("--width"),   dest = "width",   type = "double", default = 10, help = "Plot width (in)", metavar = "NUM"),
  make_option(c("--height"),  dest = "height",  type = "double", default = 4.0, help = "Plot height (in) (kept low)", metavar = "NUM")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$in_csv) || is.na(opt$in_csv) || opt$in_csv == "") {
  stop("Missing --in-csv.", call. = FALSE)
}

opt$width <- apply_plot_theme_fonts(opt$width)

# -----------------------------------------------------------------------------
# LOAD & PROCESS
# -----------------------------------------------------------------------------
df <- read_csv(opt$in_csv, show_col_types = FALSE)

required_cols <- c("block", "wallet_type", "value_sum_usd")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")), call. = FALSE)
}

df_clean <- df %>%
  mutate(
    block = as.integer(block),
    wallet_type = as.character(wallet_type),
    value_sum_usd = as.numeric(value_sum_usd)
  ) %>%
  filter(wallet_type %in% c("CA", "EOA")) %>%
  group_by(block, wallet_type) %>%
  summarise(total_wealth_usd = sum(value_sum_usd, na.rm = TRUE), .groups = "drop")

df_complete <- df_clean %>%
  complete(block, wallet_type, fill = list(total_wealth_usd = 0))

df_final <- df_complete %>%
  group_by(block) %>%
  mutate(
    total_all = sum(total_wealth_usd, na.rm = TRUE),
    wealth_pct = if_else(total_all > 0, 100 * total_wealth_usd / total_all, 0)
  ) %>%
  ungroup() %>%
  inner_join(b2d, by = "block") %>%
  mutate(wallet_type = factor(wallet_type, levels = c("EOA", "CA"))) %>%
  arrange(date)

if (nrow(df_final) == 0) {
  stop("No rows left after processing.", call. = FALSE)
}

# -----------------------------------------------------------------------------
# SUMMARY STATS (CA vs EOA split over time)
# -----------------------------------------------------------------------------
summary_df <- df_final %>%
  select(date, wallet_type, wealth_pct) %>%
  arrange(date)

calc_stats <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(list(
      mean = NA_real_, median = NA_real_, sd = NA_real_,
      min = NA_real_, max = NA_real_,
      p10 = NA_real_, p25 = NA_real_, p75 = NA_real_, p90 = NA_real_
    ))
  }
  list(
    mean   = mean(x),
    median = median(x),
    sd     = sd(x),
    min    = min(x),
    max    = max(x),
    p10    = as.numeric(quantile(x, 0.10)),
    p25    = as.numeric(quantile(x, 0.25)),
    p75    = as.numeric(quantile(x, 0.75)),
    p90    = as.numeric(quantile(x, 0.90))
  )
}

cat("\n============================================================\n")
cat("[summary] Wealth split over time (percent of total ERC-20 wealth)\n")
cat(sprintf("[summary] Time span: %s  ->  %s  (%d snapshots)\n",
            format(min(summary_df$date), "%Y-%m-%d"),
            format(max(summary_df$date), "%Y-%m-%d"),
            length(unique(summary_df$date))))
cat("============================================================\n\n")

for (wt in c("CA", "EOA")) {
  vec <- summary_df %>% filter(wallet_type == wt) %>% pull(wealth_pct)
  st <- calc_stats(vec)

  min_row <- summary_df %>% filter(wallet_type == wt) %>% arrange(wealth_pct, date) %>% slice(1)
  max_row <- summary_df %>% filter(wallet_type == wt) %>% arrange(desc(wealth_pct), date) %>% slice(1)

  cat(sprintf(">>> %s\n", wt))
  cat(sprintf("  mean   : %6.2f%%\n", st$mean))
  cat(sprintf("  median : %6.2f%%\n", st$median))
  cat(sprintf("  sd     : %6.2f\n",   st$sd))
  cat(sprintf("  min    : %6.2f%%  (%s)\n", st$min, format(min_row$date, "%b '%y")))
  cat(sprintf("  max    : %6.2f%%  (%s)\n", st$max, format(max_row$date, "%b '%y")))
  cat(sprintf("  p10/p25 : %6.2f%% / %6.2f%%\n", st$p10, st$p25))
  cat(sprintf("  p75/p90 : %6.2f%% / %6.2f%%\n", st$p75, st$p90))
  cat("\n")
}

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
p <- ggplot(df_final, aes(x = date, y = wealth_pct, fill = wallet_type, color = wallet_type)) +
  geom_area(
    linewidth = 0.2,
    alpha = 1
  ) +
  scale_fill_manual(values = COLOR_MAP, drop = FALSE) +
  scale_color_manual(values = COLOR_MAP, drop = FALSE) +
  scale_x_date(
    date_breaks = "6 months",
    date_labels = "%b '%y",
    expand = expansion(mult = c(0.01, 0.06))
  ) +
  scale_y_continuous(
    limits = c(0, 100.1),
    breaks = seq(0, 100, by = 25),
    labels = function(x) paste0(x, "%")
  ) +
  labs(
    x = NULL,
    y = "Share of total wealth",
    fill = NULL,
    color = NULL
  ) +
  theme_minimal(base_size = FONT_BASE) +
  theme(
    axis.title    = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text     = element_text(size = FONT_AXIS_TEXT,  color = "grey30"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "grey90", linetype = "dashed"),
    legend.position = "bottom",
    legend.title    = element_text(size = FONT_LEGEND_TITLE),
    legend.text     = element_text(size = FONT_LEGEND_TEXT),
    legend.key.height = unit(0.4, "cm"),
    legend.key.width  = unit(0.4, "cm"),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(10, 8, 6, 10)
  )

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
out_file <- opt$out_pdf
if (!grepl("\\.pdf$", out_file, ignore.case = TRUE)) {
  out_file <- paste0(out_file, ".pdf")
}

ggsave(filename = out_file, plot = p, width = opt$width, height = opt$height)
message(sprintf("[ok] Saved: %s", out_file))
