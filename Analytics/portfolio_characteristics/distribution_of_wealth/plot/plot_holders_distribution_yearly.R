#!/usr/bin/env Rscript
# =============================================================================
# plot_holders_distribution_yearly.R
#
# Description:
#   Horizontal stacked bar chart showing, for each year in 2020-2025, how
#   tokens distribute across holder-count buckets (0-100, 100-1K, 1K-10K,
#   >10K). Also prints a LaTeX table with yearly averages, medians and
#   bucket shares.
#
# Input:
#   CSV with columns: date, holders (hardcoded:
#   ../data/token_inequality_per_block.csv).
#
# Output:
#   PDF (hardcoded: holders_distribution_yearly.pdf) and a LaTeX table
#   written to stdout.
#
# Usage:
#   Rscript plot_holders_distribution_yearly.R
# =============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

# -----------------------------------------------------------------------------
# CONFIGURATION & PALETTE
# -----------------------------------------------------------------------------
INPUT_FILE  <- "../data/token_inequality_per_block.csv"
OUTPUT_FILE <- "holders_distribution_yearly.pdf"

# Brewer Paired palette (low -> high bucket).
PAIRED_COLORS <- c(
  "#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
  "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928"
)

BUCKET_COLORS <- c(
  "0-100"   = PAIRED_COLORS[1],
  "100-1K"  = PAIRED_COLORS[2],
  "1K-10K"  = PAIRED_COLORS[3],
  ">10K"    = PAIRED_COLORS[4]
)

# -----------------------------------------------------------------------------
# LOAD & PROCESS
# -----------------------------------------------------------------------------
if (!file.exists(INPUT_FILE)) {
  stop(paste("Error: File", INPUT_FILE, "not found."))
}

df <- read_csv(INPUT_FILE, show_col_types = FALSE)

df_clean <- df %>%
  mutate(
    date = as.Date(date),
    year = as.numeric(format(date, "%Y")),
    holders = as.numeric(holders)
  ) %>%
  filter(!is.na(date), !is.na(holders), !is.na(year)) %>%
  filter(year >= 2020, year <= 2025) %>%
  mutate(
    period = as.character(year),
    bucket = cut(
      holders,
      breaks = c(-Inf, 100, 1000, 10000, Inf),
      labels = c("0-100", "100-1K", "1K-10K", ">10K"),
      right = TRUE
    )
  )

# -----------------------------------------------------------------------------
# AGGREGATE
# -----------------------------------------------------------------------------
plot_data <- df_clean %>%
  count(period, bucket) %>%
  group_by(period) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup()

stats_df <- df_clean %>%
  group_by(period) %>%
  summarise(
    avg_holders = mean(holders),
    median_holders = median(holders),
    .groups = "drop"
  )

# -----------------------------------------------------------------------------
# FACTOR ORDERING
# -----------------------------------------------------------------------------
bucket_levels <- c("0-100", "100-1K", "1K-10K", ">10K")
plot_data <- plot_data %>%
  mutate(
    bucket = factor(bucket, levels = bucket_levels, ordered = TRUE)
  )

period_levels <- as.character(2020:2025)
plot_data <- plot_data %>%
  mutate(
    period = factor(period, levels = rev(period_levels))
  )

plot_data <- plot_data %>%
  mutate(label_text = ifelse(pct >= 0.04, sprintf("%.1f%%", pct * 100), ""))

# -----------------------------------------------------------------------------
# PLOT
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

PLOT_WIDTH  <- 10
PLOT_HEIGHT <- 4
PLOT_WIDTH  <- apply_plot_theme_fonts(PLOT_WIDTH)

p <- ggplot(plot_data, aes(x = period, y = pct, fill = bucket)) +
  geom_col(
    width = 0.96,
    color = "white",
    linewidth = 0.2,
    position = position_stack(reverse = TRUE)
  ) +
  geom_text(
    aes(label = label_text),
    position = position_stack(vjust = 0.5, reverse = TRUE),
    color = "black",
    size = 3.5,
    fontface = "bold"
  ) +
  coord_flip() +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  scale_x_discrete(
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  scale_fill_manual(
    values = BUCKET_COLORS,
    breaks = bucket_levels,
    name   = "N of token holders"
  ) +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = FONT_BASE) +
  theme(
    axis.title = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text  = element_text(size = FONT_AXIS_TEXT,  color = "grey30"),

    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "grey90", linetype = "dashed"),
    panel.grid.minor   = element_blank(),

    legend.position = "bottom",
    legend.title    = element_text(size = FONT_LEGEND_TITLE, color = "black"),
    legend.text     = element_text(size = FONT_LEGEND_TEXT),

    plot.background = element_rect(fill = "white", color = NA),
    plot.margin     = margin(8, 8, 8, 8)
  ) +
  guides(fill = guide_legend(nrow = 1, reverse = FALSE))

ggsave(OUTPUT_FILE, plot = p, width = PLOT_WIDTH, height = PLOT_HEIGHT, device = "pdf")
message(sprintf("[ok] Saved plot to: %s", OUTPUT_FILE))

# -----------------------------------------------------------------------------
# LATEX TABLE
# -----------------------------------------------------------------------------
table_wide <- plot_data %>%
  select(period, bucket, pct) %>%
  pivot_wider(names_from = bucket, values_from = pct, values_fill = 0) %>%
  left_join(stats_df, by = "period")

table_wide$period <- factor(table_wide$period, levels = period_levels)
table_wide <- table_wide %>% arrange(period)

cat("\n\\begin{table}[H]\n")
cat("\\centering\n")
cat("\\caption{Token holder structure: Average, Median, percentages of holder ranges (Yearly).}\n")
cat("\\label{tab:holders_yearly}\n")
cat("\\begin{tabular}{crrrrrr}\n")
cat("\\toprule\n")
cat("& Average & Median & $0-10^2$ & $10^2-10^3$ & $10^3-10^4$ & $>10^4$ \\\\\n")
cat("\\midrule\n")

for (i in seq_len(nrow(table_wide))) {
  row <- table_wide[i, ]

  fmt_num <- function(x) format(round(x), big.mark = "{,}", scientific = FALSE)
  fmt_pct <- function(x) sprintf("%.1f \\%%", x * 100)

  cat(sprintf(
    "%s & %s & %s & %s & %s & %s & %s \\\\\n",
    row$period,
    fmt_num(row$avg_holders),
    fmt_num(row$median_holders),
    fmt_pct(row[["0-100"]]),
    fmt_pct(row[["100-1K"]]),
    fmt_pct(row[["1K-10K"]]),
    fmt_pct(row[[">10K"]])
  ))
}

cat("\\bottomrule\n")
cat("\\end{tabular}\n")
cat("\\end{table}\n")