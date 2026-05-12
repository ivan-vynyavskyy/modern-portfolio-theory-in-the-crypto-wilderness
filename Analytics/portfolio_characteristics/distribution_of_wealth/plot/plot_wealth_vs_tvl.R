#!/usr/bin/env Rscript
# =============================================================================
# plot_wealth_vs_tvl.R
#
# Description:
#   Two-line chart comparing total market wealth (EOA + CA) against TVL
#   (CA-only holdings) over time. Each series is shaded with a translucent
#   area fill, and the final data points are annotated with their value.
#
# Input:
#   CSV with columns: block, wallet_type, value_sum_usd
#   (default: ../data/value_buckets_minimal.csv).
#
# Output:
#   Single PDF (default: wealth_vs_tvl.pdf).
#
# Usage:
#   Rscript plot_wealth_vs_tvl.R \
#     --in-csv ../data/value_buckets_minimal.csv \
#     --out-pdf wealth_vs_tvl.pdf
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
  library(lubridate)
  library(ggrepel)
})

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_IN_CSV  <- "../data/value_buckets_minimal.csv"
DEFAULT_OUT_PDF <- "../graphics/wealth_vs_tvl.pdf"
DEFAULT_WIDTH   <- 10
DEFAULT_HEIGHT  <- 4

# Brewer Paired palette: dark = lines, light = area fills.
COLOR_WEALTH      <- "#1F78B4"
COLOR_TVL         <- "#FF7F00"
COLOR_WEALTH_FILL <- "#A6CEE3"
COLOR_TVL_FILL    <- "#FDBF6F"

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
# CLI SETUP
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("-i", "--in-csv"),  type = "character", default = DEFAULT_IN_CSV,
              help = "Input CSV path"),
  make_option(c("-o", "--out-pdf"), type = "character", default = DEFAULT_OUT_PDF,
              help = "Output PDF path")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (!file.exists(opt$`in-csv`)) stop("Input file not found.")

plot_width <- apply_plot_theme_fonts(DEFAULT_WIDTH)

# -----------------------------------------------------------------------------
# DATA PROCESSING
# -----------------------------------------------------------------------------
cat("Loading data...\n")
df <- read_csv(opt$`in-csv`, show_col_types = FALSE) %>%
  mutate(block = as.integer(block)) %>%
  left_join(b2d, by = "block") %>%
  filter(!is.na(date))

# Total wealth = EOA + CA summed across all buckets
df_wealth <- df %>%
  filter(wallet_type %in% c("EOA", "CA")) %>%
  group_by(date) %>%
  summarise(value = sum(value_sum_usd, na.rm = TRUE), .groups = "drop") %>%
  mutate(series = "Total Wealth (EOA + CA)")

# TVL = CA only, summed across all buckets
df_tvl <- df %>%
  filter(wallet_type == "CA") %>%
  group_by(date) %>%
  summarise(value = sum(value_sum_usd, na.rm = TRUE), .groups = "drop") %>%
  mutate(series = "TVL (CA Holdings)")

df_lines <- bind_rows(df_wealth, df_tvl)

# Wide format for shaded area fills
df_wide <- df_lines %>%
  tidyr::pivot_wider(names_from = series, values_from = value) %>%
  rename(wealth = `Total Wealth (EOA + CA)`, tvl = `TVL (CA Holdings)`)

# End-point labels
fmt_usd_si <- function(x) paste0("$", scales::label_number(
  accuracy = 0.1, scale_cut = scales::cut_short_scale())(x))

last_points <- df_lines %>%
  group_by(series) %>%
  filter(date == max(date)) %>%
  ungroup() %>%
  mutate(label = fmt_usd_si(value))

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
cat("Generating plot...\n")

series_colors <- c(
  "Total Wealth (EOA + CA)" = COLOR_WEALTH,
  "TVL (CA Holdings)"       = COLOR_TVL
)
series_lty <- c(
  "Total Wealth (EOA + CA)" = "solid",
  "TVL (CA Holdings)"       = "solid"
)

# Explicit y-axis breaks and labels
y_breaks <- c(0, 500e9, 1e12, 1.5e12)
y_labels <- c("$0", "$500B", "$1T", "$1.5T")

p <- ggplot() +

  # Shaded areas under each series
  geom_area(
    data  = df_wide,
    aes(x = date, y = wealth),
    fill  = COLOR_WEALTH_FILL, alpha = 0.35
  ) +
  geom_area(
    data  = df_wide,
    aes(x = date, y = tvl),
    fill  = COLOR_TVL_FILL, alpha = 0.50
  ) +

  # Primary series lines
  geom_line(
    data = df_lines,
    aes(x = date, y = value, color = series, linetype = series),
    linewidth = 1.4
  ) +

  # End-point markers and labels
  geom_point(data = last_points, aes(x = date, y = value, color = series), size = 3.5) +
  geom_label_repel(
    data        = last_points,
    aes(x = date, y = value, label = label, color = series),
    nudge_x     = 30,
    direction   = "y",
    hjust       = 0,
    fontface    = "bold",
    size        = 4.5,
    show.legend = FALSE,
    segment.size  = 0.5
  ) +

  scale_color_manual(values = series_colors, name = NULL) +
  scale_linetype_manual(values = series_lty, name = NULL) +
  scale_y_continuous(
    breaks = y_breaks,
    labels = y_labels,
    expand = expansion(mult = c(0, 0.1))
  ) +
  scale_x_date(
    date_breaks  = "6 months",
    date_labels  = "%b '%y",
    expand       = expansion(mult = c(0, 0.05))
  ) +

  theme_minimal(base_size = FONT_BASE) +
  theme(
    plot.title    = element_blank(),
    plot.subtitle = element_blank(),
    plot.caption  = element_text(color = "#777777", size = 10,
                                 margin = margin(t = 15), face = "italic"),

    axis.title = element_blank(),
    axis.text  = element_text(size = FONT_AXIS_TEXT, color = "#333333"),

    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "#e5e5e5", linetype = "dashed"),

    legend.position  = "bottom",
    legend.title     = element_text(size = FONT_LEGEND_TITLE, color = "black"),
    legend.text      = element_text(size = FONT_LEGEND_TEXT, margin = margin(r = 10)),
    legend.key.size  = unit(0.5, "cm"),

    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin      = margin(10, 25, 10, 25)
  )

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
cat(sprintf("Saving to %s...\n", opt$`out-pdf`))
ggsave(opt$`out-pdf`, plot = p, width = plot_width, height = DEFAULT_HEIGHT, device = "pdf")
cat("Done.\n")