#!/usr/bin/env Rscript
# =============================================================================
# plot_avg_ineq_combined.R
#
# Description:
#   Two-panel plot showing the average and median of per-token Gini and HHI
#   inequality measures over time. Aggregates across all tokens above a
#   configurable holder-count threshold (default: > 100 holders).
#
# Input:
#   CSV with columns: date, gini, hhi, holders, token_address.
#
# Output:
#   Single two-panel PDF (Gini on top, HHI on bottom).
#
# Usage:
#   Rscript plot_avg_ineq_combined.R \
#     --in-csv ../data/token_inequality_per_block.csv \
#     --out-pdf avg_ineq_combined.pdf
# =============================================================================

quiet_require <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

pkgs <- c("optparse", "readr", "dplyr", "stringr", "lubridate",
          "ggplot2", "scales", "ggrepel", "patchwork")
invisible(lapply(pkgs, quiet_require))

# -----------------------------------------------------------------------------
# CLI OPTIONS
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("--in-csv"), type = "character",
              help = "Inequality CSV (date, gini, hhi, holders, token_address)."),
  make_option(c("--out-pdf"), type = "character",
              default = "avg_ineq_combined.pdf",
              help = "Output PDF path [default %default]."),
  make_option(c("-m", "--min-holders"), type = "integer", default = 100,
              help = "Keep rows where holders > this threshold [default %default]."),
  make_option(c("-s", "--start-date"), type = "character", default = "2020-01",
              help = "Only include rows with date >= YYYY-MM [default %default]."),
  make_option(c("--width"),  type = "double", default = 12,
              help = "PDF width in inches [default %default]."),
  make_option(c("--height"), type = "double", default = 7,
              help = "PDF height in inches [default %default].")
)

opt <- parse_args(OptionParser(option_list = option_list))

csv_path <- opt[["in-csv"]]
if (is.null(csv_path) || !file.exists(csv_path)) {
  stop("Provide a valid --in-csv path.")
}

# -----------------------------------------------------------------------------
# LOAD & CLEAN
# -----------------------------------------------------------------------------
cat("Reading data...\n")
df <- read_csv(csv_path, show_col_types = FALSE, col_types = cols(.default = col_character()))

# Drop any duplicate header rows that may appear mid-file
df <- df %>% filter(date != "date")

token_col <- if ("token_address" %in% names(df)) "token_address" else "token"

df <- df %>%
  mutate(
    date    = ymd(date, quiet = TRUE),
    holders = as.numeric(holders),
    gini    = as.numeric(gini),
    hhi     = as.numeric(hhi)
  ) %>%
  filter(!is.na(date), !is.na(holders), !is.na(gini), !is.na(hhi))

if (!is.null(opt[["start-date"]])) {
  sd <- suppressWarnings(ymd(opt[["start-date"]], quiet = TRUE))
  if (is.na(sd)) sd <- suppressWarnings(ymd(paste0(opt[["start-date"]], "-01"), quiet = TRUE))
  if (!is.na(sd)) df <- df %>% filter(date >= sd)
}

df <- df %>% filter(holders > opt[["min-holders"]])
cat(sprintf("Rows after filtering: %d\n", nrow(df)))

# -----------------------------------------------------------------------------
# AGGREGATE PER DATE
# -----------------------------------------------------------------------------
agg <- df %>%
  group_by(date) %>%
  summarise(
    gini_avg = mean(gini, na.rm = TRUE),
    gini_med = median(gini, na.rm = TRUE),
    hhi_avg  = mean(hhi, na.rm = TRUE),
    hhi_med  = median(hhi, na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  arrange(date)

# -----------------------------------------------------------------------------
# RESHAPE FOR PLOTTING
# -----------------------------------------------------------------------------
library(tidyr)

gini_long <- agg %>%
  select(date, Average = gini_avg, Median = gini_med) %>%
  pivot_longer(-date, names_to = "stat", values_to = "val") %>%
  mutate(stat = factor(stat, levels = c("Average", "Median")))

hhi_long <- agg %>%
  select(date, Average = hhi_avg, Median = hhi_med) %>%
  pivot_longer(-date, names_to = "stat", values_to = "val") %>%
  mutate(stat = factor(stat, levels = c("Average", "Median")))

# -----------------------------------------------------------------------------
# THEME & PALETTE
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
    axis.title         = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text          = element_text(size = FONT_AXIS_TEXT,  color = "grey30"),
    axis.line          = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "grey90", linetype = "dashed"),
    legend.title       = element_blank(),
    legend.text        = element_text(
      size = FONT_LEGEND_TEXT, color = "grey30", face = "bold"
    )
  )

pal <- c("Average" = "#1f77b4", "Median" = "#ff7f0e")

# -----------------------------------------------------------------------------
# PLOT BUILDER
# -----------------------------------------------------------------------------
make_plot <- function(long_df, y_lab, y_lims, remove_x = FALSE) {

  last_vals <- long_df %>%
    group_by(stat) %>%
    slice_max(date, n = 1) %>%
    mutate(label_text = sprintf("%.2f", val))

  p <- ggplot(long_df, aes(x = date, y = val, color = stat, group = stat)) +
    geom_line(linewidth = 1.1) +
    geom_text_repel(
      data = last_vals,
      aes(label = label_text),
      nudge_x   = 20,
      direction  = "y",
      hjust      = 0,
      fontface   = "bold",
      size       = 3.5,
      show.legend = FALSE
    ) +
    scale_color_manual(values = pal) +
    scale_y_continuous(labels = number_format(accuracy = 0.01)) +
    scale_x_date(
      date_breaks = "6 months",
      date_labels = "%b '%y",
      expand = expansion(mult = c(0.01, 0.05))
    ) +
    coord_cartesian(ylim = y_lims, clip = "off") +
    labs(x = NULL, y = y_lab) +
    fintech_theme

  if (remove_x) {
    p <- p + theme(
      axis.text.x  = element_blank(),
      axis.title.x = element_blank(),
      axis.ticks.x = element_blank(),
      plot.margin  = margin(10, 40, 2, 10)
    )
  } else {
    p <- p + theme(plot.margin = margin(0, 40, 10, 10))
  }
  p
}

# -----------------------------------------------------------------------------
# BUILD & SAVE
# -----------------------------------------------------------------------------
cat("Generating plots...\n")

p_gini <- make_plot(gini_long, "Gini Coefficient", c(0.90, 1.00), remove_x = TRUE)
p_hhi  <- make_plot(hhi_long,  "HHI",              c(0.00, 0.40), remove_x = FALSE)

p_combined <- (p_gini / p_hhi) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

out_path <- opt[["out-pdf"]]
if (!grepl("\\.pdf$", out_path, ignore.case = TRUE)) out_path <- paste0(out_path, ".pdf")
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

ggsave(out_path, p_combined, width = opt$width, height = opt$height, device = grDevices::pdf)
cat(sprintf("[ok] Saved: %s\n", out_path))