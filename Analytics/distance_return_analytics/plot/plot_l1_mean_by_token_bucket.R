#!/usr/bin/env Rscript
# =============================================================================
# plot_l1_mean_by_token_bucket.R
#
# Description:
#   Appendix figure: three panels (MaxRet | MinVar | MaxSR) showing
#   mean L1 distance over time for granular portfolio-size buckets
#   (1, 2, 3, 4, 5, 6-10, 11-20, 21+).
#
# Input:
#   CSV with columns: date, strategy, token_bucket, mean_dist.
#
# Output:
#   Single PDF (default: ../graphics/l1_mean_by_token_bucket.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(tibble)
  library(ggplot2)
  library(scales)
  library(lubridate)
  library(patchwork)
  library(ggrepel)
})

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_IN_CSV  <- "../data/per_block_distance_by_token_bucket.csv"
DEFAULT_OUT_PDF <- "../graphics/l1_mean_by_token_bucket.pdf"

# Ordered bucket levels (natural portfolio-size order)
BUCKET_ORDER <- c("1", "2", "3", "4", "5", "6-10", "11-20", "21+")

BUCKET_LABELS <- c(
  "1"     = "1",
  "2"     = "2",
  "3"     = "3",
  "4"     = "4",
  "5"     = "5",
  "6-10"  = "6\u201310",
  "11-20" = "11\u201320",
  "21+"   = "21+"
)

# Viridis Plasma palette, 8 discrete stops
BUCKET_COLORS <- c(
  "1"     = "#0D0887",
  "2"     = "#46039F",
  "3"     = "#7201A8",
  "4"     = "#9C179E",
  "5"     = "#BD3786",
  "6-10"  = "#D8576B",
  "11-20" = "#ED7953",
  "21+"   = "#FDCA26"
)

STRATEGY_ORDER <- c("safer_risk", "better_return", "max_sharpe")
STRATEGY_TITLES <- c(
  "better_return" = "MaxRet",
  "safer_risk"    = "MinVar",
  "max_sharpe"    = "MaxSR"
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
  make_option(c("--height"), type = "double", default = 5.5,
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
df <- read_csv(opt$`in-csv`, show_col_types = FALSE) %>%
  mutate(
    date        = as.Date(date),
    token_group = factor(token_group, levels = BUCKET_ORDER)
  ) %>%
  filter(!is.na(date), !is.na(mean_dist),
         token_group %in% BUCKET_ORDER)

cat(sprintf("[info] %d rows loaded\n", nrow(df)))

# Per-strategy, per-bucket: time-series average (for legend)
legend_stats <- df %>%
  group_by(optimisation, token_group) %>%
  summarise(
    avg_mean = round(mean(mean_dist), 1),
    .groups  = "drop"
  ) %>%
  mutate(
    display = BUCKET_LABELS[as.character(token_group)],
    label   = paste0(display, " (", avg_mean, "%)")
  )

# -----------------------------------------------------------------------------
# THEME
# -----------------------------------------------------------------------------
fintech_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    plot.title    = element_text(face = "bold", size = 12, color = "grey20",
                                hjust = 0.5),
    plot.subtitle = element_blank(),

    axis.title   = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text    = element_text(color = "grey30"),

    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "grey90", linetype = "dashed"),

    strip.text       = element_text(face = "bold", size = FONT_STRIP, color = "grey20"),
    strip.background = element_blank(),

    legend.position  = "bottom",
    legend.direction = "horizontal",
    legend.title     = element_text(size = FONT_LEGEND_TITLE, color = "grey30",
                                    face = "bold", hjust = 0.5),
    legend.text      = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),
    legend.key.width = unit(0.7, "cm"),
    legend.spacing.x = unit(0.15, "cm"),

    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),

    plot.margin = margin(10, 10, 5, 10)  # right margin for % labels
  )

# -----------------------------------------------------------------------------
# HELPER: build one panel
# -----------------------------------------------------------------------------
build_panel <- function(data, stats, panel_title) {

  # Colour mapping by token_group
  bucket_colors <- BUCKET_COLORS[levels(data$token_group)]

  # Endpoint labels: last date per bucket, showing only avg %
  endpoint_df <- data %>%
    group_by(token_group) %>%
    filter(date == max(date)) %>%
    ungroup() %>%
    left_join(stats %>% select(token_group, avg_mean), by = "token_group") %>%
    mutate(
      end_label = paste0(avg_mean, "%"),
      label_x   = max(date) + 100
    )

  ggplot(data, aes(x = date, y = mean_dist, color = token_group)) +

    geom_line(linewidth = 0.7) +

    # % labels anchored in the right margin
    geom_text_repel(
      data = endpoint_df,
      aes(x = label_x, label = end_label),
      hjust             = 0,
      direction         = "y",
      segment.size      = 0.3,
      segment.color     = "grey70",
      min.segment.length = 0,       # always draw connector
      size              = 2.8,
      fontface          = "bold",
      force             = 0.5,
      xlim              = c(NA, NA),
      show.legend       = FALSE
    ) +

    scale_color_manual(
      values = bucket_colors,
      labels = BUCKET_LABELS
    ) +

    scale_x_date(
      breaks = seq.Date(as.Date("2020-01-01"), as.Date("2025-01-01"), by = "1 year"),
      date_labels = "'%y",
      expand = expansion(mult = c(0.01, 0.10))
    ) +

    scale_y_continuous(
      limits = c(0, 100),
      breaks = seq(0, 100, by = 20),
      labels = function(x) paste0(round(x, 0), "%"),
      expand = expansion(mult = c(0, 0.02))
    ) +

    labs(
      title = panel_title,
      x = NULL,
      y = "Mean L1 distance (%)"
    ) +

    guides(color = guide_legend(
      nrow = 1,
      title = "Tokens held",
      title.position = "top"
    )) +

    coord_cartesian(clip = "off") +

    fintech_theme
}

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
cat("[info] Building plot...\n")

panels <- list()
for (i in seq_along(STRATEGY_ORDER)) {
  strat <- STRATEGY_ORDER[i]
  sub_df    <- df %>% filter(optimisation == strat)
  sub_stats <- legend_stats %>% filter(optimisation == strat)

  p <- build_panel(sub_df, sub_stats, STRATEGY_TITLES[strat])
  if (i > 1) p <- p + labs(y = NULL)
  panels[[i]] <- p
}

p_final <- panels[[1]] + panels[[2]] + panels[[3]] +
  plot_layout(ncol = 3, guides = "collect") &
  theme(legend.position = "bottom")

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
out_file <- opt$`out-pdf`
if (!grepl("\\.pdf$", out_file, ignore.case = TRUE)) {
  out_file <- paste0(out_file, ".pdf")
}
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

ggsave(out_file, plot = p_final, width = opt$width, height = opt$height, device = "pdf")
cat(sprintf("[ok] Saved PDF: %s\n", out_file))