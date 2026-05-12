#!/usr/bin/env Rscript
# =============================================================================
# plot_l1_mean_by_token_group.R
#
# Description:
#   Three-facet plot (MinVar | MaxRet | MaxSR). Each panel shows two
#   lines: <5 tokens vs >=5 tokens. Shared legend for colour mapping.
#
# Input:
#   CSV with columns: date, strategy, token_group, mean_dist.
#
# Output:
#   Single PDF (default: ../graphics/l1_mean_by_token_group.pdf).
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
})

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_IN_CSV  <- "../data/per_block_distance_by_token_group.csv"
DEFAULT_OUT_PDF <- "../graphics/l1_mean_by_token_group.pdf"

GROUP_COLORS <- c(
  "< 5 tokens"  = "#FF7F00",
  ">= 5 tokens" = "#1F78B4"
)

GROUP_DISPLAY <- c(
  "lt5"  = "< 5 tokens",
  "gte5" = ">= 5 tokens"
)

# Strategy display names and panel order
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
# LOAD & PROCESS
# -----------------------------------------------------------------------------
cat("[info] Loading data...\n")
df <- read_csv(opt$`in-csv`, show_col_types = FALSE) %>%
  mutate(
    date  = as.Date(date),
    group = GROUP_DISPLAY[token_group]
  ) %>%
  filter(!is.na(date), !is.na(mean_dist),
         token_group %in% c("lt5", "gte5"))

df$group <- factor(df$group, levels = GROUP_DISPLAY)

cat(sprintf("[info] %d rows loaded\n", nrow(df)))

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
    legend.title     = element_blank(),
    legend.text      = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),
    legend.key.width = unit(0.8, "cm"),
    legend.spacing.x = unit(0.2, "cm"),

    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),

    plot.margin = margin(5, 10, 15, 10)
  )

# -----------------------------------------------------------------------------
# HELPER: build one panel (one strategy)
# -----------------------------------------------------------------------------
build_panel <- function(data, panel_title) {

  ggplot(data, aes(x = date, y = mean_dist, color = group)) +

    # Mean line
    geom_line(linewidth = 0.9) +

    scale_color_manual(values = GROUP_COLORS) +

    scale_x_date(
      breaks = seq.Date(as.Date("2020-01-01"), as.Date("2025-01-01"), by = "1 year"),
      date_labels = "'%y",
      expand = expansion(mult = c(0.01, 0.03))
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
      y = "Mean distance (%)"
    ) +

    coord_cartesian(clip = "off") +

    fintech_theme
}

# -----------------------------------------------------------------------------
# PLOT: three panels, one per strategy
# -----------------------------------------------------------------------------
cat("[info] Building plot...\n")

panels <- list()
for (i in seq_along(STRATEGY_ORDER)) {
  strat <- STRATEGY_ORDER[i]
  sub_df    <- df %>% filter(optimisation == strat)

  p <- build_panel(sub_df, STRATEGY_TITLES[strat])

  # Only leftmost panel gets y-axis label
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