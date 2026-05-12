#!/usr/bin/env Rscript
# =============================================================================
# plot_tokens_vs_distance.R
#
# Description:
#   Relationship between token count and L1 distance, faceted by
#   strategy. Median line + IQR ribbon + dot-size encoding for sample
#   count, with log-scaled x-axis to expose the logarithmic decay.
#
# Input:
#   CSV with columns: strategy, num_tokens, n, mean_dist, median_dist,
#   p25_dist, p75_dist.
#
# Output:
#   Single PDF (default: ../graphics/tokens_vs_distance.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(scales)
})

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_IN_CSV  <- "../data/tokens_vs_dist_agg.csv"
DEFAULT_OUT_PDF <- "../graphics/tokens_vs_distance.pdf"

STRATEGY_LABELS <- c(
  "safer_risk"    = "Min Variance (Same Ret)",
  "better_return" = "Max Return (Same Vol)",
  "max_sharpe"    = "Max Sharpe Ratio"
)

STRATEGY_COLORS <- c(
  "Min Variance (Same Ret)" = "#33A02C",
  "Max Return (Same Vol)"   = "#1F78B4",
  "Max Sharpe Ratio"        = "#E31A1C"
)

# Cap token count to avoid noisy tail with very few observations
MAX_TOKENS <- 50
MIN_OBS    <- 30

# x-axis: skip '26
LABEL_YEARS <- seq.Date(as.Date("2020-01-01"), as.Date("2025-01-01"), by = "1 year")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("-i", "--in-csv"), type = "character", default = DEFAULT_IN_CSV,
              help = "Input aggregated CSV [default: %default]"),
  make_option(c("-o", "--out-pdf"), type = "character", default = DEFAULT_OUT_PDF,
              help = "Output PDF path [default: %default]"),
  make_option(c("--max-tokens"), type = "integer", default = MAX_TOKENS,
              help = "Cap token count display [default: %default]"),
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
cat("[info] Loading pre-aggregated data...\n")
df <- read_csv(opt$`in-csv`, show_col_types = FALSE)

df <- df %>%
  filter(
    num_tokens >= 2,
    num_tokens <= opt$`max-tokens`,
    n          >= MIN_OBS
  ) %>%
  mutate(
    strategy = factor(STRATEGY_LABELS[strategy], levels = STRATEGY_LABELS)
  ) %>%
  filter(!is.na(strategy))

cat(sprintf("[info] %d rows, token range %d–%d, %d strategies\n",
            nrow(df), min(df$num_tokens), max(df$num_tokens),
            n_distinct(df$strategy)))

# -----------------------------------------------------------------------------
# THEME
# -----------------------------------------------------------------------------
fintech_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    plot.title    = element_blank(),
    plot.subtitle = element_blank(),

    axis.title   = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text    = element_text(color = "grey30"),

    panel.grid.major = element_line(color = "grey94", linetype = "dashed"),
    panel.grid.minor = element_blank(),

    strip.text       = element_text(face = "bold", size = FONT_STRIP, color = "grey20"),
    strip.background = element_blank(),

    legend.position  = "bottom",
    legend.title     = element_blank(),
    legend.text      = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),

    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),

    plot.margin = margin(10, 15, 10, 10)
  )

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
cat("[info] Building plot...\n")

p <- ggplot(df, aes(x = num_tokens)) +

  # --- IQR ribbon (p25–p75) ---
  geom_ribbon(
    aes(ymin = p25_dist, ymax = p75_dist, fill = strategy),
    alpha = 0.18
  ) +

  # --- Median line ---
  geom_line(
    aes(y = median_dist, color = strategy, linetype = "Median"),
    linewidth = 0.9
  ) +

  # --- Mean as dots ---
  geom_point(
    aes(y = mean_dist, color = strategy, shape = "Mean"),
    size = 1.6, alpha = 0.6
  ) +

  # --- Facet ---
  facet_wrap(~ strategy, ncol = 3) +

  # --- Scales ---
  scale_x_log10(
    breaks = c(2, 3, 5, 10, 20, 50),
    labels = c("2", "3", "5", "10", "20", "50"),
    expand = expansion(mult = c(0.02, 0.04))
  ) +

  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, by = 20),
    labels = function(x) paste0(round(x, 0), "%"),
    expand = expansion(mult = c(0, 0.02))
  ) +

  # Colour/fill driven by facet — suppress their legends
  scale_color_manual(values = STRATEGY_COLORS, guide = "none") +
  scale_fill_manual(values = STRATEGY_COLORS, guide = "none") +

  # Legend for line type
  scale_linetype_manual(
    name   = NULL,
    values = c("Median" = "solid"),
    guide  = guide_legend(order = 1, override.aes = list(
      linewidth = 0.9, color = "grey25"
    ))
  ) +

  # Legend for point shape
  scale_shape_manual(
    name   = NULL,
    values = c("Mean" = 16),
    guide  = guide_legend(order = 2, override.aes = list(
      size = 2.5, color = "grey25", alpha = 1
    ))
  ) +

  # Annotation: IQR explanation (appears once, bottom-right of legend row)
  annotate(
    "rect", xmin = 0, xmax = 0, ymin = 0, ymax = 0,
    fill = NA, color = NA
  ) +

  labs(
    x = "Number of tokens (log scale)",
    y = "L1 distance (%)"
  ) +

  fintech_theme +

  # Tweak legend layout: line + dot + manual IQR note
  theme(
    legend.position  = "bottom",
    legend.box       = "horizontal",
    legend.spacing.x = unit(0.8, "cm"),
    legend.margin    = margin(t = 5)
  )

# Add IQR ribbon note to the legend area via a secondary grob
# (ggplot can't auto-legend a ribbon cleanly, so we add a caption-style note)
p <- p +
  labs(caption = "Shaded band = interquartile range (P25–P75)") +
  theme(
    plot.caption = element_text(
      color = "grey45", size = 9.5, face = "italic",
      hjust = 0.5, margin = margin(t = 2, b = 2)
    )
  )

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
out_file <- opt$`out-pdf`
if (!grepl("\\.pdf$", out_file, ignore.case = TRUE)) out_file <- paste0(out_file, ".pdf")
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

ggsave(out_file, plot = p, width = opt$width, height = opt$height, device = "pdf")
cat(sprintf("[ok] Saved PDF: %s\n", out_file))