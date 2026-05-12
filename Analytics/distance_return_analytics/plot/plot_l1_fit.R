#!/usr/bin/env Rscript
# =============================================================================
# plot_l1_fit.R
#
# Description:
#   Three-facet plot (one per strategy) showing actual mean L1 distance
#   vs a fitted power-decay curve d(n) = c - a * n^{-b}. Each facet
#   includes the R-squared and asymptote annotation.
#
# Input:
#   CSV with columns: num_tokens, strategy, mean_dist (or similar).
#
# Output:
#   Single PDF (default: ../graphics/l1_fit_vs_actual.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_IN_CSV  <- "../data/tokens_vs_dist_agg.csv"
DEFAULT_OUT_PDF <- "../graphics/l1_fit_vs_actual.pdf"

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

# Fitted parameters: d(n) = c - a * n^{-b},  c <= 100
FIT_PARAMS <- list(
  "better_return" = list(a = 113.19, b = 0.4803, c = 100.00, r2 = 0.9937),
  "safer_risk"    = list(a = 138.29, b = 0.7774, c =  80.69, r2 = 0.9999),
  "max_sharpe"    = list(a =  99.36, b = 0.9022, c =  88.62, r2 = 0.9999)
)

MAX_TOKENS <- 50
MIN_OBS    <- 30

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("-i", "--in-csv"), type = "character", default = DEFAULT_IN_CSV,
              help = "Input aggregated CSV [default: %default]"),
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
df <- read_csv(opt$`in-csv`, show_col_types = FALSE)

df <- df %>%
  filter(
    num_tokens >= 2,
    num_tokens <= MAX_TOKENS,
    n          >= MIN_OBS
  ) %>%
  mutate(
    strategy_raw = strategy,
    strategy     = factor(STRATEGY_LABELS[strategy],
                          levels = STRATEGY_LABELS)
  ) %>%
  filter(!is.na(strategy))

# Generate fitted curves (smooth, 200 points per strategy)
n_seq <- seq(2, MAX_TOKENS, length.out = 200)

fit_df <- bind_rows(lapply(names(FIT_PARAMS), function(s) {
  p <- FIT_PARAMS[[s]]
  data.frame(
    num_tokens  = n_seq,
    fitted_dist = p$c - p$a * n_seq^(-p$b),
    strategy    = factor(STRATEGY_LABELS[s], levels = STRATEGY_LABELS),
    r2          = p$r2,
    asymptote   = p$c
  )
}))

# Asymptote lines (only show if c < 100)
asym_df <- fit_df %>%
  distinct(strategy, asymptote) %>%
  filter(asymptote < 99.99)

# R² + asymptote annotation
r2_labels <- fit_df %>%
  distinct(strategy, r2, asymptote) %>%
  mutate(
    label = ifelse(
      asymptote >= 99.99,
      sprintf("R\u00b2 = %.4f", r2),
      sprintf("R\u00b2 = %.4f,  c = %.1f%%", r2, asymptote)
    )
  )

cat(sprintf("[info] %d actual rows, %d fitted points\n", nrow(df), nrow(fit_df)))

# -----------------------------------------------------------------------------
# THEME
# -----------------------------------------------------------------------------
fintech_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    plot.title    = element_blank(),
    plot.subtitle = element_blank(),

    axis.title   = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text    = element_text(color = "grey30"),

    panel.grid.major = element_line(color = "grey92", linetype = "dashed"),
    panel.grid.minor = element_blank(),

    strip.text       = element_text(face = "bold", size = FONT_STRIP, color = "grey20"),
    strip.background = element_blank(),

    legend.position  = "bottom",
    legend.title     = element_blank(),
    legend.text      = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),
    legend.key.width = unit(1.8, "cm"),

    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),

    plot.margin = margin(10, 15, 10, 10)
  )

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
cat("[info] Building plot...\n")

p <- ggplot() +

  # Asymptote horizontal line (only for c < 100)
  geom_hline(
    data = asym_df,
    aes(yintercept = asymptote),
    linetype = "dotted", color = "grey60", linewidth = 0.5
  ) +

  # Actual data: dots + solid line
  geom_point(
    data = df,
    aes(x = num_tokens, y = mean_dist, color = strategy),
    size = 1.4, alpha = 0.5
  ) +
  geom_line(
    data = df,
    aes(x = num_tokens, y = mean_dist, color = strategy, linetype = "Actual"),
    linewidth = 0.6
  ) +

  # Fitted curve: dashed line
  geom_line(
    data = fit_df,
    aes(x = num_tokens, y = fitted_dist, color = strategy, linetype = "Fitted"),
    linewidth = 1.0
  ) +

  # R² + asymptote annotation
  geom_text(
    data = r2_labels,
    aes(label = label),
    x = 2, y = 95,
    hjust = 0, vjust = 1,
    size = 3.5, color = "grey35", fontface = "italic"
  ) +

  # Facet
  facet_wrap(~ strategy, ncol = 3) +

  # Scales
  scale_color_manual(values = STRATEGY_COLORS, guide = "none") +

  scale_linetype_manual(
    name   = NULL,
    values = c("Actual" = "solid", "Fitted" = "dashed"),
    guide  = guide_legend(
      override.aes = list(
        linewidth = c(0.6, 1.0),
        color     = c("grey30", "grey30")
      ))
  ) +

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

  labs(
    x = "Number of tokens (log scale)",
    y = "Mean L1 distance (%)",
    caption = expression(paste(
      "Dashed: fitted  ", d(n) == c - a %.% n^{-b},
      ",   c \u2264 100%;   dotted: asymptote c"
    ))
  ) +

  fintech_theme +

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