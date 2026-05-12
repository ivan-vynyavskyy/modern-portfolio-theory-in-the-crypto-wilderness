#!/usr/bin/env Rscript
# =============================================================================
# plot_tokens_vs_distance_combined.R
#
# Single clean plot: 3 mean lines (one per strategy), direct labels.
# X = Number of tokens (log scale), Y = Mean L1 distance (%)
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
DEFAULT_OUT_PDF <- "../graphics/tokens_vs_distance_combined.pdf"

STRATEGY_LABELS <- c(
  "safer_risk"    = "Min Variance",
  "better_return" = "Max Return",
  "max_sharpe"    = "Max Sharpe"
)

STRATEGY_COLORS <- c(
  "Min Variance" = "#33A02C",
  "Max Return"   = "#1F78B4",
  "Max Sharpe"   = "#E31A1C"
)

STRATEGY_LINETYPES <- c(
  "Min Variance" = "solid",
  "Max Return"   = "dashed",
  "Max Sharpe"   = "dotted"
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
  make_option(c("--max-tokens"), type = "integer", default = MAX_TOKENS,
              help = "Cap token count display [default: %default]"),
  make_option(c("--width"),  type = "double", default = 10,
              help = "Plot width in inches [default: %default]"),
  make_option(c("--height"), type = "double", default = 4,
              help = "Plot height in inches [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))
if (!file.exists(opt$`in-csv`)) stop("Input file not found: ", opt$`in-csv`)

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

# Endpoint labels: pick the rightmost point (max tokens) per strategy
label_df <- df %>%
  group_by(strategy) %>%
  filter(num_tokens == max(num_tokens)) %>%
  ungroup()

cat(sprintf("[info] %d rows, token range %d-%d, %d strategies\n",
            nrow(df), min(df$num_tokens), max(df$num_tokens),
            n_distinct(df$strategy)))

# -----------------------------------------------------------------------------
# THEME
# -----------------------------------------------------------------------------
fintech_theme <- theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_blank(),
    plot.subtitle = element_blank(),

    axis.title   = element_text(size = 12, color = "grey30"),
    axis.text    = element_text(color = "grey30"),

    panel.grid.major = element_line(color = "grey92", linetype = "dashed"),
    panel.grid.minor = element_blank(),

    legend.position  = "none",

    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),

    plot.margin = margin(10, 80, 10, 10)  # extra right margin for labels
  )

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
cat("[info] Building plot...\n")

p <- ggplot(df, aes(x = num_tokens, y = mean_dist, color = strategy,
                    linetype = strategy)) +

  # Mean lines
  geom_line(linewidth = 1.1) +

  # Small dots at each data point
  geom_point(size = 1.2, alpha = 0.5) +

  # Direct labels at right endpoints of each curve
  geom_text(
    data = label_df,
    aes(label = strategy),
    hjust = 0, nudge_x = 0.03,   # nudge in log-space
    size = 3.8, fontface = "bold",
    show.legend = FALSE
  ) +

  scale_color_manual(values = STRATEGY_COLORS) +
  scale_linetype_manual(values = STRATEGY_LINETYPES) +

  # x-axis: token count (log scale)
  scale_x_log10(
    breaks = c(2, 3, 5, 10, 20, 50),
    labels = c("2", "3", "5", "10", "20", "50"),
    expand = expansion(mult = c(0.02, 0.02))
  ) +

  # y-axis: L1 distance (%)
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, by = 10),
    labels = function(y) paste0(round(y, 0), "%"),
    expand = expansion(mult = c(0.02, 0.02))
  ) +

  labs(
    x = "Number of tokens (log scale)",
    y = "Mean distance (%)"
  ) +

  coord_cartesian(clip = "off") +

  fintech_theme

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
out_file <- opt$`out-pdf`
if (!grepl("\\.pdf$", out_file, ignore.case = TRUE)) out_file <- paste0(out_file, ".pdf")
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

ggsave(out_file, plot = p, width = opt$width, height = opt$height, device = "pdf")
cat(sprintf("[ok] Saved PDF: %s\n", out_file))