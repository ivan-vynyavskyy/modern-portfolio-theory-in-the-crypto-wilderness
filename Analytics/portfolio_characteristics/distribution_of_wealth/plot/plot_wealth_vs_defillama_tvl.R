#!/usr/bin/env Rscript
# =============================================================================
# plot_wealth_vs_defillama_tvl.R
#
# Description:
#   Two-line chart comparing total wallet-held ERC-20 value (EOA + CA) against
#   Ethereum DeFi TVL as reported by DeFiLlama, over the monthly snapshots.
#   Each series is shaded with a translucent area fill, and the final data
#   points are annotated with their value. Two-series variant of the paper
#   figure; the paper (Appendix B) uses plot_wealth_ca_defillama_tvl.R, which
#   adds the contract-held (CA) series on the same axis.
#
#   The DeFiLlama series is the external TVL reference. It is de-duplicated
#   across receipt/wrapper tokens and restricted to DeFi protocols, so it is
#   deliberately much smaller than our contract-held aggregate.
#
# Input:
#   CSV with columns: block, wallet_type, value_sum_usd
#   (default: ../data/value_buckets_minimal.csv).
#   DeFiLlama TVL cache CSV with columns: date, tvl_usd
#   (default: ../data/defillama_eth_tvl.csv; fetched and written if missing).
#
# Output:
#   Single PDF (default: ../graphics/wealth_vs_defillama_tvl.pdf).
#
# Usage:
#   Rscript plot_wealth_vs_defillama_tvl.R \
#     --in-csv  ../data/value_buckets_minimal.csv \
#     --tvl-csv ../data/defillama_eth_tvl.csv \
#     --out-pdf ../graphics/wealth_vs_defillama_tvl.pdf
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
DEFAULT_TVL_CSV <- "../data/defillama_eth_tvl.csv"
DEFAULT_OUT_PDF <- "../graphics/wealth_vs_defillama_tvl.pdf"
DEFAULT_WIDTH   <- 10
DEFAULT_HEIGHT  <- 4

# Brewer Paired palette: dark = lines, light = area fills.
COLOR_WEALTH      <- "#1F78B4"
COLOR_TVL         <- "#33A02C"
COLOR_WEALTH_FILL <- "#A6CEE3"
COLOR_TVL_FILL    <- "#B2DF8A"

SERIES_WEALTH <- "Total Wealth (EOA + CA)"
SERIES_TVL    <- "Ethereum DeFi TVL (DefiLlama)"

# -----------------------------------------------------------------------------
# DATA MAPPING (Block -> Date) and TVL loader, sourced from shared files
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
source(file.path(script_dir, "load_defillama_tvl.R"))
source(file.path(script_dir, "..", "..", "..", "shared", "plot_theme.R"))

# -----------------------------------------------------------------------------
# CLI SETUP
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("-i", "--in-csv"),  type = "character", default = DEFAULT_IN_CSV,
              help = "Input CSV path"),
  make_option(c("-t", "--tvl-csv"), type = "character", default = DEFAULT_TVL_CSV,
              help = "DeFiLlama TVL cache CSV path (read if present, else fetched and written)"),
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
  mutate(series = SERIES_WEALTH)

# DeFiLlama Ethereum TVL, joined to the snapshot dates present in the input
tvl_all <- load_defillama_tvl(cache_csv = opt$`tvl-csv`)
df_tvl <- join_defillama_tvl(sort(unique(df_wealth$date)), tvl_all) %>%
  filter(!is.na(tvl_usd)) %>%
  transmute(date, value = tvl_usd, series = SERIES_TVL)

# Fixed series order (legend + draw order): wealth first, TVL reference second
df_lines <- bind_rows(df_wealth, df_tvl) %>%
  mutate(series = factor(series, levels = c(SERIES_WEALTH, SERIES_TVL)))

# Wide format for shaded area fills
df_wide <- df_lines %>%
  tidyr::pivot_wider(names_from = series, values_from = value) %>%
  rename(wealth = all_of(SERIES_WEALTH), tvl = all_of(SERIES_TVL))

# End-point labels
fmt_usd_si <- function(x) paste0("$", scales::label_number(
  accuracy = 0.1, scale_cut = scales::cut_short_scale())(x))

last_points <- df_lines %>%
  group_by(series) %>%
  filter(date == max(date)) %>%
  ungroup() %>%
  mutate(label = fmt_usd_si(value))

# Console summary (peak and endpoint per series) for cross-checking
summary_tbl <- df_lines %>%
  group_by(series) %>%
  summarise(
    peak_date  = date[which.max(value)],
    peak_value = fmt_usd_si(max(value)),
    end_date   = max(date),
    end_value  = fmt_usd_si(value[which.max(date)]),
    .groups    = "drop"
  )
print(as.data.frame(summary_tbl), row.names = FALSE)

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
cat("Generating plot...\n")

series_colors <- setNames(c(COLOR_WEALTH, COLOR_TVL), c(SERIES_WEALTH, SERIES_TVL))
series_lty    <- setNames(c("solid", "solid"),        c(SERIES_WEALTH, SERIES_TVL))

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
cat("[saved]", opt$`out-pdf`, "\n")
