#!/usr/bin/env Rscript
# =============================================================================
# plot_portfolio_size_stacked_by_type.R
#
# Description:
#   Same portfolio-size vs wealth-category stacked bar chart as
#   plot_size_stats_by_wealth.R, but split by wallet type (EOA vs CA).
#   Produces three PDFs: EOA only, CA only, and a combined faceted view.
#
# Input:
#   CSV with columns: wallet_type, wealth_bucket, size_bucket, wallet_pct
#   (default: ../data/portfolio_size_hist_by_wealth.csv).
#
# Output:
#   Three PDFs suffixed _EOA, _CA and _Combined (base path set via
#   --out_pdf, default: ../graphics/portfolio_size_stacked.pdf).
#
# Usage:
#   RScript plot_portfolio_size_stacked_by_type.R \
#     --input ../data/portfolio_size_hist_by_wealth.csv \
#     --out_pdf ../graphics/portfolio_size_stacked.pdf
# =============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(viridis)
  library(optparse)
  library(tools)
})

# -----------------------------------------------------------------------------
# CLI OPTIONS
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("-i", "--input"), type="character", default="../data/portfolio_size_hist_by_wealth.csv",
              help="Input CSV path [default %default]"),
  make_option(c("-o", "--out_pdf"), type="character", default="../graphics/portfolio_size_stacked.pdf",
              help="Base Output PDF path. Script appends _EOA, _CA, _Combined [default %default]"),
  make_option(c("--width"), type="double", default=10, help="Plot width (in)"),
  make_option(c("--height"), type="double", default=6, help="Plot height (in)")
)

opt <- parse_args(OptionParser(option_list=option_list))

# -----------------------------------------------------------------------------
# LOAD & PROCESS
# -----------------------------------------------------------------------------
if (!file.exists(opt$input)) stop(paste("Input file not found:", opt$input))

df_hist <- read_csv(opt$input, show_col_types = FALSE)

wealth_levels <- c("0–1","1–100","100–1,000","1,000–10,000","10,000–100,000",">100,000")
wealth_labels <- c(
  "0–1"            = "$0–$1",
  "1–100"          = "$1–$100",
  "100–1,000"      = "$100–$1,000",
  "1,000–10,000"   = "$1,000–$10,000",
  "10,000–100,000" = "$10,000–$100,000",
  ">100,000"       = ">$100,000"
)

size_levels <- c("1","2","3","4","5","6–10","11–20","21–50","51–100",">100")

df_hist <- df_hist %>%
  mutate(
    wealth_bucket = factor(wealth_bucket, levels = wealth_levels),
    wealth_label  = factor(wealth_labels[as.character(wealth_bucket)],
                           levels = wealth_labels[wealth_levels]),
    size_bucket   = factor(size_bucket, levels = c("0", size_levels)),
    wallet_type   = factor(wallet_type)
  ) %>%
  filter(size_bucket != "0") %>%
  mutate(size_bucket = factor(as.character(size_bucket), levels = size_levels))

df_hist_avg <- df_hist %>%
  group_by(wallet_type, wealth_label, size_bucket) %>%
  summarise(wallet_pct = mean(wallet_pct, na.rm = TRUE), .groups = "drop")

# -----------------------------------------------------------------------------
# THEME
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
    axis.title    = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text     = element_text(size = FONT_AXIS_TEXT,  color = "grey30"),
    axis.line     = element_blank(),

    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "grey90", linetype = "dashed"),

    plot.background   = element_rect(fill = "white", color = NA),
    panel.background  = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA),

    legend.title = element_text(
      size = FONT_LEGEND_TITLE, color = "grey30", face = "bold"
    ),
    legend.text = element_text(size = FONT_LEGEND_TEXT, color = "grey30"),
    legend.position = "right",

    strip.text = element_text(
      size = FONT_STRIP, face = "bold", color = "grey20"
    ),
    strip.background = element_blank(),

    plot.margin = margin(10, 10, 10, 10),
    axis.text.x = element_text(margin = margin(t = 6))
  )

# -----------------------------------------------------------------------------
# PLOT BUILDER
# -----------------------------------------------------------------------------
generate_plot <- function(data, is_split = FALSE) {
  p <- ggplot(data, aes(x = wealth_label, y = wallet_pct, fill = size_bucket)) +
    geom_col(width = 0.70) +

    scale_y_continuous(
      labels = function(x) paste0(round(x, 0), "%"),
      expand = expansion(mult = c(0.01, 0.05))
    ) +

    # Viridis option C (plasma) gives good contrast across the 10 levels
    scale_fill_viridis_d(option = "C", end = 0.95, direction = -1) +

    scale_x_discrete(guide = guide_axis(n.dodge = 2)) +

    labs(
      x = "Wealth category (USD)",
      y = "Avg % of wallets",
      fill = "Portfolio size\n(# assets)"
    ) +
    fintech_theme

  if (is_split) {
    p <- p + facet_wrap(~wallet_type)
  }

  p
}

# -----------------------------------------------------------------------------
# BUILD & SAVE
# -----------------------------------------------------------------------------
p_eoa      <- generate_plot(df_hist_avg %>% filter(wallet_type == "EOA"), is_split = FALSE)
p_ca       <- generate_plot(df_hist_avg %>% filter(wallet_type == "CA"),  is_split = FALSE)
p_combined <- generate_plot(df_hist_avg, is_split = TRUE)

base_path <- file_path_sans_ext(opt$out_pdf)
out_dir   <- dirname(base_path)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

save_pdf <- function(plot_obj, suffix) {
  fname <- paste0(base_path, "_", suffix, ".pdf")
  ggsave(
    filename = fname,
    plot = plot_obj,
    width = opt$width,
    height = opt$height,
    device = "pdf"
  )
  cat(sprintf("[ok] Saved PDF: %s\n", fname))
}

save_pdf(p_eoa,      "EOA")
save_pdf(p_ca,       "CA")
save_pdf(p_combined, "Combined")