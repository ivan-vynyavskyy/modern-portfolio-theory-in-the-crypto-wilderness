#!/usr/bin/env Rscript
# =============================================================================
# plot_wealth_accounts_split.R
#
# Description:
#   Stacked-area plot of wealth-bucket share over time, faceted by wallet
#   type (EOA / CA). Two variants are supported: a single unified area
#   ('area') or a low/high split into two separate PDFs ('area_split',
#   the default).
#
# Input:
#   CSV with columns: block, wallet_type, bucket, wallet_pct.
#
# Output:
#   PDF(s) saved to the path given via --out_pdf. In 'area_split' mode two
#   files are produced with suffixes '_low.pdf' and '_high.pdf'.
#
# Usage:
#   Rscript plot_wealth_accounts_split.R \
#     --csv ../data/value_buckets_minimal.csv \
#     --out_pdf wealth_accounts_split.pdf
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(stringr)
  library(lubridate)
  library(ggplot2)
  library(scales)
})

# -----------------------------------------------------------------------------
# PALETTE
# -----------------------------------------------------------------------------
PALETTE_ALL <- c(
  "0-1"      = "#264653",
  "1-100"    = "#2A9D8F",
  "100-1K"   = "#8AB17D",
  "1K-10K"   = "#E9C46A",
  "10K-100K" = "#F4A261",
  ">100K"    = "#E76F51"
)

# -----------------------------------------------------------------------------
# CLI OPTIONS
# -----------------------------------------------------------------------------
option_list <- list(
  make_option(c("-i", "--csv"), type="character", help="Input CSV path (required)"),
  make_option(c("--plot_type"), type="character", default="area_split",
              help="Type: 'area_split' (default) or 'area' (unified)"),
  make_option(c("-o", "--out_pdf"), type="character", default=NULL,
              help="Output PDF path base name (e.g., 'fig.pdf' -> 'fig_low.pdf', 'fig_high.pdf')"),
  make_option(c("--include_all"), action="store_true", default=FALSE,
              help="Include 'ALL' wallet_type facet? (Default: Only EOA + CA)"),
  make_option(c("--start_date"), type="character", default=NULL,
              help="Filter start date (YYYY-MM-DD)"),
  make_option(c("--width"), type="double", default=12, help="Plot width (in)"),
  make_option(c("--height"), type="double", default=7, help="Plot height (in)"),
  make_option(c("--no_show"), action="store_true", default=FALSE,
              help="Do not print plot to stdout/window")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (is.null(opt$csv) || !file.exists(opt$csv)) stop("Error: Input CSV not found.")

plot_type <- tolower(trimws(opt$plot_type))
if (!plot_type %in% c("area", "area_split")) {
  stop("Invalid plot_type. Only 'area' and 'area_split' are supported.")
}

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
opt$width <- apply_plot_theme_fonts(opt$width)

# -----------------------------------------------------------------------------
# DATA LOADING & PREP
# -----------------------------------------------------------------------------
df <- read_csv(opt$csv, show_col_types = FALSE)

df <- df %>%
  mutate(
    block = as.integer(block),
    wallet_type = str_trim(as.character(wallet_type)),
    bucket = str_trim(as.character(bucket)),
    wallet_pct = suppressWarnings(as.numeric(wallet_pct))
  ) %>%
  filter(!is.na(block), !is.na(wallet_pct)) %>%
  left_join(b2d, by = "block")

if (!is.null(opt$start_date)) {
  start_dt <- ymd(opt$start_date)
  df <- df %>% filter(date >= start_dt)
}

if (!opt$include_all) {
  df <- df %>% filter(wallet_type %in% c("EOA", "CA"))
}

if (max(df$wallet_pct, na.rm=TRUE) <= 1.05) {
  df$wallet_pct <- df$wallet_pct * 100
}

df <- df %>%
  mutate(
    bucket_norm = bucket %>%
      str_replace_all("[\\s\\$]", "") %>%
      str_replace_all("[–—]", "-"),
    bucket_label = case_when(
      bucket_norm %in% c("0-1","0-1USD") ~ "0-1",
      bucket_norm %in% c("1-100","1-100USD") ~ "1-100",
      bucket_norm %in% c("100-1K","100-1000","100-1000USD") ~ "100-1K",
      bucket_norm %in% c("1K-10K","1000-10000","1K-10000") ~ "1K-10K",
      bucket_norm %in% c("10K-100K","10000-100000") ~ "10K-100K",
      bucket_norm %in% c(">100K","100K+","100000+") ~ ">100K",
      TRUE ~ bucket
    )
  )

level_order <- c("0-1", "1-100", "100-1K", "1K-10K", "10K-100K", ">100K")
df$bucket_label <- factor(df$bucket_label, levels = level_order, ordered = TRUE)
df <- df %>% arrange(date)

# -----------------------------------------------------------------------------
# THEME & X SCALE
# -----------------------------------------------------------------------------
fintech_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    axis.title    = element_text(size = FONT_AXIS_TITLE, color = "grey30"),
    axis.text     = element_text(size = FONT_AXIS_TEXT,  color = "grey30"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "grey90", linetype = "dashed"),
    legend.position = "bottom",
    legend.title    = element_text(size = FONT_LEGEND_TITLE, color = "black"),
    legend.text     = element_text(size = FONT_LEGEND_TEXT),
    plot.background = element_rect(fill = "white", color = NA),
    strip.text      = element_text(size = FONT_STRIP, face = "bold"),
    plot.margin     = margin(t = 10, r = 35, b = 10, l = 10)
  )

x_scale_forced <- scale_x_date(
  limits = c(min(df$date), as.Date("2025-12-31")),
  date_breaks = "6 months",
  date_labels = "%b '%y",
  expand = expansion(mult = c(0.01, 0.03))
)

# -----------------------------------------------------------------------------
# AREA GEOM
# The first factor level of `bucket_label` must render at the bottom of the
# stack, which requires `position_stack(reverse = TRUE)` under geom_area.
# -----------------------------------------------------------------------------
area_geom <- geom_area(
  alpha = 0.95,
  size = 0.1,
  color = "white",
  position = position_stack(reverse = TRUE)
)

# -----------------------------------------------------------------------------
# PLOT BUILDERS
# -----------------------------------------------------------------------------
make_area <- function(d) {
  ggplot(d, aes(x = date, y = wallet_pct, fill = bucket_label, group = bucket_label)) +
    area_geom +
    facet_wrap(~ wallet_type, ncol = 1, scales = "free_y") +
    scale_fill_manual(values = PALETTE_ALL, drop = FALSE, name = "Wealth bins [USD]") +
    scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.05))) +
    x_scale_forced +
    coord_cartesian(clip = "off") +
    labs(y = "Share (%)", x = NULL) +
    fintech_theme
}

make_area_split <- function(d,
                            low_cats = c("0-1", "1-100", "100-1K"),
                            high_cats = c("1K-10K", "10K-100K", ">100K")) {

  d_low  <- d %>% filter(bucket_label %in% low_cats)
  d_high <- d %>% filter(bucket_label %in% high_cats)

  d_low$bucket_label  <- droplevels(d_low$bucket_label)
  d_high$bucket_label <- droplevels(d_high$bucket_label)

  p_low <- ggplot(d_low, aes(x = date, y = wallet_pct, fill = bucket_label, group = bucket_label)) +
    area_geom +
    facet_wrap(~ wallet_type, ncol = 1) +
    scale_fill_manual(values = PALETTE_ALL, drop = TRUE, name = "Wealth bins [USD]") +
    scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.06))) +
    x_scale_forced +
    coord_cartesian(clip = "off") +
    labs(y = "Share (%)", x = NULL) +
    fintech_theme

  p_high <- ggplot(d_high, aes(x = date, y = wallet_pct, fill = bucket_label, group = bucket_label)) +
    area_geom +
    facet_wrap(~ wallet_type, ncol = 1) +
    scale_fill_manual(values = PALETTE_ALL, drop = TRUE, name = "Wealth bins [USD]") +
    scale_y_continuous(labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.06))) +
    x_scale_forced +
    coord_cartesian(clip = "off") +
    labs(y = "Share (%)", x = NULL) +
    fintech_theme

  list(low = p_low, high = p_high)
}

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------
plots <- list()
if (plot_type == "area") {
  plots$main <- make_area(df)
} else {
  plots <- make_area_split(df)
}

# -----------------------------------------------------------------------------
# SAVE (PDF ONLY)
# -----------------------------------------------------------------------------
if (!is.null(opt$out_pdf)) {
  if (plot_type == "area_split") {
    base <- sub("\\.pdf$", "", opt$out_pdf, ignore.case = TRUE)
    out_low  <- paste0(base, "_low.pdf")
    out_high <- paste0(base, "_high.pdf")

    ggsave(out_low,  plots$low,  width = opt$width, height = opt$height)
    ggsave(out_high, plots$high, width = opt$width, height = opt$height)
    cat(sprintf("[ok] Saved PDFs: %s and %s\n", out_low, out_high))
  } else {
    out_main <- opt$out_pdf
    if (!grepl("\\.pdf$", out_main, ignore.case = TRUE)) {
      out_main <- paste0(out_main, ".pdf")
    }
    ggsave(out_main, plots$main, width = opt$width, height = opt$height)
    cat(sprintf("[ok] Saved PDF: %s\n", out_main))
  }
}

# -----------------------------------------------------------------------------
# DISPLAY (Optional)
# -----------------------------------------------------------------------------
if (!opt$no_show && interactive()) {
  if (plot_type == "area_split") {
    print(plots$low)
    print(plots$high)
  } else {
    print(plots$main)
  }
}
