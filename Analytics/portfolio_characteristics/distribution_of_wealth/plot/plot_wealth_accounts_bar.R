#!/usr/bin/env Rscript
# =============================================================================
# plot_wealth_accounts_bar.R
#
# Description:
#   Horizontal stacked bar chart showing average wealth-bucket
#   composition for EOA and CA wallet types.
#
# Input:
#   CSV with columns: block, wallet_type, bucket, wallet_pct
#   (same as plot_wealth_accounts_split.R).
#
# Output:
#   Single PDF (default: ../graphics/wealth_accounts_bar.pdf).
#
# Usage:
#   Rscript plot_wealth_accounts_bar.R \
#     --csv ../data/value_buckets_minimal.csv \
#     --out_pdf ../graphics/wealth_accounts_bar.pdf
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(stringr)
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
  make_option(c("-i", "--csv"), type = "character",
              help = "Input CSV path (required)"),
  make_option(c("-o", "--out_pdf"), type = "character",
              default = "../graphics/wealth_accounts_bar.pdf",
              help = "Output PDF path [default: %default]"),
  make_option(c("--width"),  type = "double", default = 10,
              help = "Plot width in inches [default: %default]"),
  make_option(c("--height"), type = "double", default = 3,
              help = "Plot height in inches [default: %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$csv) || !file.exists(opt$csv)) stop("Error: Input CSV not found.")

# -----------------------------------------------------------------------------
# SHARED THEME
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

# -----------------------------------------------------------------------------
# DATA LOADING & PREP
# -----------------------------------------------------------------------------
cat("[info] Loading data...\n")
df <- read_csv(opt$csv, show_col_types = FALSE) %>%
  mutate(
    block       = as.integer(block),
    wallet_type = str_trim(as.character(wallet_type)),
    bucket      = str_trim(as.character(bucket)),
    wallet_pct  = suppressWarnings(as.numeric(wallet_pct))
  ) %>%
  filter(!is.na(block), !is.na(wallet_pct),
         wallet_type %in% c("EOA", "CA"))

# Normalise to percentage if stored as fraction
if (max(df$wallet_pct, na.rm = TRUE) <= 1.05) {
  df$wallet_pct <- df$wallet_pct * 100
}

# Standardise bucket labels
df <- df %>%
  mutate(
    bucket_norm = bucket %>%
      str_replace_all("[\\s\\$]", "") %>%
      str_replace_all("[–—]", "-"),
    bucket_label = case_when(
      bucket_norm %in% c("0-1", "0-1USD")               ~ "0-1",
      bucket_norm %in% c("1-100", "1-100USD")            ~ "1-100",
      bucket_norm %in% c("100-1K", "100-1000", "100-1000USD") ~ "100-1K",
      bucket_norm %in% c("1K-10K", "1000-10000", "1K-10000")  ~ "1K-10K",
      bucket_norm %in% c("10K-100K", "10000-100000")     ~ "10K-100K",
      bucket_norm %in% c(">100K", "100K+", "100000+")    ~ ">100K",
      TRUE ~ bucket
    )
  )

level_order <- c("0-1", "1-100", "100-1K", "1K-10K", "10K-100K", ">100K")
df$bucket_label <- factor(df$bucket_label, levels = level_order, ordered = TRUE)

# Average across all blocks
df_avg <- df %>%
  group_by(wallet_type, bucket_label) %>%
  summarise(
    mean_pct = mean(wallet_pct, na.rm = TRUE),
    .groups  = "drop"
  )

cat(sprintf("[info] %d rows after averaging\n", nrow(df_avg)))

# Compute cumulative x positions for label placement
df_avg <- df_avg %>%
  mutate(pct_label = paste0(round(mean_pct, 1), "%")) %>%
  group_by(wallet_type) %>%
  arrange(bucket_label) %>%
  mutate(
    cum_pct  = cumsum(mean_pct),
    x_center = cum_pct - mean_pct / 2,
    is_small = mean_pct < 3
  ) %>%
  ungroup()

# Small segments: labels above the bar
# 10K-100K: angled line nudged left; >100K: straight vertical
df_small <- df_avg %>%
  filter(is_small) %>%
  mutate(
    y_num   = as.numeric(factor(wallet_type, levels = c("CA", "EOA"))),
    y_bar   = y_num + 0.32,
    y_label = y_num + 0.42,
    x_label = ifelse(bucket_label == "10K-100K", x_center - 3, x_center)
  )

df_large <- df_avg %>% filter(!is_small)

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
cat("[info] Building plot...\n")

p <- ggplot(df_avg, aes(x = mean_pct, y = wallet_type,
                        fill = bucket_label, group = bucket_label)) +

  geom_bar(stat = "identity", position = position_stack(reverse = TRUE),
           width = 0.6, color = "white", linewidth = 0.3) +

  # Labels inside large segments
  geom_text(
    data = df_large,
    aes(x = x_center, label = pct_label),
    size = 3.2, color = "white", fontface = "bold"
  ) +

  # Leader lines from small segments upward
  geom_segment(
    data = df_small,
    aes(x = x_center, xend = x_label,
        y = y_bar, yend = y_label),
    color = "grey50", linewidth = 0.3,
    inherit.aes = FALSE
  ) +

  # Labels above small segments
  geom_text(
    data = df_small,
    aes(x = x_label, y = y_label, label = pct_label, color = bucket_label),
    size = 2.8, fontface = "bold", vjust = -0.3,
    inherit.aes = FALSE, show.legend = FALSE
  ) +

  scale_fill_manual(values = PALETTE_ALL, drop = FALSE,
                    name = "Wealth bins [USD]") +
  scale_color_manual(values = PALETTE_ALL, guide = "none") +

  scale_x_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, by = 20),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.04))
  ) +

  labs(x = "Average share (%)", y = NULL) +

  coord_cartesian(clip = "off") +

  guides(fill = guide_legend(nrow = 1)) +

  theme_minimal(base_size = FONT_BASE) +
  theme(
    axis.title.x   = element_text(size = FONT_AXIS_TITLE, color = "grey30",
                                  margin = margin(t = 8, b = 4)),
    axis.text.y    = element_text(size = FONT_AXIS_TEXT, color = "grey30",
                                  face = "bold"),
    axis.text.x    = element_text(size = FONT_AXIS_TEXT, color = "grey40"),

    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_line(color = "grey92", linetype = "dashed"),

    legend.position   = "bottom",
    legend.title      = element_text(size = FONT_LEGEND_TITLE, color = "black"),
    legend.text       = element_text(size = FONT_LEGEND_TEXT),
    legend.margin     = margin(t = 0, b = 0),
    legend.box.margin = margin(t = 2, b = 0),
    legend.box.spacing = unit(8, "pt"),

    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin      = margin(25, 10, 10, 10)
  )

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
out_file <- opt$out_pdf
if (!grepl("\\.pdf$", out_file, ignore.case = TRUE)) {
  out_file <- paste0(out_file, ".pdf")
}
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

ggsave(out_file, plot = p, width = opt$width, height = opt$height, device = "pdf")
cat(sprintf("[ok] Saved PDF: %s\n", out_file))