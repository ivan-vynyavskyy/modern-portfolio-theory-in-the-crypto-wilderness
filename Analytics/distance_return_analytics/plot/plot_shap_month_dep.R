#!/usr/bin/env Rscript
# =============================================================================
# plot_shap_month_dep.R
#
# Description:
#   SHAP dependence plot for entry month: which months predict
#   positive vs negative returns. The curve traces the boom-bust
#   cycle.
#
# Input:
#   rf_shap_values_baseline.csv
#
# Output:
#   Single PDF (default: ../graphics/shap_month_dep_baseline.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(optparse)
  library(scales)
})

option_list <- list(
  make_option("--in-csv",  type = "character",
              default = "../data/rf_shap_data/rf_shap_values_baseline.csv"),
  make_option("--out-pdf", type = "character", default = "../graphics/shap_month_dep_baseline.pdf"),
  make_option("--width",   type = "double", default = 10),
  make_option("--height",  type = "double", default = 5)
)
opts <- parse_args(OptionParser(option_list = option_list))


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
opts$width <- apply_plot_theme_fonts(opts$width)
df <- read.csv(opts$`in-csv`)

# month integer → calendar date for x-axis
df$month_int <- df$feat_month
df$month_date <- as.Date(paste0(
  2020 + (df$month_int - 1) %/% 12, "-",
  sprintf("%02d", ((df$month_int - 1) %% 12) + 1), "-01"
))
df$shap_month <- df$month  # SHAP value for month feature

# Subsample for plotting
set.seed(42)
df_plot <- df %>% slice_sample(n = min(5000, nrow(df)))

# Compute per-month median SHAP for a smooth trend line
monthly_median <- df %>%
  group_by(month_date) %>%
  summarise(
    median_shap = median(shap_month, na.rm = TRUE),
    .groups = "drop"
  )

p <- ggplot() +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4, linetype = "dashed") +
  # Individual points (jittered slightly for visibility)
  geom_point(data = df_plot,
             aes(x = month_date, y = shap_month),
             colour = "#1F78B4", size = 0.5, alpha = 0.15) +
  # Monthly median trend
  geom_line(data = monthly_median,
            aes(x = month_date, y = median_shap),
            colour = "#E31A1C", linewidth = 0.8) +
  geom_point(data = monthly_median,
             aes(x = month_date, y = median_shap),
             colour = "#E31A1C", size = 1.8) +
  scale_x_date(
    date_breaks = "6 months",
    date_labels = "%b %Y",
    expand = expansion(mult = 0.02)
  ) +
  scale_y_continuous(labels = function(x) paste0(x, " pp")) +
  labs(
    x = NULL,
    y = "SHAP value (impact on predicted return)"
  ) +
  theme_minimal(base_size = FONT_BASE) +
  theme(
    panel.grid.major.y = element_line(colour = "grey85", linetype = "dashed"),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.x        = element_text(angle = 45, hjust = 1)
  )

ggsave(opts$`out-pdf`, p, width = opts$width, height = opts$height)
cat("[saved]", opts$`out-pdf`, "\n")