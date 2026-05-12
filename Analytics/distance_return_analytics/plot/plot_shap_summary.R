#!/usr/bin/env Rscript
# =============================================================================
# plot_shap_summary.R
#
# Description:
#   SHAP summary (beeswarm) plot with points coloured by feature
#   value. Vertical line at zero.
#
# Input:
#   rf_shap_values_baseline.csv
#
# Output:
#   Single PDF (default: ../graphics/shap_summary_baseline.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(optparse)
  library(scales)
})

option_list <- list(
  make_option("--in-csv",  type = "character",
              default = "../data/rf_shap_values_baseline.csv"),
  make_option("--out-pdf", type = "character", default = "../graphics/shap_summary_baseline.pdf"),
  make_option("--width",   type = "double", default = 10),
  make_option("--height",  type = "double", default = 4.5)
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

features <- c("month", "log_value_usd", "num_tokens",
              "beta_baseline", "holds_eth", "holds_btc")
feat_display <- c("Entry month", "Log portfolio value", "Token count",
                  "Portfolio beta", "Holds ETH wrapper", "Holds BTC wrapper")

# Reshape to long
shap_long <- data.frame()
for (i in seq_along(features)) {
  feat <- features[i]
  feat_col <- paste0("feat_", feat)
  tmp <- data.frame(
    feature    = feat_display[i],
    shap_value = df[[feat]],
    feat_value = df[[feat_col]]
  )
  shap_long <- rbind(shap_long, tmp)
}

# Normalise feature values to [0, 1] per feature for colour
shap_long <- shap_long %>%
  group_by(feature) %>%
  mutate(feat_norm = (feat_value - min(feat_value, na.rm = TRUE)) /
           max(1e-12, max(feat_value, na.rm = TRUE) - min(feat_value, na.rm = TRUE))) %>%
  ungroup()

# Order by mean |SHAP| (bottom = least important, top = most)
feat_order <- shap_long %>%
  group_by(feature) %>%
  summarise(mean_abs = mean(abs(shap_value))) %>%
  arrange(mean_abs) %>%
  pull(feature)
shap_long$feature <- factor(shap_long$feature, levels = feat_order)

# Subsample for plotting (3K per feature = 18K total)
set.seed(42)
shap_plot <- shap_long %>%
  group_by(feature) %>%
  slice_sample(n = 3000) %>%
  ungroup()

p <- ggplot(shap_plot, aes(x = shap_value, y = feature, colour = feat_norm)) +
  geom_vline(xintercept = 0, colour = "grey40", linewidth = 0.4) +
  geom_jitter(height = 0.25, width = 0, size = 0.35, alpha = 0.5) +
  scale_colour_gradient2(
    low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
    midpoint = 0.5,
    name = "Feature\nvalue",
    labels = c("Low", "", "High"),
    breaks = c(0, 0.5, 1)
  ) +
  scale_x_continuous(labels = function(x) paste0(x, " pp")) +
  labs(
    x = "SHAP value (impact on predicted return)",
    y = NULL
  ) +
  theme_minimal(base_size = FONT_BASE) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey85", linetype = "dashed"),
    panel.grid.minor.x = element_blank(),
    legend.position    = "right",
    legend.key.height  = unit(1.5, "cm"),
    legend.key.width   = unit(0.3, "cm"),
    axis.text.y        = element_text(size = FONT_AXIS_TEXT)
  )

ggsave(opts$`out-pdf`, p, width = opts$width, height = opts$height)
cat("[saved]", opts$`out-pdf`, "\n")