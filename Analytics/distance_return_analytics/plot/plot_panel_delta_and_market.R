#!/usr/bin/env Rscript
# =============================================================================
# plot_panel_delta_and_market.R
#
# Description:
#   Combined two-panel figure. Top: delta (strategy - market) per
#   monthly snapshot. Bottom: median returns with ETH price overlay
#   on the right axis.
#
# Input:
#   CSV with per-block return statistics + ETH price CSV.
#
# Output:
#   Single PDF (default: ../graphics/returns_panel_delta_and_market.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(optparse)
  library(scales)
  library(zoo)
  library(patchwork)
})

option_list <- list(
  make_option("--in-csv",     type = "character", default = "../data/camp_analysis_results_full/returns_per_block.csv"),
  make_option("--price-csv",  type = "character", default = "../data/weth_price.csv"),
  make_option("--out-pdf",    type = "character", default = "../graphics/returns_panel_delta_and_market.pdf"),
  make_option("--width",      type = "double", default = 10),
  make_option("--height",     type = "double", default = 8)
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
# -----------------------------------------------------------------------------
# LOAD DATA
df <- read.csv(opts$`in-csv`)
df$block_date <- as.Date(df$block_date)

# -----------------------------------------------------------------------------
# DELTAS FOR TOP PANEL
delta_all <- df %>%
  transmute(
    block_date,
    `Baseline (actual)` = (median_ret_baseline - median_market_return) * 100,
    `MaxRet`            = (median_ret_better_return - median_market_return) * 100,
    `MinVar`            = (median_ret_safer_risk - median_market_return) * 100,
    `MaxSR`             = (median_ret_max_sharpe - median_market_return) * 100,
    `Equal-weight`      = (median_ret_equal_weight - median_market_return) * 100,
    `MCap-weight`       = (median_ret_mcap_weight - median_market_return) * 100
  ) %>%
  pivot_longer(-block_date, names_to = "strategy", values_to = "delta_pct") %>%
  mutate(strategy = factor(strategy, levels = c(
    "Baseline (actual)", "MaxRet", "MinVar",
    "MaxSR", "Equal-weight", "MCap-weight"
  )))

d_baseline <- delta_all %>% filter(strategy == "Baseline (actual)")
d_strat    <- delta_all %>% filter(strategy != "Baseline (actual)")

# -----------------------------------------------------------------------------
# BOTTOM PANEL: ONLY MARKET + BASELINE
ret_bottom <- df %>%
  transmute(
    block_date,
    `Market benchmark`  = median_market_return * 100,
    `Baseline (actual)` = median_ret_baseline * 100
  ) %>%
  pivot_longer(-block_date, names_to = "strategy", values_to = "return_pct") %>%
  mutate(strategy = factor(strategy, levels = c(
    "Market benchmark", "Baseline (actual)"
  )))

# -----------------------------------------------------------------------------
# ETH PRICE (60-DAY MA)
price <- read.csv(opts$`price-csv`)
price$date <- as.Date(price$date)
price <- price %>%
  arrange(date) %>%
  mutate(ma60 = zoo::rollmean(daily_close_usd, k = 60, fill = NA, align = "right"))

block_dates <- unique(df$block_date)
price_monthly <- data.frame(block_date = block_dates) %>%
  rowwise() %>%
  mutate(eth_price = {
    diffs <- abs(as.numeric(price$date - block_date))
    idx <- which.min(diffs)
    if (diffs[idx] <= 15) price$ma60[idx] else NA_real_
  }) %>%
  ungroup() %>%
  filter(!is.na(eth_price))

# Scale ETH price to bottom panel y-axis
ret_range   <- range(ret_bottom$return_pct, na.rm = TRUE)
price_range <- range(price_monthly$eth_price, na.rm = TRUE)
scale_factor <- diff(ret_range) / diff(price_range)
offset       <- ret_range[1] - price_range[1] * scale_factor
price_monthly <- price_monthly %>%
  mutate(price_scaled = eth_price * scale_factor + offset)

# -----------------------------------------------------------------------------
# SHARED
shared_x <- scale_x_date(
  date_breaks = "6 months",
  date_labels = "%b %Y",
  expand = expansion(mult = 0.02)
)

base_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    panel.grid.major.y = element_line(colour = "grey85", linetype = "dashed"),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    legend.text        = element_text(size = FONT_LEGEND_TEXT)
  )

# -----------------------------------------------------------------------------
# TOP PANEL
pal_top <- c(
  "Baseline (actual)" = "#E31A1C",
  "MaxRet"            = "#1F78B4",
  "MinVar"            = "#33A02C",
  "MaxSR"             = "#FF7F00",
  "Equal-weight"      = "#6A3D9A",
  "MCap-weight"       = "#B15928"
)

p_top <- ggplot() +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4, linetype = "dashed") +
  geom_line(data = d_baseline,
            aes(x = block_date, y = delta_pct, colour = strategy),
            linewidth = 0.7, alpha = 0.85) +
  geom_point(data = d_strat,
             aes(x = block_date, y = delta_pct,
                 colour = strategy, shape = strategy),
             size = 1.8, alpha = 0.7) +
  scale_colour_manual(
    values = pal_top,
    guide = guide_legend(
      nrow = 1,
      override.aes = list(
        linetype  = c("solid", rep("blank", 5)),
        shape     = c(NA, 17, 15, 18, 1, 2),
        linewidth = c(0.7, rep(0, 5)),
        size      = c(0, rep(1.8, 5))
      )
    )
  ) +
  scale_shape_manual(
    values = c("MaxRet" = 17, "MinVar" = 15, "MaxSR" = 18,
               "Equal-weight" = 1, "MCap-weight" = 2),
    guide = "none"
  ) +
  shared_x +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(x = NULL, y = "Strategy minus market (%)", colour = NULL) +
  base_theme +
  theme(
    legend.position = "bottom",
    axis.text.x     = element_blank(),
    axis.ticks.x    = element_blank()
  )

# -----------------------------------------------------------------------------
# BOTTOM PANEL
pal_bottom <- c(
  "Market benchmark"  = "#222222",
  "Baseline (actual)" = "#E31A1C"
)
lty_bottom <- c(
  "Market benchmark"  = "dashed",
  "Baseline (actual)" = "solid"
)
shp_bottom <- c(
  "Market benchmark"  = 4,
  "Baseline (actual)" = 16
)

p_bottom <- ggplot() +
  geom_area(data = price_monthly,
            aes(x = block_date, y = price_scaled),
            fill = "grey85", alpha = 0.5) +
  geom_line(data = price_monthly,
            aes(x = block_date, y = price_scaled),
            colour = "grey50", linewidth = 0.4) +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4) +
  geom_line(data = ret_bottom,
            aes(x = block_date, y = return_pct,
                colour = strategy, linetype = strategy),
            linewidth = 0.6) +
  geom_point(data = ret_bottom,
             aes(x = block_date, y = return_pct,
                 colour = strategy, shape = strategy),
             size = 1.4, alpha = 0.8) +
  scale_colour_manual(values = pal_bottom) +
  scale_linetype_manual(values = lty_bottom) +
  scale_shape_manual(values = shp_bottom) +
  shared_x +
  scale_y_continuous(
    name   = "Median 20-day return (%)",
    labels = function(x) paste0(x, "%"),
    sec.axis = sec_axis(
      ~ (. - offset) / scale_factor,
      name   = "wETH price — 60-day MA (USD)",
      labels = dollar_format(prefix = "$", big.mark = ",")
    )
  ) +
  labs(x = NULL, colour = NULL, linetype = NULL, shape = NULL) +
  base_theme +
  theme(
    legend.position    = "bottom",
    axis.text.x        = element_text(angle = 45, hjust = 1),
    axis.title.y.right = element_text(colour = "grey50"),
    axis.text.y.right  = element_text(colour = "grey50")
  )

# -----------------------------------------------------------------------------
# COMBINE
p_combined <- p_top / p_bottom +
  plot_layout(heights = c(1, 1.3))

ggsave(opts$`out-pdf`, p_combined, width = opts$width, height = opts$height)
cat("[saved]", opts$`out-pdf`, "\n")
