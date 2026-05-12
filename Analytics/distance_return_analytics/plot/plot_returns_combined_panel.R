#!/usr/bin/env Rscript
# =============================================================================
# plot_returns_combined_panel.R
#
# Description:
#   Combined two-panel figure:
#     Top    - Delta (strategy - market) as point plot with distinct
#              markers per strategy.
#     Bottom - Median returns + ETH price (60-day MA) on right axis.
#   Also produces a cumulative-delta variant.
#
# Input:
#   CSV with per-block return statistics + ETH price CSV.
#
# Output:
#   Two PDFs (default: ../graphics/returns_combined_panel.pdf and
#   a cumulative-delta variant).
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
  make_option("--out-pdf",    type = "character", default = "../graphics/returns_combined_panel.pdf"),
  make_option("--width",      type = "double", default = 10),
  make_option("--height",     type = "double", default = 6)
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
# LOAD RETURNS
df <- read.csv(opts$`in-csv`)
df$block_date <- as.Date(df$block_date)

# -----------------------------------------------------------------------------
# COMPUTE DELTAS
delta_df <- df %>%
  transmute(
    block_date,
    `Baseline (actual)` = (median_ret_baseline - median_market_return) * 100,
    `MaxRet`            = (median_ret_better_return - median_market_return) * 100,
    `MinVar`            = (median_ret_safer_risk - median_market_return) * 100,
    `MaxSR`             = (median_ret_max_sharpe - median_market_return) * 100,
    `Equal-weight`      = (median_ret_equal_weight - median_market_return) * 100,
    `MCap-weight`       = (median_ret_mcap_weight - median_market_return) * 100
  )

delta_long <- delta_df %>%
  pivot_longer(-block_date, names_to = "strategy", values_to = "delta_pct") %>%
  mutate(strategy = factor(strategy, levels = c(
    "Baseline (actual)", "MaxRet", "MinVar",
    "MaxSR", "Equal-weight", "MCap-weight"
  )))

# -----------------------------------------------------------------------------
# COMPUTE CUMULATIVE DELTA
cum_delta_df <- delta_df %>%
  arrange(block_date) %>%
  mutate(across(-block_date, cumsum))

cum_delta_long <- cum_delta_df %>%
  pivot_longer(-block_date, names_to = "strategy", values_to = "cum_delta_pct") %>%
  mutate(strategy = factor(strategy, levels = c(
    "Baseline (actual)", "MaxRet", "MinVar",
    "MaxSR", "Equal-weight", "MCap-weight"
  )))

# -----------------------------------------------------------------------------
# LOAD RETURNS FOR BOTTOM PANEL
ret_long <- df %>%
  select(block_date,
         median_ret_baseline, median_ret_better_return,
         median_ret_safer_risk, median_ret_max_sharpe,
         median_ret_equal_weight, median_ret_mcap_weight,
         median_market_return) %>%
  pivot_longer(-block_date, names_to = "strategy", values_to = "median_return") %>%
  mutate(
    median_return_pct = median_return * 100,
    strategy = recode(strategy,
      "median_ret_baseline"       = "Baseline (actual)",
      "median_ret_better_return"  = "MaxRet",
      "median_ret_safer_risk"     = "MinVar",
      "median_ret_max_sharpe"     = "MaxSR",
      "median_ret_equal_weight"   = "Equal-weight",
      "median_ret_mcap_weight"    = "MCap-weight",
      "median_market_return"      = "Market benchmark"
    ),
    strategy = factor(strategy, levels = c(
      "Market benchmark",
      "Baseline (actual)", "MaxRet", "MinVar",
      "MaxSR", "Equal-weight", "MCap-weight"
    ))
  )

# -----------------------------------------------------------------------------
# LOAD ETH PRICE (60-DAY MA)
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

# Scale ETH price to return axis
ret_range   <- range(ret_long$median_return_pct, na.rm = TRUE)
price_range <- range(price_monthly$eth_price, na.rm = TRUE)
scale_factor <- diff(ret_range) / diff(price_range)
offset       <- ret_range[1] - price_range[1] * scale_factor
price_monthly <- price_monthly %>%
  mutate(price_scaled = eth_price * scale_factor + offset)

# -----------------------------------------------------------------------------
# SHARED AESTHETICS
pal <- c(
  "Market benchmark"  = "#222222",
  "Baseline (actual)" = "#E31A1C",
  "MaxRet"            = "#1F78B4",
  "MinVar"            = "#33A02C",
  "MaxSR"             = "#FF7F00",
  "Equal-weight"      = "#6A3D9A",
  "MCap-weight"       = "#B15928"
)

pal_no_mkt <- pal[names(pal) != "Market benchmark"]

shp <- c(
  "Baseline (actual)" = 16,  # filled circle
  "MaxRet"            = 17,  # filled triangle
  "MinVar"            = 15,  # filled square
  "MaxSR"             = 18,  # filled diamond
  "Equal-weight"      = 1,   # open circle
  "MCap-weight"       = 2    # open triangle
)

lty <- c(
  "Market benchmark"  = "dashed",
  "Baseline (actual)" = "solid",
  "MaxRet"            = "solid",
  "MinVar"            = "solid",
  "MaxSR"             = "solid",
  "Equal-weight"      = "dotted",
  "MCap-weight"       = "dotted"
)

lty_no_mkt <- lty[names(lty) != "Market benchmark"]

shared_x <- scale_x_date(
  date_breaks = "6 months",
  date_labels = "%b %Y",
  expand = expansion(mult = 0.02)
)

shared_theme <- theme_minimal(base_size = FONT_BASE) +
  theme(
    panel.grid.major.y = element_line(colour = "grey85", linetype = "dashed"),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    legend.text        = element_text(size = FONT_LEGEND_TEXT),
    axis.text.x        = element_text(angle = 45, hjust = 1)
  )

# -----------------------------------------------------------------------------
# TOP PANEL: DELTA AS POINTS
p_top <- ggplot(delta_long, aes(x = block_date, y = delta_pct,
                                 colour = strategy, shape = strategy)) +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4, linetype = "dashed") +
  geom_point(size = 1.8, alpha = 0.75) +
  scale_colour_manual(values = pal_no_mkt) +
  scale_shape_manual(values = shp) +
  shared_x +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    x      = NULL,
    y      = "Strategy minus market (%)",
    colour = NULL, shape = NULL
  ) +
  shared_theme +
  theme(
    legend.position = "none",
    axis.text.x     = element_blank(),
    axis.ticks.x    = element_blank()
  )

# -----------------------------------------------------------------------------
# BOTTOM PANEL: RETURNS + ETH PRICE
p_bottom <- ggplot() +
  geom_area(data = price_monthly,
            aes(x = block_date, y = price_scaled),
            fill = "grey85", alpha = 0.5) +
  geom_line(data = price_monthly,
            aes(x = block_date, y = price_scaled),
            colour = "grey50", linewidth = 0.4) +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4) +
  geom_line(data = ret_long,
            aes(x = block_date, y = median_return_pct,
                colour = strategy, linetype = strategy),
            linewidth = 0.5) +
  geom_point(data = ret_long,
             aes(x = block_date, y = median_return_pct,
                 colour = strategy, shape = strategy),
             size = 1.2, alpha = 0.7) +
  scale_colour_manual(values = pal) +
  scale_linetype_manual(values = lty) +
  scale_shape_manual(values = c("Market benchmark" = 4, shp)) +
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
  shared_theme +
  theme(
    legend.position    = "bottom",
    axis.title.y.right = element_text(colour = "grey50"),
    axis.text.y.right  = element_text(colour = "grey50")
  ) +
  guides(
    colour   = guide_legend(nrow = 1),
    linetype = guide_legend(nrow = 1),
    shape    = guide_legend(nrow = 1)
  )

# -----------------------------------------------------------------------------
# COMBINE
p_combined <- p_top / p_bottom +
  plot_layout(heights = c(1, 1.4))

ggsave(opts$`out-pdf`, p_combined, width = opts$width, height = opts$height)
cat("[saved]", opts$`out-pdf`, "\n")

# -----------------------------------------------------------------------------
# BONUS: CUMULATIVE DELTA PLOT (SEPARATE FILE)
p_cum <- ggplot(cum_delta_long, aes(x = block_date, y = cum_delta_pct,
                                     colour = strategy, linetype = strategy)) +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4, linetype = "dashed") +
  geom_line(linewidth = 0.6) +
  scale_colour_manual(values = pal_no_mkt) +
  scale_linetype_manual(values = lty_no_mkt) +
  shared_x +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    x      = NULL,
    y      = "Cumulative excess return vs. market (%)",
    colour = NULL, linetype = NULL
  ) +
  shared_theme +
  theme(legend.position = "bottom") +
  guides(
    colour   = guide_legend(nrow = 1),
    linetype = guide_legend(nrow = 1)
  )

cum_pdf <- sub("\\.pdf$", "_cumulative.pdf", opts$`out-pdf`)
ggsave(cum_pdf, p_cum, width = opts$width, height = 5)
cat("[saved]", cum_pdf, "\n")
