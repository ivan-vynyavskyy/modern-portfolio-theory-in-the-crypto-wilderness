#!/usr/bin/env Rscript
# =============================================================================
# plot_returns_with_eth_price.R
#
# Description:
#   Median returns per block with wETH price overlay on the right
#   axis, reinforcing that returns move in lockstep with the ETH
#   price cycle.
#
# Input:
#   CSV with per-block return statistics + ETH price CSV.
#
# Output:
#   Single PDF (default: ../graphics/returns_with_eth_price.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(optparse)
  library(scales)
})

option_list <- list(
  make_option("--in-csv",     type = "character", default = "../data/camp_analysis_results_full/returns_per_block.csv"),
  make_option("--price-csv",  type = "character", default = "../data/weth_price.csv",
              help = "CSV with columns: date, daily_close_usd"),
  make_option("--out-pdf",    type = "character", default = "../graphics/returns_with_eth_price.pdf"),
  make_option("--width",      type = "double", default = 10),
  make_option("--height",     type = "double", default = 5.5)
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

long <- df %>%
  select(block_date,
         median_ret_baseline,
         median_ret_better_return,
         median_ret_safer_risk,
         median_ret_max_sharpe,
         median_ret_equal_weight,
         median_ret_mcap_weight) %>%
  pivot_longer(
    cols      = -block_date,
    names_to  = "strategy",
    values_to = "median_return"
  ) %>%
  mutate(
    median_return_pct = median_return * 100,
    strategy = recode(strategy,
      "median_ret_baseline"       = "Baseline (actual)",
      "median_ret_better_return"  = "MaxRet",
      "median_ret_safer_risk"     = "MinVar",
      "median_ret_max_sharpe"     = "MaxSR",
      "median_ret_equal_weight"   = "Equal-weight",
      "median_ret_mcap_weight"    = "MCap-weight"
    ),
    strategy = factor(strategy, levels = c(
      "Baseline (actual)", "MaxRet", "MinVar",
      "MaxSR", "Equal-weight", "MCap-weight"
    ))
  )

# -----------------------------------------------------------------------------
# LOAD ETH PRICE
price <- read.csv(opts$`price-csv`)
price$date <- as.Date(price$date)

# Get monthly price: take the price closest to each block_date
block_dates <- unique(df$block_date)
price_monthly <- data.frame(block_date = block_dates) %>%
  rowwise() %>%
  mutate(
    eth_price = {
      # Find closest date in price data (within ±15 days)
      diffs <- abs(as.numeric(price$date - block_date))
      idx <- which.min(diffs)
      if (diffs[idx] <= 15) price$daily_close_usd[idx] else NA_real_
    }
  ) %>%
  ungroup()

# -----------------------------------------------------------------------------
# COMPUTE AXIS SCALING
# Map ETH price to the return y-axis range
ret_range  <- range(long$median_return_pct, na.rm = TRUE)
price_range <- range(price_monthly$eth_price, na.rm = TRUE)

# Linear transform: price -> return scale
scale_factor <- diff(ret_range) / diff(price_range)
offset       <- ret_range[1] - price_range[1] * scale_factor

price_monthly <- price_monthly %>%
  mutate(price_scaled = eth_price * scale_factor + offset)

# -----------------------------------------------------------------------------
# COLOURS
pal <- c(
  "Baseline (actual)" = "#E31A1C",
  "MaxRet"            = "#1F78B4",
  "MinVar"            = "#33A02C",
  "MaxSR"             = "#FF7F00",
  "Equal-weight"      = "#6A3D9A",
  "MCap-weight"       = "#B15928"
)

lty <- c(
  "Baseline (actual)" = "solid",
  "MaxRet"            = "solid",
  "MinVar"            = "solid",
  "MaxSR"             = "solid",
  "Equal-weight"      = "dotted",
  "MCap-weight"       = "dotted"
)

shp <- c(
  "Baseline (actual)" = 16,
  "MaxRet"            = 17,
  "MinVar"            = 15,
  "MaxSR"             = 18,
  "Equal-weight"      = 1,
  "MCap-weight"       = 2
)

# -----------------------------------------------------------------------------
# PLOT
p <- ggplot() +
  # ETH price as shaded area (background)
  geom_area(data = price_monthly,
            aes(x = block_date, y = price_scaled),
            fill = "grey85", alpha = 0.5) +
  geom_line(data = price_monthly,
            aes(x = block_date, y = price_scaled),
            colour = "grey50", linewidth = 0.4) +
  # Zero line
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4) +
  # Strategy returns
  geom_line(data = long,
            aes(x = block_date, y = median_return_pct,
                colour = strategy, linetype = strategy),
            linewidth = 0.55) +
  geom_point(data = long,
             aes(x = block_date, y = median_return_pct,
                 colour = strategy, shape = strategy),
             size = 1.4, alpha = 0.8) +
  # Scales
  scale_colour_manual(values = pal) +
  scale_linetype_manual(values = lty) +
  scale_shape_manual(values = shp) +
  scale_x_date(
    date_breaks = "6 months",
    date_labels = "%b %Y",
    expand      = expansion(mult = 0.02)
  ) +
  scale_y_continuous(
    name   = "Median 20-day return (%)",
    labels = function(x) paste0(x, "%"),
    sec.axis = sec_axis(
      ~ (. - offset) / scale_factor,
      name   = "wETH price (USD)",
      labels = dollar_format(prefix = "$", big.mark = ",")
    )
  ) +
  labs(
    x      = NULL,
    colour = NULL, linetype = NULL, shape = NULL
  ) +
  theme_minimal(base_size = FONT_BASE) +
  theme(
    panel.grid.major.y = element_line(colour = "grey90", linetype = "dashed"),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    legend.position    = "bottom",
    legend.text        = element_text(size = FONT_LEGEND_TEXT),
    axis.text.x        = element_text(angle = 45, hjust = 1),
    axis.title.y.right = element_text(colour = "grey50"),
    axis.text.y.right  = element_text(colour = "grey50")
  ) +
  guides(
    colour   = guide_legend(nrow = 1),
    linetype = guide_legend(nrow = 1),
    shape    = guide_legend(nrow = 1)
  )

ggsave(opts$`out-pdf`, p, width = opts$width, height = opts$height)
cat("[saved]", opts$`out-pdf`, "\n")
