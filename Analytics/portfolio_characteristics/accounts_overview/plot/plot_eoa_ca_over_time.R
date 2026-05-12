#!/usr/bin/env Rscript
# =============================================================================
# plot_eoa_ca_over_time.R
#
# Description:
#   Dual-axis line chart of EOA and CA account counts over time.
#   Left axis shows EOA (orange), right axis shows CA (blue), both
#   with end-of-line labels. Also prints annual CAGR growth stats.
#
# Input:
#   CSV with columns: block, date, eoa_accounts, ca_accounts.
#
# Output:
#   Single PDF (default: ../graphics/eoa_ca_over_time_dual_axis.pdf).
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(lubridate)
  library(ggplot2)
  library(scales)
  library(ggrepel)
})

# ------------------------------------------------------------------------------------------------------------------------------
# CLI OPTIONS
# ------------------------------------------------------------------------------------------------------------------------------
option_list <- list(
  make_option(c("--input"), type="character",
              help="Input CSV: block,date,eoa_accounts,ca_accounts", metavar="FILE"),
  make_option(c("--out-pdf"), type="character",
              default="../graphics/eoa_ca_over_time_dual_axis.pdf",
              help="Output PDF path [default: %default]", metavar="FILE"),
  make_option(c("--width"), type="double", default=10, help="Plot width in inches"),
  make_option(c("--height"), type="double", default=5.5, help="Plot height in inches"),
  make_option(c("--cummax"), action="store_true", default=FALSE,
              help="Force series to be monotonic non-decreasing")
)

opt <- parse_args(OptionParser(option_list=option_list))
if (is.null(opt$input)) stop("Missing --input.")

# ------------------------------------------------------------------------------------------------------------------------------
# DATA MAPPING — sourced from shared files
# ------------------------------------------------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------------------------------------------------
# PALETTE
# ------------------------------------------------------------------------------------------------------------------------------
COLOR_CA  <- "#1f77b4"
COLOR_EOA <- "#ff7f0e"

COLORS <- c(
  "Contract accounts (CA)" = COLOR_CA,
  "Externally owned accounts (EOA)" = COLOR_EOA
)

# ------------------------------------------------------------------------------------------------------------------------------
# LOAD & PROCESS
# ------------------------------------------------------------------------------------------------------------------------------
df <- read_csv(opt$input, show_col_types = FALSE) %>%
  mutate(
    date = ymd(date),
    eoa_accounts = as.numeric(eoa_accounts),
    ca_accounts  = as.numeric(ca_accounts)
  ) %>%
  filter(!is.na(date)) %>%
  arrange(date)

if (isTRUE(opt$cummax)) {
  df <- df %>%
    mutate(
      eoa_accounts = cummax(eoa_accounts),
      ca_accounts  = cummax(ca_accounts)
    )
}

# ------------------------------------------------------------------------------------------------------------------------------
# GROWTH STATS
# ------------------------------------------------------------------------------------------------------------------------------
cat("--------------------------------------------------\n")
cat(" Growth Statistics \n")
cat("--------------------------------------------------\n")

start_row <- head(df, 1)
end_row   <- tail(df, 1)

# Calculate duration in years (approx 365.25 days per year)
years_diff <- as.numeric(difftime(end_row$date, start_row$date, units = "days")) / 365.25

if (years_diff > 0) {
  calc_cagr <- function(start_val, end_val, years) {
    if (start_val <= 0) return(NA)
    (end_val / start_val)^(1 / years) - 1
  }

  eoa_cagr <- calc_cagr(start_row$eoa_accounts, end_row$eoa_accounts, years_diff)
  eoa_abs  <- (end_row$eoa_accounts - start_row$eoa_accounts) / years_diff

  ca_cagr  <- calc_cagr(start_row$ca_accounts, end_row$ca_accounts, years_diff)
  ca_abs   <- (end_row$ca_accounts - start_row$ca_accounts) / years_diff

  cat(sprintf("Time Span:       %.2f years (%s to %s)\n\n",
              years_diff, start_row$date, end_row$date))

  if (!is.na(eoa_cagr)) {
    cat(sprintf("EOA Growth Rate: %.2f%% per year (CAGR)\n", eoa_cagr * 100))
  } else {
    cat("EOA Growth Rate: N/A (Start value <= 0)\n")
  }
  cat(sprintf("EOA Avg Increase: +%s accounts/year\n\n", comma(round(eoa_abs))))

  if (!is.na(ca_cagr)) {
    cat(sprintf("CA Growth Rate:  %.2f%% per year (CAGR)\n", ca_cagr * 100))
  } else {
    cat("CA Growth Rate:  N/A (Start value <= 0)\n")
  }
  cat(sprintf("CA Avg Increase:  +%s accounts/year\n", comma(round(ca_abs))))

} else {
  cat("[WARN] Dataset spans less than 1 day. Cannot calculate annual growth.\n")
}
cat("--------------------------------------------------\n")


# -----------------------------------------------------------------------------
# Dual-axis scaling
# -----------------------------------------------------------------------------
# We scale CA so both series are comparable on the same plot.
max_eoa <- max(df$eoa_accounts, na.rm = TRUE)
max_ca  <- max(df$ca_accounts,  na.rm = TRUE)

scale_factor <- ifelse(max_ca == 0, 1, max_eoa / max_ca)

df <- df %>%
  mutate(ca_scaled = ca_accounts * scale_factor)

# -----------------------------------------------------------------------------
# ANNOTATION POINTS (last date)
# -----------------------------------------------------------------------------
last_row <- df %>% slice_max(order_by = date, n = 1, with_ties = FALSE)

label_points <- tibble::tibble(
  date = last_row$date,
  series = c("Externally owned accounts (EOA)", "Contract accounts (CA)"),
  y = c(last_row$eoa_accounts, last_row$ca_scaled),
  label = c(
    comma(round(last_row$eoa_accounts)),
    comma(round(last_row$ca_accounts))
  )
)

max_date_extended <- max(df$date) + months(6)

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
p <- ggplot(df, aes(x = date)) +

  geom_line(
    aes(y = eoa_accounts, color = "Externally owned accounts (EOA)"),
    linewidth = 1.2
  ) +

  geom_line(
    aes(y = ca_scaled, color = "Contract accounts (CA)"),
    linewidth = 1.2
  ) +

  geom_point(
    data = label_points,
    aes(x = date, y = y, color = series),
    size = 3
  ) +

  geom_label_repel(
    data = label_points %>% filter(series == "Externally owned accounts (EOA)"),
    aes(x = date, y = y, label = label, color = series),
    nudge_x = -40,
    nudge_y = -0.02 * max(df$eoa_accounts, na.rm = TRUE),
    direction = "y",
    hjust = 1,
    fontface = "bold",
    size = 4,
    show.legend = FALSE,
    xlim = c(-Inf, Inf),
    ylim = c(-Inf, Inf)
  ) +

  geom_label_repel(
    data = label_points %>% filter(series == "Contract accounts (CA)"),
    aes(x = date, y = y, label = label, color = series),
    nudge_x = 40,
    nudge_y = 0.02 * max(df$eoa_accounts, na.rm = TRUE),
    direction = "y",
    hjust = 0,
    fontface = "bold",
    size = 4,
    show.legend = FALSE,
    xlim = c(-Inf, Inf),
    ylim = c(-Inf, Inf)
  ) +

  scale_x_date(
    limits = c(min(df$date), max_date_extended),
    date_breaks = "6 months",
    date_labels = "%b '%y",
    expand = expansion(mult = c(0.01, 0))
  ) +

  scale_y_continuous(
    labels = comma,
    name = "Externally owned accounts (EOA)",
    sec.axis = sec_axis(
      trans = ~ . / scale_factor,
      name = "Contract accounts (CA)",
      labels = comma
    )
  ) +

  scale_color_manual(values = COLORS) +

  labs(
    x = NULL,
    color = NULL
  ) +

  coord_cartesian(clip = "off") +

  theme_minimal(base_size = FONT_BASE) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = FONT_LEGEND_TEXT, face = "bold"),
    legend.margin = margin(t = 20),

    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "grey90", linetype = "dashed"),

    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(20, 35, 20, 20),

    axis.text = element_text(size = FONT_AXIS_TEXT, color = "grey30"),

    axis.title.y.left = element_text(
      size = FONT_AXIS_TITLE, color = COLOR_EOA,
      face = "bold", angle = 90, margin = margin(r = 10)
    ),

    axis.title.y.right = element_text(
      size = FONT_AXIS_TITLE, color = COLOR_CA,
      face = "bold", angle = 90, margin = margin(l = 10)
    )
  )

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
out_file <- opt$`out-pdf`
if (!grepl("\\.pdf$", out_file, ignore.case = TRUE)) {
  out_file <- paste0(out_file, ".pdf")
}

dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)

ggsave(
  filename = out_file,
  plot = p,
  width = opt$width,
  height = opt$height,
  bg = "white"
)

cat(sprintf("[ok] Saved: %s\n", out_file))