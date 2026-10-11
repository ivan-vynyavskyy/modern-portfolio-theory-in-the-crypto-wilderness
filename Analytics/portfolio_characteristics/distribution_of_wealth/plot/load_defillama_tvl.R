# =============================================================================
# load_defillama_tvl.R
#
# Shared helper: fetch (or read from cache) the DeFiLlama Ethereum TVL series
# and join it to the monthly snapshot dates used by the plot scripts in this
# directory. Source it via:
#
#   script_dir <- (function() {
#     args <- commandArgs(trailingOnly = FALSE)
#     file_arg <- args[grep("^--file=", args)]
#     if (length(file_arg) > 0) dirname(normalizePath(sub("^--file=", "", file_arg)))
#     else getwd()
#   })()
#   source(file.path(script_dir, "load_defillama_tvl.R"))
#
# After sourcing, the following are available:
#
#   DEFILLAMA_ETH_TVL_URL
#   load_defillama_tvl(cache_csv, refresh)   -> tibble(date, tvl_usd)
#   join_defillama_tvl(dates, tvl)           -> tibble(date, tvl_usd)
#
# Data source (free, no key):
#   https://api.llama.fi/v2/historicalChainTvl/Ethereum
#   JSON array of {date: <unix seconds, UTC 00:00>, tvl: <usd>}, one point per
#   day. Snapshot dates are first-of-month 00:00 UTC and match the daily
#   points exactly.
#
# The cache CSV (columns: date, tvl_usd) is shipped in ../data so that the
# plots can be regenerated offline (cluster compute nodes have no internet).
# Regenerate it with refresh = TRUE on a machine with internet if the
# snapshot set changes.
# =============================================================================

suppressPackageStartupMessages({
  library(tibble)
  library(dplyr)
  library(readr)
})

DEFILLAMA_ETH_TVL_URL <- "https://api.llama.fi/v2/historicalChainTvl/Ethereum"

# -----------------------------------------------------------------------------
# load_defillama_tvl(cache_csv, refresh)
#
# Return a tibble with columns `date` (Date) and `tvl_usd` (double), one row
# per day (or per snapshot date if the shipped cache is used). If `cache_csv`
# exists and `refresh` is FALSE the cache is read and returned as-is;
# otherwise the series is fetched from DeFiLlama, written to `cache_csv`, and
# returned.
# -----------------------------------------------------------------------------
load_defillama_tvl <- function(cache_csv = "../data/defillama_eth_tvl.csv",
                               refresh   = FALSE) {
  if (!refresh && file.exists(cache_csv)) {
    tvl <- readr::read_csv(
      cache_csv,
      col_types = readr::cols(date = readr::col_date(), tvl_usd = readr::col_double())
    )
    return(tvl %>% arrange(date))
  }

  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required to fetch DeFiLlama TVL (no cache found at ", cache_csv, ").")
  }
  cat(sprintf("Fetching DeFiLlama Ethereum TVL from %s ...\n", DEFILLAMA_ETH_TVL_URL))
  raw <- jsonlite::fromJSON(DEFILLAMA_ETH_TVL_URL)
  tvl <- tibble(
    date    = as.Date(as.POSIXct(raw$date, origin = "1970-01-01", tz = "UTC")),
    tvl_usd = as.numeric(raw$tvl)
  ) %>%
    arrange(date)

  dir.create(dirname(cache_csv), showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(tvl, cache_csv)
  cat(sprintf("Cached %d daily points to %s\n", nrow(tvl), cache_csv))
  tvl
}

# -----------------------------------------------------------------------------
# join_defillama_tvl(dates, tvl)
#
# Map each snapshot date to its DeFiLlama TVL. Exact-date match first; as a
# guard, any snapshot date without an exact match falls back to the nearest
# earlier TVL point (with a warning). Dates before the first TVL point stay NA.
# Returns a tibble(date, tvl_usd) with one row per input date, in input order.
# -----------------------------------------------------------------------------
join_defillama_tvl <- function(dates, tvl) {
  dates <- as.Date(dates)
  tvl   <- tvl %>% arrange(date)

  exact <- match(dates, tvl$date)
  out   <- tvl$tvl_usd[exact]

  missing <- which(is.na(exact))
  if (length(missing) > 0) {
    idx  <- findInterval(as.numeric(dates[missing]), as.numeric(tvl$date))
    fill <- ifelse(idx >= 1, tvl$tvl_usd[pmax(idx, 1)], NA_real_)
    out[missing] <- fill
    warning(sprintf(
      "join_defillama_tvl: %d snapshot date(s) had no exact DeFiLlama match, used nearest earlier point: %s",
      length(missing), paste(format(dates[missing]), collapse = ", ")
    ))
  }

  tibble(date = dates, tvl_usd = out)
}
