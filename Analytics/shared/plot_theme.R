# =============================================================================
# plot_theme.R
#
# Shared typography (font sizes in points) for every plot script under
# Analytics/. Source it from a plot script via:
#
#   script_dir <- (function() {
#     args <- commandArgs(trailingOnly = FALSE)
#     file_arg <- args[grep("^--file=", args)]
#     if (length(file_arg) > 0) {
#       dirname(normalizePath(sub("^--file=", "", file_arg)))
#     } else {
#       getwd()
#     }
#   })()
#   source(file.path(script_dir, "..", "..", "..", "shared", "plot_theme.R"))
#
# The `../../../` assumes the caller lives at Analytics/<area>/<topic>/plot/.
# Adjust the relative depth if the caller sits elsewhere.
#
# After sourcing, call apply_plot_theme_fonts(width) once (typically right
# after parsing CLI options) to populate the FONT_* globals based on the
# target plot width. Each script's theme() block then references those
# FONT_* globals as before. Because the globals hold absolute point sizes,
# typography stays visually consistent across PDFs that are later embedded
# at the same display width.
#
#   opt$width <- apply_plot_theme_fonts(opt$width)
#
# Widths above 12" are clamped to 12" (both the fonts and the returned
# effective width). Widths exactly at 10" or 12" use the presets below;
# intermediate / sub-10" widths are linearly interpolated between the two.
# =============================================================================

# ---- Preset for width = 10" (Standard) --------------------------------------
FONT_BASE_S         <- 14
FONT_AXIS_TITLE_S   <- 12
FONT_AXIS_TEXT_S    <- 11
FONT_LEGEND_TITLE_S <- 13
FONT_LEGEND_TEXT_S  <- 11
FONT_STRIP_S        <- 12

# ---- Preset for width = 12" (Large) -----------------------------------------
FONT_BASE_L         <- 17
FONT_AXIS_TITLE_L   <- 14
FONT_AXIS_TEXT_L    <- 13
FONT_LEGEND_TITLE_L <- 15
FONT_LEGEND_TEXT_L  <- 13
FONT_STRIP_L        <- 14

# ---- Resolved font sizes (filled in by apply_plot_theme_fonts) --------------
FONT_BASE         <- NA_real_
FONT_AXIS_TITLE   <- NA_real_
FONT_AXIS_TEXT    <- NA_real_
FONT_LEGEND_TITLE <- NA_real_
FONT_LEGEND_TEXT  <- NA_real_
FONT_STRIP        <- NA_real_

# -----------------------------------------------------------------------------
# apply_plot_theme_fonts(width)
#
# Resolve the FONT_* globals for the given target plot width (in inches) and
# return the effective width to use when saving the PDF. Widths > 12 are
# clamped to 12; widths between 10 and 12 interpolate linearly between the
# Standard and Large presets; widths < 10 extrapolate below Standard.
# -----------------------------------------------------------------------------
apply_plot_theme_fonts <- function(width) {
  width_capped <- min(width, 12)
  t <- (width_capped - 10) / (12 - 10)

  lerp <- function(s, l) s + t * (l - s)

  FONT_BASE         <<- lerp(FONT_BASE_S,         FONT_BASE_L)
  FONT_AXIS_TITLE   <<- lerp(FONT_AXIS_TITLE_S,   FONT_AXIS_TITLE_L)
  FONT_AXIS_TEXT    <<- lerp(FONT_AXIS_TEXT_S,    FONT_AXIS_TEXT_L)
  FONT_LEGEND_TITLE <<- lerp(FONT_LEGEND_TITLE_S, FONT_LEGEND_TITLE_L)
  FONT_LEGEND_TEXT  <<- lerp(FONT_LEGEND_TEXT_S,  FONT_LEGEND_TEXT_L)
  FONT_STRIP        <<- lerp(FONT_STRIP_S,        FONT_STRIP_L)

  width_capped
}
