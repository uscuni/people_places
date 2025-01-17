#!/bin/sh
# Restore previous GeoTIFF env vars if they were set

unset GEOTIFF_CSV
if [ -n "${_CONDA_SET_GEOTIFF_CSV:-}" ]; then
    export GEOTIFF_CSV="$_CONDA_SET_GEOTIFF_CSV"
    unset _CONDA_SET_GEOTIFF_CSV
fi

