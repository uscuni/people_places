#!/bin/sh

# Store existing GeoTIFF env vars and set to this conda env
# so other GeoTIFF installs don't pollute the environment

if [ -n "${GEOTIFF_CSV:-}" ]; then
    export _CONDA_SET_GEOTIFF_CSV="$GEOTIFF_CSV"
fi

# On Linux GEOTIFF_CSV is in $CONDA_PREFIX/share/epsg_csv, but
# Windows keeps it in $CONDA_PREFIX/Library/share/epsg_csv
if [ -d "${CONDA_PREFIX}/share/epsg_csv" ]; then
    export GEOTIFF_CSV="${CONDA_PREFIX}/share/epsg_csv"
elif [ -d "${CONDA_PREFIX}/Library/share/epsg_csv" ]; then
    export GEOTIFF_CSV="${CONDA_PREFIX}/Library/share/epsg_csv"
fi


