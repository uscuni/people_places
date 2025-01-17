# Configure GeoTIFF
#
# Set
#  GeoTIFF_FOUND = 1
#  GeoTIFF_INCLUDE_DIRS = /usr/local/include
#  GeoTIFF_LIBRARY_DIRS = /usr/local/lib
#  GeoTIFF_BINARY_DIRS = /usr/local/bin
#  GeoTIFF_VERSION = 1.4.1 (for example)

# For compatibility with FindGeoTIFF.cmake, also set
# GEOTIFF_FOUND
# GEOTIFF_INCLUDE_DIR
# GEOTIFF_LIBRARY
# GEOTIFF_LIBRARIES

# Tell the user project where to find our headers and libraries
get_filename_component (_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
get_filename_component (_ROOT "${_DIR}/../../.." ABSOLUTE)
set (GeoTIFF_INCLUDE_DIRS "${_ROOT}/include")
set (GeoTIFF_LIBRARY_DIRS "${_ROOT}/lib")
set (GeoTIFF_BINARY_DIRS "${_ROOT}/bin")
unset (_ROOT)
unset (_DIR)

include(CMakeFindDependencyMacro)
find_dependency(TIFF)

set (GeoTIFF_LIBRARIES geotiff_library)
if("ON")
  set (GeoTIFF_SHARED_LIBRARIES ${GeoTIFF_LIBRARIES})
else()
  set (GeoTIFF_STATIC_LIBRARIES ${GeoTIFF_LIBRARIES})
  find_dependency(PROJ CONFIG)

endif()

if(NOT GeoTIFF_FIND_QUIETLY)
  message (STATUS "Reading ${CMAKE_CURRENT_LIST_FILE}")
  # GeoTIFF_VERSION is set by version file
  message (STATUS
    "GeoTIFF configuration, version ${GeoTIFF_VERSION}")
  message (STATUS "  include directory: \${GeoTIFF_INCLUDE_DIRS}")
  if("ON")
    message (STATUS "  \${GeoTIFF_LIBRARIES} set to shared library")
  else()
    message (STATUS "  \${GeoTIFF_LIBRARIES} set to static libraries")
  endif()
endif()

# Read in the exported definition of the library
include ("${CMAKE_CURRENT_LIST_DIR}/geotiff-depends.cmake")

# For compatibility with FindGeoTIFF.cmake
set (GEOTIFF_FOUND 1)
set (GEOTIFF_LIBRARIES ${GeoTIFF_LIBRARIES})
set (GEOTIFF_INCLUDE_DIR ${GeoTIFF_INCLUDE_DIRS})
set (GEOTIFF_LIBRARY ${GeoTIFF_LIBRARIES})
