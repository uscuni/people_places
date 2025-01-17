# Version checking for GeoTIFF

set (PACKAGE_VERSION "1.7.3")
set (PACKAGE_VERSION_MAJOR "1")
set (PACKAGE_VERSION_MINOR "7")
set (PACKAGE_VERSION_PATCH "3")

if (NOT PACKAGE_FIND_NAME STREQUAL "GeoTIFF")
  # Check package name (in particular, because of the way cmake finds
  # package config files, the capitalization could easily be "wrong").
  # This is necessary to ensure that the automatically generated
  # variables, e.g., <package>_FOUND, are consistently spelled.  Make
  # this a WARNING, because this is a user error that needs to be fixed.
  message (WARNING
    "Mismatched package names: use find_package(GeoTIFF ...) instead"
    " of find_package(${PACKAGE_FIND_NAME} ...)")
  set (PACKAGE_VERSION_UNSUITABLE TRUE)
elseif (NOT (APPLE OR CMAKE_SIZEOF_VOID_P EQUAL 8))
  # Reject if there's a 32-bit/64-bit mismatch (may not be necessary
  # with Apple since a multi-architecture library might be built for
  # that platform).
  message (STATUS
    "${CMAKE_CURRENT_LIST_FILE} unsuitable because package built with "
    "sizeof(*void) =  8")
  set (PACKAGE_VERSION_UNSUITABLE TRUE)
elseif (PACKAGE_FIND_VERSION)
  if (PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
    set (PACKAGE_VERSION_EXACT TRUE)
  elseif (PACKAGE_FIND_VERSION VERSION_LESS PACKAGE_VERSION
    AND PACKAGE_FIND_VERSION_MAJOR EQUAL PACKAGE_VERSION_MAJOR)
    set (PACKAGE_VERSION_COMPATIBLE TRUE)
  endif ()
endif ()
