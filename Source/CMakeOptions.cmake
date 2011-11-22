#
# Configure output paths for libraries and executables.
#


#
# Try to find VTK and include its settings (otherwise complain)
#
IF(NOT VTK_BINARY_DIR)
  FIND_PACKAGE(VTK REQUIRED)
  INCLUDE(${VTK_USE_FILE})
ENDIF(NOT VTK_BINARY_DIR)

#
# Build shared libs ?
#
# Defaults to the same VTK setting.
#

# Standard CMake option for building libraries shared or static by default.
OPTION(BUILD_SHARED_LIBS
       "Build with shared libraries."
       ${VTK_BUILD_SHARED_LIBS})
# Copy the CMake option to a setting with VTKMY_ prefix for use in
# our project.  This name is used in vtkmyConfigure.h.in.
SET(VTKMY_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

IF (VTK_WRAP_PYTHON)

  OPTION(VTKMY_WRAP_PYTHON
         "Wrap classes into the Python interpreted language."
         ON)

  IF (VTKMY_WRAP_PYTHON)
    SET(VTK_WRAP_PYTHON_FIND_LIBS ON)
    INCLUDE(${VTK_CMAKE_DIR}/vtkWrapPython.cmake)
    IF (WIN32)
      IF (NOT BUILD_SHARED_LIBS)
        MESSAGE(FATAL_ERROR "Python support requires BUILD_SHARED_LIBS to be ON.")
        SET (VTKMY_CAN_BUILD 0)
      ENDIF (NOT BUILD_SHARED_LIBS)
    ENDIF (WIN32)
  ENDIF (VTKMY_WRAP_PYTHON)

ELSE (VTK_WRAP_PYTHON)

  IF (VTKMY_WRAP_PYTHON)
    MESSAGE("Warning. VTKMY_WRAP_PYTHON is ON but the VTK version you have "
            "chosen has not support for Python (VTK_WRAP_PYTHON is OFF).  "
            "Please set VTKMY_WRAP_PYTHON to OFF.")
    SET (VTKMY_WRAP_PYTHON OFF)
  ENDIF (VTKMY_WRAP_PYTHON)

ENDIF (VTK_WRAP_PYTHON)

# Setup our local hints file in case wrappers need them.
SET(VTK_WRAP_HINTS ${CMAKE_CURRENT_SOURCE_DIR}/Wrapping/hints)







