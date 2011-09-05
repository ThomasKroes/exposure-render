
PROJECT(ExposureRender)

FIND_PACKAGE(CUDA)

IF(NOT VTK_BINARY_DIR)
FIND_PACKAGE(VTK)
IF(NOT VTK_DIR)
  MESSAGE(FATAL_ERROR "Please set VTK_DIR.")
ENDIF(NOT VTK_DIR)
INCLUDE(${VTK_USE_FILE})
ENDIF(NOT VTK_BINARY_DIR)

# use what QVTK built with
SET(QT_MOC_EXECUTABLE ${VTK_QT_MOC_EXECUTABLE} CACHE FILEPATH "")
SET(QT_UIC_EXECUTABLE ${VTK_QT_UIC_EXECUTABLE} CACHE FILEPATH "")
SET(QT_QMAKE_EXECUTABLE ${VTK_QT_QMAKE_EXECUTABLE} CACHE FILEPATH "")
FIND_PACKAGE(Qt4 REQUIRED)
set(QT_USE_QTXML TRUE)
INCLUDE(${QT_USE_FILE})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_EXPORTING")

# Use the include path and library for Qt that is used by VTK.
INCLUDE_DIRECTORIES(
	${QT_INCLUDE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CUDA_SDK_ROOT_DIR}/common/inc
	${CUDA_TOOLKIT_INCLUDE}
)

# Set your files and resources here  
SET(SimpleViewSrcs 
	Main.cpp 
	MainWindow.cpp
	
	# Lighting
	LightSettingsWidget.cpp 
	LightingDockWidget.cpp 
	LightsWidget.cpp
	LightingWidget.cpp
	Lighting.cpp
	BackgroundIlluminationWidget.cpp

	# Transfer function editor
	TransferFunction.cpp
	TransferFunctionCanvas.cpp
	TransferFunctionView.cpp
	TransferFunctionGradient.cpp
	TransferFunctionWidget.cpp
	NodePropertiesWidget.cpp

	# Camera
	Camera.cpp
	CameraDockWidget.cpp
	FilmWidget.cpp
	ApertureWidget.cpp
	ProjectionWidget.cpp
	FocusWidget.cpp
	
	Filter.cpp
	Flags.cpp
	Geometry.cpp
	Light.cpp
	MonteCarlo.cpp
	Random.cpp
	Spectrum.cpp
	Statistics.cpp
	VolumePoint.cpp
	StatisticsDockWidget.cpp
	StatisticsWidget.cpp
	CameraWidget.cpp
	
	
	LoadVolume.cpp
	InteractorStyleRealisticCamera.cpp
	RenderThread.cpp
	SettingsDockWidget.cpp
	LoadSettingsDialog.cpp
	VtkWidget.cpp
	Scene.cpp

	
	NodeItem.cpp
	AppearanceDockWidget.cpp
	PresetsWidget.cpp
	Preset.cpp
	Controls.cpp
	EdgeItem.cpp
	Material.cpp
	Sample.cpp
)

SET(SimpleViewUI SimpleView.ui)
SET(SimpleViewHeaders 
	MainWindow.h

	# Lighting
	LightSettingsWidget.h
	LightingDockWidget.h
	LightsWidget.h
	LightingWidget.h
	Lighting.h
	BackgroundIlluminationWidget.h

	# Transfer function editor
	TransferFunction.h
	TransferFunctionCanvas.h
	TransferFunctionView.h
	TransferFunctionGradient.h
	TransferFunctionWidget.h
	NodePropertiesWidget.h

	# Camera
	Camera.h
	CameraWidget.h
	CameraDockWidget.h
	FilmWidget.h
	ApertureWidget.h
	ProjectionWidget.h
	FocusWidget.h

	Defines.h
	Dll.h
	Enumerations.h
	Exception.h
	Filter.h
	Flags.h
	Geometry.h
	Light.h
	MonteCarlo.h
	Random.h
	Spectrum.h
	Statistics.h
	VolumePoint.h
	StatisticsDockWidget.h
	StatisticsWidget.h
	
	
	
	LoadVolume.h
	InteractorStyleRealisticCamera.h
	RenderThread.h
	SettingsDockWidget.h
	LoadSettingsDialog.h
	VtkWidget.h
	Scene.h

	
	NodeItem.h
	AppearanceDockWidget.h
	PresetsWidget.h
	Preset.h
	Controls.h
	EdgeItem.h
	Material.h
	Sample.h
)

SET(SimpleViewResources Icons/icons.qrc)

# The rest should just work (sure...)
QT4_WRAP_UI(UISrcs ${SimpleViewUI})
QT4_WRAP_CPP(MOCSrcs ${SimpleViewHeaders} )
QT4_ADD_RESOURCES(ResourceSrcs ${SimpleViewResources})

SOURCE_GROUP("Resources" FILES
  ${SimpleViewUI}
  ${SimpleViewResources}
  ${EXE_ICON}
)

SOURCE_GROUP("Generated" FILES
  ${UISrcs}
  ${MOCSrcs}
  ${ResourceSrcs}
  ${RCS_SOURCES}
)

# Group sources related to the transfer function editor together
SET(TransferFunctionHeaders
	
)

SET(TransferFunctionSrcs
	
)

# Make the group
SOURCE_GROUP("Transfer Function" FILES
	${TransferFunctionHeaders}
	${TransferFunctionSrcs}
)

# Group sources related to Cuda together
SET(CudaHeaders
	VolumeTracer.cuh
	VolumeTracer.cu
	VolumeTracerBackup.cuh
	VolumeTracerBackup.cu
)

SET(CudaSources
	VolumeTracer.cuh
	VolumeTracer.cu
	VolumeTracerBackup.cuh
	VolumeTracerBackup.cu
)

# Make the group
SOURCE_GROUP("Cuda Kernels" FILES
	${CudaHeaders}
	${CudaSources}
)

ADD_DEFINITIONS(-DQT_GUI_LIBS -DQT_CORE_LIB -DQT_XML_LIB -DQT3_SUPPORT)
SET_SOURCE_FILES_PROPERTIES(${SimpleViewSrcs} ${TransferFunctionHeaders} ${TransferFunctionSrcs} ${CudaHeaders} ${CudaSources} PROPERTIES OBJECT_DEPENDS "${UISrcs}")

# It's nice to have the ui in the windows project file...just double click on it
# and designer comes up on that ui file :)
IF (${CMAKE_BUILD_TOOL} MATCHES "msdev")
  SET (SimpleViewSrcs ${SimpleViewSrcs} ${SimpleViewUI} ${TransferFunctionHeaders} ${TransferFunctionSrcs} ${CudaSources})
  SET (TransferFunctionHeaders ${TransferFunctionHeaders})
  SET (TransferFunctionSrcs ${TransferFunctionSrcs})
ENDIF (${CMAKE_BUILD_TOOL} MATCHES "msdev")
IF (${CMAKE_BUILD_TOOL} MATCHES "devenv")
  SET (SimpleViewSrcs ${SimpleViewSrcs} ${SimpleViewUI} ${TransferFunctionHeaders} ${TransferFunctionSrcs} ${CudaSources})
  SET (TransferFunctionHeaders ${TransferFunctionHeaders})
  SET (TransferFunctionSrcs ${TransferFunctionSrcs})
ENDIF (${CMAKE_BUILD_TOOL} MATCHES "devenv")

CUDA_ADD_EXECUTABLE(ExposureRender MACOSX_BUNDLE ${SimpleViewSrcs} ${UISrcs} ${MOCSrcs} ${ResourceSrcs} ${TransferFunctionHeaders} ${TransferFunctionSrcs} ${CudaHeaders} ${CudaSources}
)

TARGET_LINK_LIBRARIES(ExposureRender
  QVTK
  ${QT_LIBRARIES}
  vtkRendering
  vtkGraphics
  vtkIO
  vtkCommon
  vtkViews
)

### Example .mhd and .raw files ###
INSTALL_FILES(/Examples FILES "${CMAKE_CURRENT_SOURCE_DIR}\\\\Bonsai.mhd")
INSTALL_FILES(/Examples FILES "${CMAKE_CURRENT_SOURCE_DIR}\\\\Bonsai.raw")

### Preset files ###
INSTALL_FILES(/ FILES ${CMAKE_CURRENT_BINARY_DIR}\\\\AppearancePresets.xml)
INSTALL_FILES(/ FILES ${CMAKE_CURRENT_BINARY_DIR}\\\\LightingPresets.xml)
INSTALL_FILES(/ FILES ${CMAKE_CURRENT_BINARY_DIR}\\\\CameraPresets.xml)

### ReadMe and Copyright ###
INSTALL_FILES(/ FILES ${CMAKE_CURRENT_SOURCE_DIR}\\\\ReadMe.txt)
INSTALL_FILES(/ FILES ${CMAKE_CURRENT_SOURCE_DIR}\\\\Copyright.txt)

INSTALL_TARGETS(/ ExposureRender)

INCLUDE(InstallRequiredSystemLibraries)

SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Exposure Render")
SET(CPACK_PACKAGE_VENDOR "TU Delft")
SET(CPACK_RESOURCE_FILE_WELCOME ${CMAKE_CURRENT_SOURCE_DIR}/Welcome.txt)
SET(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt)
SET(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/Copyright.txt)

SET(CPACK_PACKAGE_VERSION_MAJOR "1")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "0")

SET(CPACK_PACKAGE_INSTALL_DIRECTORY "Exposure Render")

# SET(CPACK_NSIS_EXTRA_INSTALL_COMMANDS ExecWait '\\\"$INSTDIR\\\\vcredist_x86.exe\\\" /q:a)

#SET(CPACK_PACKAGE_ICON "c:\\WindowsMovieMaker.bmp")
SET(CPACK_NSIS_DISPLAY_NAME "Exposure Render")
SET(CPACK_NSIS_CONTACT "t.kroes@tudelft.nl")
SET(CPACK_NSIS_HELP_LINK "http://graphics.tudelft.nl/")
SET(CPACK_NSIS_URL_INFO_ABOUT "http://code.google.com/p/exposure-render/")
SET(CPACK_NSIS_MUI_FINISHPAGE_RUN "ExposureRender")

set(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")
SET(CPACK_PACKAGE_EXECUTABLES "ExposureRender;Exposure Render")


 
INCLUDE(CPack)

































