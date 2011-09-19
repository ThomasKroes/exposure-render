cmake_minimum_required(VERSION 2.8)

PROJECT(ExposureRender)

### Set Build Type ###
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")

FIND_PACKAGE(CUDA)

IF(NOT VTK_BINARY_DIR)
FIND_PACKAGE(VTK)
IF(NOT VTK_DIR)
  MESSAGE(FATAL_ERROR "Please set VTK_DIR.")
ENDIF(NOT VTK_DIR)
INCLUDE(${VTK_USE_FILE})
ENDIF(NOT VTK_BINARY_DIR)

MESSAGE(STATUS "vv" ${CMAKE_INSTALL_PREFIX})

# use what QVTK built with
SET(QT_MOC_EXECUTABLE ${VTK_QT_MOC_EXECUTABLE} CACHE FILEPATH "")
SET(QT_UIC_EXECUTABLE ${VTK_QT_UIC_EXECUTABLE} CACHE FILEPATH "")
SET(QT_QMAKE_EXECUTABLE ${VTK_QT_QMAKE_EXECUTABLE} CACHE FILEPATH "")
FIND_PACKAGE(Qt4 REQUIRED)

# Enable the use of XML in Qt
SET(QT_USE_QTXML TRUE)

INCLUDE(${QT_USE_FILE})

# Enable the export of symbols
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
SET(ExposureRenderSources 
	
	# Main
	MainWindow.cpp
	Main.cpp 
	
	# Lighting
	LightSettingsWidget.cpp 
	LightingDockWidget.cpp 
	LightsWidget.cpp
	LightingWidget.cpp
	Lighting.cpp
	Background.cpp
	BackgroundIlluminationWidget.cpp

	# Appearance
	TransferFunction.cpp
	TransferFunctionCanvas.cpp
	TransferFunctionView.cpp
	TransferFunctionGradient.cpp
	TransferFunctionWidget.cpp
	NodePropertiesWidget.cpp
	AppearanceDockWidget.cpp
	NodeItem.cpp
	EdgeItem.cpp
	Histogram.cpp
	Node.cpp

	# Camera
	Camera.cpp
	CameraDockWidget.cpp
	FilmWidget.cpp
	ApertureWidget.cpp
	ProjectionWidget.cpp
	FocusWidget.cpp
	CameraWidget.cpp
	Film.cpp
	Aperture.cpp
	Projection.cpp
	Focus.cpp

	# Presets
	PresetsWidget.cpp
	Preset.cpp

	# Statistics
	StatisticsDockWidget.cpp
	StatisticsWidget.cpp
	
	# Core
	Stable.cpp
	Filter.cpp
	Flags.cpp
	Geometry.cpp
	Light.cpp
	MonteCarlo.cpp
	Random.cpp
	Spectrum.cpp
	Statistics.cpp
	VolumePoint.cpp
	InteractorStyleRealisticCamera.cpp
	RenderThread.cpp
	SettingsDockWidget.cpp
	LoadSettingsDialog.cpp
	VtkWidget.cpp
	Scene.cpp
	Controls.cpp
	Material.cpp
	Sample.cpp
	LightTransport.cpp
	StartupDialog.cpp
	Utilities.cpp

	# Logging
	Logger.cpp
	LogWidget.cpp
	LogDockWidget.cpp
)

SET(ExposureRenderHeaders 
	
	# Main
	MainWindow.h

	# Lighting
	LightSettingsWidget.h
	LightingDockWidget.h
	LightsWidget.h
	LightingWidget.h
	Lighting.h
	Background.h
	BackgroundIlluminationWidget.h

	# Appearance
	TransferFunction.h
	TransferFunctionCanvas.h
	TransferFunctionView.h
	TransferFunctionGradient.h
	TransferFunctionWidget.h
	NodePropertiesWidget.h
	AppearanceDockWidget.h
	NodeItem.h
	EdgeItem.h
	Histogram.h
	Node.h

	# Camera
	Camera.h
	CameraWidget.h
	CameraDockWidget.h
	FilmWidget.h
	ApertureWidget.h
	ProjectionWidget.h
	FocusWidget.h
	Film.h
	Aperture.h
	Projection.h
	Focus.h

	# Presets
	PresetsWidget.h
	Preset.h

	# Statistics
	StatisticsDockWidget.h
	StatisticsWidget.h
	
	# Core
	Stable.h
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
	InteractorStyleRealisticCamera.h
	RenderThread.h
	SettingsDockWidget.h
	LoadSettingsDialog.h
	VtkWidget.h
	Scene.h
	Controls.h
	Material.h
	Sample.h
	LightTransport.h
	LogWidget.h
	StartupDialog.h
	Utilities.h

	# Logging
	Logger.h
	LogWidget.h
	LogDockWidget.h
)

SET(ExposureRenderMocHeaders 
	# Main
	MainWindow.h

	# Lighting
	LightSettingsWidget.h
	LightingDockWidget.h
	LightsWidget.h
	LightingWidget.h
	Lighting.h
	Background.h
	BackgroundIlluminationWidget.h

	# Appearance
	TransferFunction.h
	TransferFunctionView.h
	TransferFunctionWidget.h
	NodePropertiesWidget.h
	AppearanceDockWidget.h
	Histogram.h
	Node.h

	# Camera
	Camera.h
	CameraWidget.h
	CameraDockWidget.h
	FilmWidget.h
	ApertureWidget.h
	ProjectionWidget.h
	FocusWidget.h
	Film.h
	Aperture.h
	Projection.h
	Focus.h

	# Presets
	PresetsWidget.h
	Preset.h

	# Statistics
	StatisticsDockWidget.h
	StatisticsWidget.h
	
	# Core
	RenderThread.h
	SettingsDockWidget.h
	LoadSettingsDialog.h
	StartupDialog.h
	VtkWidget.h
	Controls.h

	# Logging
	Logger.h
	LogWidget.h
	LogDockWidget.h
)

# Resources
SET(ExposureRenderResources Icons/Icons.qrc)

# PATCHED QT4_WRAP_CPP for use with StdAfx
# Original source: Qt4Macros.cmake
MACRO (PCH_QT4_WRAP_CPP outfiles )
  # get include dirs
  QT4_GET_MOC_FLAGS(moc_flags)
  QT4_EXTRACT_OPTIONS(moc_files moc_options ${ARGN})
 
  FOREACH (it ${moc_files})
    
	GET_FILENAME_COMPONENT(it ${it} ABSOLUTE)
    QT4_MAKE_OUTPUT_FILE(${it} moc_ cxx outfile)
    set (moc_flags_append "-fStable.h" "-f${it}") # StdAfx hack.
 
	# message (STATUS "Moc Flags append ${moc_flags_append}")
	QT4_CREATE_MOC_COMMAND(${it} ${outfile} "${moc_flags};${moc_flags_append}" "${moc_options}")
	
	SET(${outfiles} ${${outfiles}} ${outfile})

  ENDFOREACH(it)
ENDMACRO (PCH_QT4_WRAP_CPP)

# QT4_CREATE_MOC_COMMAND(Icons/Icons.qrc qrc_Icons.cxx "${moc_flags};${moc_flags_append}" "${moc_options}")

# The rest should just work (sure...)
PCH_QT4_WRAP_CPP(MOCSrcs ${ExposureRenderMocHeaders} )
# QT4_ADD_RESOURCES(ResourceSrcs ${ExposureRenderResources})

SOURCE_GROUP("Resources" FILES
  ${ExposureRenderResources}
  ${EXE_ICON}
)

SOURCE_GROUP("Generated" FILES
	${UISrcs}
	${MOCSrcs}
	${ResourceSrcs}
	qrc_Icons.cxx
	${RCS_SOURCES}
)

# Main sources
SET(MainSources
	MainWindow.h
	MainWindow.cpp
	Main.cpp
)

# Make the main group
SOURCE_GROUP("Main" FILES ${MainSources})

# Appearance sources
SET(AppearanceSources
	AppearanceDockWidget.h
	TransferFunction.h
	TransferFunctionCanvas.h
	TransferFunctionView.h
	TransferFunctionGradient.h
	TransferFunctionWidget.h
	NodePropertiesWidget.h
	AppearanceDockWidget.h
	NodeItem.h
	EdgeItem.h
	Histogram.h
	Node.h
	AppearanceDockWidget.cpp
	TransferFunction.cpp
	TransferFunctionCanvas.cpp
	TransferFunctionView.cpp
	TransferFunctionGradient.cpp
	TransferFunctionWidget.cpp
	NodePropertiesWidget.cpp
	AppearanceDockWidget.cpp
	NodeItem.cpp
	EdgeItem.cpp
	Histogram.cpp
	Node.cpp
)

# Make the appearance group
SOURCE_GROUP("Appearance" FILES ${AppearanceSources})

# Lighting sources
SET(LightingSources
	LightSettingsWidget.h
	LightingDockWidget.h
	LightsWidget.h
	LightingWidget.h
	Lighting.h
	Background.h
	BackgroundIlluminationWidget.h
	LightSettingsWidget.cpp 
	LightingDockWidget.cpp 
	LightsWidget.cpp
	LightingWidget.cpp
	Lighting.cpp
	Background.cpp
	BackgroundIlluminationWidget.cpp
)

# Make the lighting group
SOURCE_GROUP("Lighting" FILES ${LightingSources})

# Camera sources
SET(CameraSources
	Camera.h
	CameraDockWidget.h
	FilmWidget.h
	ApertureWidget.h
	ProjectionWidget.h
	FocusWidget.h
	CameraWidget.h
	Film.h
	Aperture.h
	Projection.h
	Focus.h
	Camera.cpp
	CameraDockWidget.cpp
	FilmWidget.cpp
	ApertureWidget.cpp
	ProjectionWidget.cpp
	FocusWidget.cpp
	CameraWidget.cpp
	Film.cpp
	Aperture.cpp
	Projection.cpp
	Focus.cpp
)

# Make the camera group
SOURCE_GROUP("Camera" FILES ${CameraSources})

# Presets sources
SET(PresetsSources
	PresetsWidget.h
	Preset.h
	PresetsWidget.cpp
	Preset.cpp
)

# Make the presets group
SOURCE_GROUP("Presets" FILES ${PresetsSources})

# Statistics sources
SET(StatisticsSources
	StatisticsDockWidget.h
	StatisticsWidget.h
	StatisticsDockWidget.cpp
	StatisticsWidget.cpp
)

# Make the statistics group
SOURCE_GROUP("Statistics" FILES ${StatisticsSources})
	
# Core sources
SET(CoreSources
	Stable.h
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
	InteractorStyleRealisticCamera.h
	RenderThread.h
	SettingsDockWidget.h
	LoadSettingsDialog.h
	VtkWidget.h
	Scene.h
	Controls.h
	Material.h
	Sample.h
	LightTransport.h
	StartupDialog.h
	Utilities.h
	Stable.cpp
	Filter.cpp
	Flags.cpp
	Geometry.cpp
	Light.cpp
	MonteCarlo.cpp
	Random.cpp
	Spectrum.cpp
	Statistics.cpp
	VolumePoint.cpp
	InteractorStyleRealisticCamera.cpp
	RenderThread.cpp
	SettingsDockWidget.cpp
	LoadSettingsDialog.cpp
	VtkWidget.cpp
	Scene.cpp
	Controls.cpp
	Material.cpp
	Sample.cpp
	LightTransport.cpp
	StartupDialog.cpp
	Utilities.cpp
)

# Make the core group
SOURCE_GROUP("Core" FILES ${CoreSources})

# CUDA sources
SET(CudaSources
	SingleScattering.cuh
	MultipleScattering.cuh
	Blur.cuh
	Utilities.cuh
	Core.cuh
	ComputeEstimate.cuh
	Transport.cuh
	RayMarching.cuh
	Woodcock.cuh
	Denoise.cuh
	Core.cu
)

# Make the CUDA group
SOURCE_GROUP("CUDA" FILES ${CudaSources})

# Logging sources
SET(LoggingSources
	Logger.h
	LogWidget.h
	LogDockWidget.h
	Logger.cpp
	LogWidget.cpp
	LogDockWidget.cpp
)

# Make the logging group
SOURCE_GROUP("Logging" FILES ${LoggingSources})

ADD_DEFINITIONS(-DQT_GUI_LIBS -DQT_CORE_LIB -DQT_XML_LIB -DQT3_SUPPORT)

list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=compute_20")

# Create executable
CUDA_ADD_EXECUTABLE(ExposureRender MACOSX_BUNDLE ${ExposureRenderSources} ${UISrcs} ${MOCSrcs} ${ResourceSrcs} ${AppearanceSources} ${LightingSources} ${CameraSources} ${PresetsSources} ${CudaHeaders} ${CoreSources} ${CudaSources} ${LoggingSources})



TARGET_LINK_LIBRARIES(ExposureRender
  QVTK
  ${QT_LIBRARIES}
  vtkRendering
  vtkGraphics
  vtkIO
  vtkCommon

  vtkViews
)

### Precompiled headers ###
if (MSVC)
   set_target_properties(ExposureRender PROPERTIES COMPILE_FLAGS "/YuStable.h")
   set_source_files_properties(Stable.cpp PROPERTIES COMPILE_FLAGS "/YcStable.h")
  # set_source_files_properties(Stable.cpp PROPERTIES COMPILE_FLAGS "/YcUtilities.h")
endif(MSVC)

### VTK dependencies ###
FILE(GLOB VtkDlls "${VTK_DIR}/bin/RelWithDebInfo/*.dll")
INSTALL_FILES(/Bin FILES ${VtkDlls})

### CUDA dependencies ###
SET(CudaDlls ${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart32_40_17.dll)
INSTALL_FILES(/Bin FILES ${CudaDlls})

### QT dependencies ###
SET(QtDlls ${QT_BINARY_DIR}/QtCore4.dll ${QT_BINARY_DIR}/QtGui4.dll ${QT_BINARY_DIR}/QtXml4.dll ${QT_BINARY_DIR}/QtSql4.dll)
INSTALL_FILES(/Bin FILES ${QtDlls})

MESSAGE(STATUS "CMAKE_INSTALL_PREFIX_CACHED: " ${CMAKE_INSTALL_PREFIX_CACHED}/${CMAKE_BUILD_TYPE})


SET(CMAKE_INSTALL_PREFIX_CACHED ${CMAKE_INSTALL_PREFIX})
SET(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}")

MESSAGE(STATUS "CMAKE_INSTALL_PREFIX_CACHED: " ${CMAKE_INSTALL_PREFIX_CACHED})
MESSAGE(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})

# INSTALL_FILES("${CMAKE_CURRENT_BINARY_DIR}/ReadMe.txt" FILES "${CMAKE_CURRENT_BINARY_DIR}/Readme.txt")
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/Readme.txt COPYONLY)

### Put back the original cached prefix string ###
SET(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX_CACHED})

### Example .mhd and .raw files ###
# INSTALL_FILES(/Examples FILES "${CMAKE_CURRENT_BINARY_DIR}\\\\Bonsai.mhd")
# INSTALL_FILES(/Examples FILES "${CMAKE_CURRENT_BINARY_DIR}\\\\Bonsai.raw")

### Preset files ###
FILE(GLOB Presets "${CMAKE_CURRENT_SOURCE_DIR}/Presets/*.xml")
INSTALL_FILES(/Bin FILES ${Presets})

### Icon files ###
FILE(GLOB Icons "${CMAKE_CURRENT_SOURCE_DIR}/Icons/*.png")
INSTALL_FILES(/Bin/Icons FILES ${Icons})

### ReadMe and Copyright ###
# INSTALL_FILES(/ FILES ${CMAKE_CURRENT_SOURCE_DIR}\\\\ReadMe.txt)
INSTALL_FILES(/ FILES ${CMAKE_CURRENT_SOURCE_DIR}\\\\Copyright.txt)

INSTALL_TARGETS(/Bin ExposureRender)

# Requires re-distributables
INCLUDE(InstallRequiredSystemLibraries)

### Configure NSIS ###
SET(CPACK_GENERATOR NSIS)
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Exposure Render")
SET(CPACK_PACKAGE_VENDOR "TU Delft")
SET(CPACK_RESOURCE_FILE_WELCOME ${CMAKE_CURRENT_SOURCE_DIR}/Welcome.txt)
SET(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt)
SET(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/Copyright.txt)
SET(CPACK_PACKAGE_DEFAULT_LOCATION /bin)
SET(CPACK_PACKAGE_VERSION_MAJOR "1")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "0")

# Menu links
set(CPACK_NSIS_MENU_LINKS "http://graphics.tudelft.nl/" "TU Delft Graphics Website" "http://code.google.com/p/exposure-render/" "Exposure Render Website" "http://code.google.com/p/exposure-render/downloads" "Data sets" "http://code.google.com/p/exposure-render/wiki" "Wiki")

# Icon in the add/remove control panel, this must be an .exe file 
set(CPACK_NSIS_INSTALLED_ICON_NAME bin\\\\${EXE_TARGET_NAME}.exe)

# Install directory
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "Exposure Render")

# SET(CPACK_PACKAGE_ICON "c:\\WindowsMovieMaker.bmp")
SET(CPACK_NSIS_PACKAGE_NAME "Exposure Render")
SET(CPACK_NSIS_DISPLAY_NAME "Exposure Render")
SET(CPACK_NSIS_CONTACT "t.kroes@tudelft.nl")
SET(CPACK_NSIS_HELP_LINK "http://graphics.tudelft.nl/")
SET(CPACK_NSIS_URL_INFO_ABOUT "http://code.google.com/p/exposure-render/")
SET(CPACK_NSIS_MUI_FINISHPAGE_RUN "bin\\\\ExposureRender")

SET(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")

# Shortcut in start menu
SET(CPACK_PACKAGE_EXECUTABLES "bin\\\\ExposureRender;Exposure Render")

INCLUDE(CPack)


