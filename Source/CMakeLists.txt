#	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
#	All rights reserved.
#
#	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#	
#	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 2.8)

PROJECT(ExposureRender)

### Export of symbols ###
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_EXPORTING")

### Set Build Type ###
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")

FIND_PACKAGE(CUDA)

# Supported streaming architectures
SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_10,code=sm_10;${CUDA_NVCC_FLAGS}")
SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_10,code=compute_10;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_11,code=sm_11;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_11,code=compute_11;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_12,code=sm_12;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_12,code=compute_12;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_13,code=sm_13;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_13,code=compute_13;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=sm_20;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=compute_20;${CUDA_NVCC_FLAGS}")

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

SET(QT_USE_QTXML TRUE)
SET(QT_USE_OPENGL TRUE)
SET(QT_USE_QTNETWORK TRUE)
SET(VTK_USE_QVTK_QTOPENGL)

FIND_PACKAGE(Qt4 REQUIRED)

# Enable the use of XML in Qt



INCLUDE(${QT_USE_FILE})



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
	
	ReadMe.txt
	Copyright.txt

	# Main
	MainWindow.cpp
	Main.cpp 
	LoadSettingsDialog.cpp
	StartupDialog.cpp
	AboutDialog.cpp
	SettingsDockWidget.cpp
	HardwareWidget.cpp
	UpdateWidget.cpp
	HttpGet.cpp

	# Lighting
	LightWidget.cpp 
	LightingDockWidget.cpp 
	LightsWidget.cpp
	LightingWidget.cpp
	Lighting.cpp
	Light.cpp
	Background.cpp
	BackgroundWidget.cpp

	# Appearance
	TransferFunction.cpp
	TransferFunctionView.cpp
	TransferFunctionWidget.cpp
	NodePropertiesWidget.cpp
	AppearanceDockWidget.cpp
	NodeItem.cpp
	EdgeItem.cpp
	Histogram.cpp
	Node.cpp
	GradientRamp.cpp
	HistogramItem.cpp
	BackgroundItem.cpp
	TransferFunctionItem.cpp
	NodeSelectionWidget.cpp
	GridItem.cpp
	AppearanceSettingsWidget.cpp
	Appearance.cpp

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
	
	# Slicing
	SlicingDockWidget.cpp
	SlicingWidget.cpp
	Slice.cpp

	# Core
	Stable.cpp
	Flags.cpp
	Geometry.cpp
	InteractorStyleRealisticCamera.cpp
	RenderThread.cpp
	VtkWidget.cpp
	Controls.cpp
	Utilities.cpp
	CudaUtilities.cpp
	Status.cpp
	Timing.cpp
	Variance.cpp
	RenderOutputDialog.cpp

	# Vtk
	vtkErVolumeMapper.cpp
	vtkErLight.cpp
	vtkErAreaLight.cpp
	vtkErBackgroundLight.cpp
	vtkErSlicePlane.cpp
	vtkErVolumeProperty.cpp
	vtkErVolumeInfo.cpp
	vtkErRenderInfo.cpp
	VtkErCamera.cpp
	VtkErAreaLightWidget.cpp
	vtkErPointWidget.cpp
	vtkErArrowSource.cpp
	VtkErVolume.cpp
	VtkErRenderCanvas.cpp

	# Logging
	Logger.cpp
	LogWidget.cpp
	LogDockWidget.cpp
)

SET(ExposureRenderHeaders 
	
	# Main
	MainWindow.h
	LoadSettingsDialog.h
	StartupDialog.h
	AboutDialog.h
	SettingsDockWidget.h
	HardwareWidget.h
	UpdateWidget.h
	HttpGet.h

	# Lighting
	LightWidget.h
	LightingDockWidget.h
	LightsWidget.h
	LightingWidget.h
	Lighting.h
	Light.h
	Background.h
	BackgroundWidget.h

	# Appearance
	TransferFunction.h
	TransferFunctionView.h
	TransferFunctionWidget.h
	NodePropertiesWidget.h
	AppearanceDockWidget.h
	NodeItem.h
	EdgeItem.h
	Histogram.h
	Node.h
	GradientRamp.h
	HistogramItem.h
	BackgroundItem.h
	TransferFunctionItem.h
	NodeSelectionWidget.h
	GridItem.h
	AppearanceSettingsWidget.h
	Appearance.h

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
	
	# Slicing
	SlicingDockWidget.h
	SlicingWidget.h
	Slice.h

	# Core
	Stable.h
	Defines.h
	Dll.h
	Enumerations.h
	Flags.h
	Geometry.h
	InteractorStyleRealisticCamera.h
	RenderThread.h
	VtkWidget.h
	Controls.h
	LogWidget.h
	Utilities.h
	CudaUtilities.h
	Status.h
	Timing.h
	Variance.h
	RenderOutputDialog.h

	# Vtk
	vtkErVolumeMapper.h
	vtkErLight.h
	vtkErAreaLight.h
	vtkErBackgroundLight.h
	vtkErSlicePlane.h
	vtkErVolumeProperty.h
	vtkErVolumeInfo.h
	vtkErRenderInfo.h
	VtkErCamera.h
	VtkErAreaLightWidget.h
	vtkErPointWidget.h
	vtkErArrowSource.h
	VtkErVolume.h
	VtkErRenderCanvas.cpp

	# Logging
	Logger.h
	LogWidget.h
	LogDockWidget.h
)

SET(ExposureRenderMocHeaders 
	# Main
	MainWindow.h
	SettingsDockWidget.h
	LoadSettingsDialog.h
	StartupDialog.h
	AboutDialog.h
	HardwareWidget.h
	UpdateWidget.h
	HttpGet.h

	# Lighting
	LightWidget.h
	LightingDockWidget.h
	LightsWidget.h
	LightingWidget.h
	Lighting.h
	Light.h
	Background.h
	BackgroundWidget.h

	# Appearance
	TransferFunction.h
	TransferFunctionView.h
	TransferFunctionWidget.h
	NodePropertiesWidget.h
	AppearanceDockWidget.h
	Histogram.h
	Node.h
	NodeSelectionWidget.h
	AppearanceSettingsWidget.h

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
	
	# Slicing
	SlicingDockWidget.h
	SlicingWidget.h
	Slice.h

	# Core
	VtkWidget.h
	Controls.h
	Status.h
	RenderOutputDialog.h

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
	LoadSettingsDialog.h
	StartupDialog.h
	AboutDialog.h
	SettingsDockWidget.h
	HardwareWidget.h
	HttpGet.h
	UpdateWidget.h
	MainWindow.cpp
	Main.cpp
	LoadSettingsDialog.cpp
	StartupDialog.cpp
	AboutDialog.cpp
	SettingsDockWidget.cpp
	HardwareWidget.cpp
	HttpGet.cpp
	UpdateWidget.cpp
)

# Make the main group
SOURCE_GROUP("Main" FILES ${MainSources})

# Appearance sources
SET(AppearanceSources
	AppearanceDockWidget.h
	TransferFunction.h
	TransferFunctionView.h
	TransferFunctionWidget.h
	NodePropertiesWidget.h
	AppearanceDockWidget.h
	NodeItem.h
	EdgeItem.h
	Histogram.h
	Node.h
	GradientRamp.h
	HistogramItem.h
	BackgroundItem.h
	TransferFunctionItem.h
	NodeSelectionWidget.h
	GridItem.h
	AppearanceSettingsWidget.h
	Appearance.h
	AppearanceDockWidget.cpp
	TransferFunction.cpp
	TransferFunctionView.cpp
	TransferFunctionWidget.cpp
	NodePropertiesWidget.cpp
	AppearanceDockWidget.cpp
	NodeItem.cpp
	EdgeItem.cpp
	Histogram.cpp
	Node.cpp
	GradientRamp.cpp
	HistogramItem.cpp
	BackgroundItem.cpp
	TransferFunctionItem.cpp
	NodeSelectionWidget.cpp
	GridItem.cpp
	AppearanceSettingsWidget.cpp
	Appearance.cpp
)

# Make the appearance group
SOURCE_GROUP("Appearance" FILES ${AppearanceSources})

# Lighting sources
SET(LightingSources
	LightWidget.h
	LightingDockWidget.h
	LightsWidget.h
	LightingWidget.h
	Lighting.h
	Light.h
	Background.h
	BackgroundWidget.h
	LightWidget.cpp 
	LightingDockWidget.cpp 
	LightsWidget.cpp
	LightingWidget.cpp
	Lighting.cpp
	Light.cpp
	Background.cpp
	BackgroundWidget.cpp
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

# Slicing sources
SET(SlicingSources
	SlicingDockWidget.h
	SlicingWidget.h
	Slice.h
	SlicingDockWidget.cpp
	SlicingWidget.cpp
	Slice.cpp
)

# Make the statistics group
SOURCE_GROUP("Slicing" FILES ${SlicingSources})

# Core sources
SET(CoreSources
	Stable.h
	Defines.h
	Dll.h
	Enumerations.h
	Flags.h
	Geometry.h
	InteractorStyleRealisticCamera.h
	RenderThread.h
	VtkWidget.h
	Controls.h
	Utilities.h
	Status.h
	Timing.h
	Variance.h
	RenderOutputDialog.h
	Stable.cpp
	Flags.cpp
	Geometry.cpp
	InteractorStyleRealisticCamera.cpp
	RenderThread.cpp
	VtkWidget.cpp
	Controls.cpp
	Utilities.cpp
	Status.cpp
	Timing.cpp
	Variance.cpp
	RenderOutputDialog.cpp
)

# Make the core group
SOURCE_GROUP("Core" FILES ${CoreSources})

# Vtk sources
SET(VtkSources
	vtkErVolumeMapper.h
	vtkErLight.h
	vtkErAreaLight.h
	vtkErBackgroundLight.h
	vtkErSlicePlane.h
	vtkErVolumeProperty.h
	vtkErVolumeInfo.h
	vtkErRenderInfo.h
	VtkErCamera.h
	VtkErAreaLightWidget.h
	vtkErPointWidget.h
	vtkErArrowSource.h
	VtkErVolume.h
	VtkErRenderCanvas.h
	vtkErVolumeMapper.cpp
	vtkErLight.cpp
	vtkErAreaLight.cpp
	vtkErBackgroundLight.cpp
	vtkErSlicePlane.cpp
	vtkErVolumeProperty.cpp
	vtkErVolumeInfo.cpp
	vtkErRenderInfo.cpp
	VtkErCamera.cpp
	VtkErAreaLightWidget.cpp
	vtkErPointWidget.cpp
	vtkErArrowSource.cpp
	VtkErVolume.cpp
	VtkErRenderCanvas.cpp
)

# Make the core group
SOURCE_GROUP("Vtk" FILES ${VtkSources})

# CUDA sources
SET(CudaSources
	SingleScattering.cuh
	MultipleScattering.cuh
	Blur.cuh
	Utilities.cuh
	Core.cuh
	Estimate.cuh
	Transport.cuh
	RayMarching.cuh
	Woodcock.cuh
	Denoise.cuh
	Variance.cuh
	NearestIntersection.cuh
	SpecularBloom.cuh
	ToneMap.cuh
	Shader.cuh
	Sample.cuh
	RNG.cuh
	Camera.cuh
	Lighting.cuh
	MonteCarlo.cuh
	Slice.cuh
	VolumeInfo.cuh
	RenderInfo.cuh
	Buffer.cuh
	Core.cu
	CudaUtilities.h
	CudaUtilities.cpp
)

# Make the CUDA group
SOURCE_GROUP("Cuda" FILES ${CudaSources})

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

IF(WIN32)
	SET(GUI_TYPE WIN32)
ENDIF(WIN32)

# Create executable
CUDA_ADD_EXECUTABLE(ExposureRender MACOSX_BUNDLE ${ExposureRenderSources} ${UISrcs} ${MOCSrcs} ${ResourceSrcs} ${VtkSources} ${MainSources} ${AppearanceSources} ${LightingSources} ${CameraSources} ${PresetsSources} ${CudaHeaders} ${CoreSources} ${CudaSources} ${LoggingSources} ${SlicingSources})



TARGET_LINK_LIBRARIES(ExposureRender
  QVTK
  ${QT_LIBRARIES}
  vtkRendering
  vtkGraphics
  vtkIO
  vtkWidgets
  vtkCommon
  vtkVolumeRendering
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
INSTALL_FILES(/bin FILES ${VtkDlls})

### CUDA dependencies ###
SET(CudaDlls ${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart32_40_17.dll)
INSTALL_FILES(/bin FILES ${CudaDlls})

### QT core dlls ###
SET(QtCoreDlls ${QT_BINARY_DIR}/QtCore4.dll ${QT_BINARY_DIR}/QtGui4.dll ${QT_BINARY_DIR}/QtXml4.dll ${QT_BINARY_DIR}/QtSql4.dll ${QT_BINARY_DIR}/QtNetwork4.dll)
INSTALL_FILES(/bin FILES ${QtCoreDlls})

### Qt Image format dlls ###
FILE(GLOB QtImageFormatDlls "${QT_BINARY_DIR}/../plugins/imageformats/*.dll")
INSTALL_FILES(/bin FILES ${QtImageFormatDlls})

### Example .mhd and .raw files ###
FILE(GLOB Examples "${CMAKE_CURRENT_SOURCE_DIR}/Examples/*.*")
INSTALL_FILES(/bin/Examples FILES ${Examples})

### Preset files ###
FILE(GLOB Presets "${CMAKE_CURRENT_SOURCE_DIR}/Presets/*.xml")
INSTALL_FILES(/bin FILES ${Presets})

### Icon files ###
FILE(GLOB Icons "${CMAKE_CURRENT_SOURCE_DIR}/Icons/*.png")
INSTALL_FILES(/bin/Icons FILES ${Icons})

### ReadMe and Copyright ###
INSTALL_FILES(/bin FILES ${CMAKE_CURRENT_SOURCE_DIR}\\\\ReadMe.txt)
INSTALL_FILES(/bin FILES ${CMAKE_CURRENT_SOURCE_DIR}\\\\Copyright.txt)

#### Install Exposure Render ###
INSTALL_TARGETS(/bin ExposureRender)

# Requires re-distributables
INCLUDE(InstallRequiredSystemLibraries)

### Configure NSIS ###
SET(CPACK_GENERATOR NSIS)
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Exposure Render")
SET(CPACK_PACKAGE_VENDOR "TU Delft")
SET(CPACK_RESOURCE_FILE_WELCOME ${CMAKE_CURRENT_SOURCE_DIR}/Welcome.txt)
SET(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt)
SET(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/Copyright.txt)
SET(CPACK_PACKAGE_DEFAULT_LOCATION /)
SET(CPACK_PACKAGE_VERSION_MAJOR "1")
SET(CPACK_PACKAGE_VERSION_MINOR "1")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(CPACK_PACKAGE_ICON ${CMAKE_CURRENT_SOURCE_DIR}\\\\installer.bmp)

# Menu links
set(CPACK_NSIS_MENU_LINKS "http://graphics.tudelft.nl/" "TU Delft Graphics Website" "http://code.google.com/p/exposure-render/" "Exposure Render Website" "http://code.google.com/p/exposure-render/downloads" "Data sets" "http://code.google.com/p/exposure-render/wiki" "Wiki")

# Icon in the add/remove control panel, this must be an .exe file 
set(CPACK_NSIS_INSTALLED_ICON_NAME ${EXE_TARGET_NAME}.exe)

### Install directory ###
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "Exposure Render")

# SET(SIDEBAR_IMAGE ${CMAKE_SOURCE_DIR}/install${LAST_PATH_DELIMITER}sidebar.bmp)
# SET(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "!define MUI_WELCOMEFINISHPAGE_BITMAP \\\"${SIDEBAR_IMAGE}\\\")

SET(CPACK_NSIS_PACKAGE_NAME "Exposure Render")
SET(CPACK_NSIS_DISPLAY_NAME "Exposure Render")
SET(CPACK_NSIS_CONTACT "t.kroes@tudelft.nl")
SET(CPACK_NSIS_HELP_LINK "http://graphics.tudelft.nl/")
SET(CPACK_NSIS_URL_INFO_ABOUT "http://code.google.com/p/exposure-render/")
SET(CPACK_NSIS_MUI_FINISHPAGE_RUN "ExposureRender")

SET(CPACK_NSIS_EXECUTABLES_DIRECTORY "bin")

# Shortcut in start menu
SET(CPACK_PACKAGE_EXECUTABLES "ExposureRender;Exposure Render")

INCLUDE(CPack)