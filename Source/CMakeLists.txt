
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
	LightingSettingsWidget.cpp 
	LightingDockWidget.cpp 
	LightsWidget.cpp
	LightingWidget.cpp
	Lighting.cpp
	BackgroundIlluminationWidget.cpp
	Camera.cpp
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
	CameraDockWidget.cpp
	FilmWidget.cpp
	ApertureWidget.cpp
	ProjectionWidget.cpp
	FocusWidget.cpp
	LoadVolume.cpp
	InteractorStyleRealisticCamera.cpp
	RenderThread.cpp
	SettingsDockWidget.cpp
	LoadSettingsDialog.cpp
	VtkWidget.cpp
	Scene.cpp
	TransferFunction.cpp
	TransferFunctionCanvas.cpp
	TransferFunctionView.cpp
	TransferFunctionGradient.cpp
	TransferFunctionWidget.cpp
	NodePropertiesWidget.cpp
	NodeItem.cpp
	AppearanceDockWidget.cpp
	PresetsWidget.cpp
	Preset.cpp
	Controls.cpp
	EdgeItem.cpp
)

SET(SimpleViewUI SimpleView.ui)
SET(SimpleViewHeaders 
	MainWindow.h
	LightingSettingsWidget.h
	LightingDockWidget.h
	LightsWidget.h
	LightingWidget.h
	Lighting.h
	BackgroundIlluminationWidget.h
	Camera.h
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
	CameraWidget.h
	CameraDockWidget.h
	FilmWidget.h
	ApertureWidget.h
	ProjectionWidget.h
	FocusWidget.h
	LoadVolume.h
	InteractorStyleRealisticCamera.h
	RenderThread.h
	SettingsDockWidget.h
	LoadSettingsDialog.h
	VtkWidget.h
	Scene.h
	TransferFunction.h
	TransferFunctionCanvas.h
	TransferFunctionView.h
	TransferFunctionGradient.h
	TransferFunctionWidget.h
	NodePropertiesWidget.h
	NodeItem.h
	AppearanceDockWidget.h
	PresetsWidget.h
	Preset.h
	Controls.h
	EdgeItem.h
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



























