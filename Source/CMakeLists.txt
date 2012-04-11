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

# Exposure Render core project
PROJECT(ExposureRender)

# Find CUDA
FIND_PACKAGE(CUDA)

# Find VTK
#IF(NOT VTK_BINARY_DIR)
#	FIND_PACKAGE(VTK)
#	IF(NOT VTK_DIR)
#	 MESSAGE(FATAL_ERROR "Please set VTK_DIR.")
#	ENDIF(NOT VTK_DIR)
#	INCLUDE(${VTK_USE_FILE})
#ENDIF(NOT VTK_BINARY_DIR)

# Supported streaming architectures
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_10,code=sm_10;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_10,code=compute_10;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_11,code=sm_11;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_11,code=compute_11;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_12,code=sm_12;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_12,code=compute_12;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_13,code=sm_13;${CUDA_NVCC_FLAGS}")
# SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_13,code=compute_13;${CUDA_NVCC_FLAGS}")
SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=sm_20;${CUDA_NVCC_FLAGS}")
SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=compute_20;${CUDA_NVCC_FLAGS}")
#SET(CUDA_NVCC_FLAGS "-OPT:Olimit=99999;${CUDA_NVCC_FLAGS}")

# Add CUDA includes
INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CUDA_SDK_ROOT_DIR}/common/inc
	${CUDA_TOOLKIT_INCLUDE}
)

# Export symbols
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_EXPORTING")

# Make the "Core" source group
SOURCE_GROUP("Core" FILES ${vtkErCoreSources})

# General
SET(ErCudaGeneral
	SingleScattering.cuh
	Utilities.cuh
	Geometry.cuh
	Core.cuh
	Estimate.cuh
	Transport.cuh
	RayMarching.cuh
	Shader.cuh
	Sample.cuh
	RNG.cuh
	MonteCarlo.cuh
	Buffer.cuh
	Framebuffer.cuh
	Color.cuh
	Ray.cuh
	Statistics.cuh
	GradientMagnitude.cuh
	AutoFocus.cuh
	Tracer.cuh
	Camera.cuh
	Volume.cuh
	Light.cuh
	Object.cuh
	ClippingObject.cuh
	Texture.cuh
	Shape.cuh
	TransferFunction.cuh
	Core.cu
)

# General group
SOURCE_GROUP("General" FILES ${ErCudaGeneral})

# Shapes
SET(ErCudaShapes
	Plane.cuh
	Box.cuh
	Disk.cuh
	Ring.cuh
	Sphere.cuh
	Cylinder.cuh
)

# Shapes group
SOURCE_GROUP("Shapes" FILES ${ErCudaShapes})

# Benchmark
SET(ErBenchmark
	Benchmark.cuh
)

# Benchmark group
SOURCE_GROUP("Benchmark" FILES ${ErBenchmark})

# API
SET(ErAPI
	ExposureRender.h
	Defines.h
	Camera.h
	Volume.h
	Light.h
	Object.h
	ClippingObject.h
	Procedural.h
	Image.h
	Texture.h
	Shape.h
	Matrix.h
	Vector.h
	TransferFunction.h
	RenderSettings.h
	Timing.h
	Exception.h
	Enums.h
	CudaUtilities.h
)

# API group
SOURCE_GROUP("API" FILES ${ErAPI})

# Shared group
SOURCE_GROUP("Shared" FILES ${ErShared})

# Filters
SET(ErCudaFilters
	Filter.cuh
	BilateralFilter.cuh
	GaussianFilter.cuh
	Blend.cuh
	ToneMap.cuh
)

# Filters group
SOURCE_GROUP("Filters" FILES ${ErCudaFilters})

# Make the library
CUDA_ADD_LIBRARY(ErCore ${ErCudaGeneral} ${ErCudaShapes} ${ErBenchmark} ${ErAPI} ${ErShared} ${ErCudaFilters} SHARED)

# This is hack, we need a more elegant solution for this!
ADD_CUSTOM_COMMAND(TARGET ErCore POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/relwithdebinfo/ErCore.dll C:/Workspaces/Voreen/voreen/modules/er/ext/er/bin)
ADD_CUSTOM_COMMAND(TARGET ErCore POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/relwithdebinfo/ErCore.lib C:/Workspaces/Voreen/voreen/modules/er/ext/er/lib)
ADD_CUSTOM_COMMAND(TARGET ErCore POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/General.cuh C:/Workspaces/Voreen/voreen/modules/er/ext/er/include)
ADD_CUSTOM_COMMAND(TARGET ErCore POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/Core.cuh C:/Workspaces/Voreen/voreen/modules/er/ext/er/include)
