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

FIND_PACKAGE(CUDA)

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
SET(General
	device.h
	cuda.h
	utilities.h
	geometry.h
	transport.h
	raymarching.h
	shader.h
	sample.h
	rng.h
	montecarlo.h
	ray.h
	volumes.h
	lights.h
	objects.h
	clippingobjects.h
	textures.h
	transferfunctions.h
	filter.h
	log.h
	defines.h
	camera.h
	procedural.h
	shape.h
	matrix.h
	vector.h
	buffer1d.h
	buffer2d.h
	buffer3d.h
	boundingbox.h
	transferfunction.h
	rendersettings.h
	timing.h
	exception.h
	enums.h
	color.h
	exposurerender.h
	exposurerender.cpp
	tracer.h
	volume.h
	light.h
	scatterevent.h
	intersection.h
	object.h
	clippingobject.h
	texture.h
	bitmap.h
	buffer.h
	framebuffer.h
)

# General group
SOURCE_GROUP("General" FILES ${General})

# Shapes
SET(Shapes
	shapes.h
	plane.h
	box.h
	disk.h
	ring.h
	sphere.h
	cylinder.h
)

# Shapes group
SOURCE_GROUP("Shapes" FILES ${Shapes})

# API
SET(Bindable
	erbindable.h
	ertracer.h
	ervolume.h
	erlight.h
	erobject.h
	erclippingobject.h
	ertexture.h
	erbitmap.h
)

# API group
SOURCE_GROUP("Bindable" FILES ${Bindable})

# Filters
SET(Cuda
	singlescattering.cuh
	estimate.cuh
	gradientmagnitude.cuh
	bilateralfilter.cuh
	gaussianfilter.cuh
	tonemap.cuh
	autofocus.cuh
	list.cuh
	wrapper.cuh
	macros.cuh
	core.cu
)

# Filters group
SOURCE_GROUP("Cuda" FILES ${Cuda})

# Make the library
CUDA_ADD_LIBRARY(ErCore ${General} ${Shapes} ${Bindable} ${Cuda} SHARED)