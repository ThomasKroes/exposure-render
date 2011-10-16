/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <host_defines.h>
#include <float.h>

#pragma warning(disable : 4244)
#pragma warning(disable : 4251)
#pragma warning(disable : 4297)

#define HO													__host__
#define DEV													__device__
#define HOD													__host__ __device__
#define CD													__device__ __constant__
#define PI_F												3.141592654f	
#define HALF_PI_F											0.5f * PI_F
#define QUARTER_PI_F										0.25f * PI_F
#define TWO_PI_F											2.0f * PI_F
#define INV_PI_F											0.31830988618379067154f
#define INV_TWO_PI_F										0.15915494309189533577f
#define FOUR_PI_F											4.0f * PI_F
#define INV_4_PI_F											1.0f / FOUR_PI_F
#define	EULER_F												2.718281828f
#define RAD_F												57.29577951308232f
#define TWO_RAD_F											2.0f * RAD_F
#define DEG_TO_RAD											1.0f / RAD_F
#define	INF_MIN												-FLT_MAX
#define	INF_MAX												FLT_MAX
#define	RAY_MIN												-100000.0f
#define	RAY_MAX												100000.0f
#define EULER_E_F											2.71828182845904523536f
#define MAX_CHAR_SIZE										128
#define MAX_CHAR_SIZE_FILE_TYPE								8
#define MAX_CHAR_SIZE_FILE_EXTENSION						8
#define MAX_CHAR_SIZE_FILE_NAME								8
#define MAX_CHAR_SIZE_FILE_PATH								128
#define NUM_ALLOCATION_SIZES								10
#define KERNEL												__global__
#define IMPORT_PROGRESS_UPDATE_INTERVAL						500
#define MAX_BXDFS											4
#define NO_HIT												-1
#define NO_NODE_ID											-1
#define HISTOGRAM_NUM_BINS									250
#define MDH_LINE_SIZE										500
#define WHITESPACE											" \t\n\r"
#define MAX_NO_TF_POINTS									20
#define MAX_NO_VOLUME_LIGHTS								3
#define	MAX_BOKEH_DATA										12
#define MB													powf(1024.0f, 2.0f)

#define PRINT_CUDA_ERROR(call) {                                    \
	cudaError err = call;                                                    \
	if( cudaSuccess != err) {                                                \
	Log("Cuda error in file '%s' in line %i : %s.\n",        \
	__FILE__, __LINE__, cudaGetErrorString( err) );              \
	exit(EXIT_FAILURE);                                                  \
	} }