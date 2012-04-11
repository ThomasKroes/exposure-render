/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#ifdef __CUDA_ARCH__
	#include <host_defines.h>
#endif

#include <float.h>
#include <algorithm>
#include <math.h>

using namespace std;

#ifdef _EXPORTING
	#define EXPOSURE_RENDER_DLL    __declspec(dllexport)
#else
	#define EXPOSURE_RENDER_DLL    __declspec(dllimport)
#endif

namespace ExposureRender
{

#ifdef __CUDA_ARCH__
	#define KERNEL						__global__
	#define HOST						__host__
	#define DEVICE						__device__
	#define DEVICE_NI					DEVICE __noinline__
	#define HOST_DEVICE					HOST DEVICE 
	#define HOST_DEVICE_NI				HOST_DEVICE __noinline__
	#define CD							__device__ __constant__
#else
	#define KERNEL
	#define HOST
	#define DEVICE
	#define DEVICE_NI
	#define HOST_DEVICE
	#define HOST_DEVICE_NI
	#define CD
#endif

#define PI_F						3.141592654f	
#define HALF_PI_F					0.5f * PI_F
#define QUARTER_PI_F				0.25f * PI_F
#define TWO_PI_F					2.0f * PI_F
#define INV_PI_F					0.31830988618379067154f
#define INV_TWO_PI_F				0.15915494309189533577f
#define FOUR_PI_F					4.0f * PI_F
#define INV_FOUR_PI_F				1.0f / FOUR_PI_F
#define	EULER_F						2.718281828f
#define RAD_F						57.29577951308232f
#define TWO_RAD_F					2.0f * RAD_F
#define DEG_TO_RAD					1.0f / RAD_F
#define METRO_SIZE					256
#define	RAY_EPS						0.0001f
#define RAY_EPS_2					2.0f * RAY_EPS
#define ONE_OVER_6					1.0f / 2.0f
#define ONE_OVER_255				1.0f / 255.0f
#define	MAX_CHAR_SIZE				256
#define MAX_NO_VOLUMES				64
#define MAX_NO_OBJECTS				64
#define MAX_NO_CLIPPING_OBJECTS		64
#define MAX_NO_TEXTURES				64
#define	MAX_NO_TIMINGS				64
#define MAX_NO_LIGHTS				64
#define MAX_NO_TF_NODES				128
#define NO_COLOR_COMPONENTS			4

}