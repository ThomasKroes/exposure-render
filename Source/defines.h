/*
    Exposure Render: An interactive photo-realistic volume rendering framework
    Copyright (C) 2011 Thomas Kroes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
#define MAX_NO_TF_NODES				128
#define NO_COLOR_COMPONENTS			4

	/*

#define VEC3_CONSTRUCTOR(classname, type)													\
HOST_DEVICE classname(const type& V1, const type& V2, const type& V3)						\
{																							\
	this->D[0] = V1;																		\
	this->D[1] = V2;																		\
	this->D[2] = V3;																		\
}

#define GET(name, type)					\
virtual type Get##name()				\
{										\
	return this->name;					\
}

#define GET_REF(name, type)				\
virtual type& Get##name()				\
{										\
	return this->name;					\
}

#define SET(name, type)					\
virtual void Set##name(type arg)		\
{										\
	if (this->name != arg)				\
		this->name = arg;				\
}

#define GET_SET(name, type)				\
GET(name, type)							\
SET(name, type)
*/

}
