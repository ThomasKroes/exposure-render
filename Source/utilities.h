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

#include <cuda_runtime.h>

#include "montecarlo.h"
										
namespace ExposureRender
{

HOST_DEVICE float GlossinessExponent(const float& Glossiness)
{
	return 1000000.0f * powf(Glossiness, 7);
}

HOST_DEVICE_NI Vec3f ToVec3f(float3 V)
{
	return Vec3f(V.x, V.y, V.z);
}

HOST_DEVICE_NI Vec3f ToVec3f(float V[3])
{
	return Vec3f(V[0], V[1], V[2]);
}

HOST_DEVICE_NI float3 FromVec3f(Vec3f V)
{
	return make_float3(V[0], V[1], V[2]);
}

HOST_DEVICE_NI ColorXYZf ToColorXYZf(float V[3])
{
	return ColorXYZf(V[0], V[1], V[2]);
}

HOST_DEVICE float G(Vec3f P1, Vec3f N1, Vec3f P2, Vec3f N2)
{
	const Vec3f W = Normalize(P2 - P1);
	return (ClampedDot(W, N1) * ClampedDot(-1.0f * W, N2)) / DistanceSquared(P1, P2);
}

HOST_DEVICE ColorXYZAf CumulativeMovingAverage(const ColorXYZAf& A, const ColorXYZAf& Ax, const int& N)
{
	return A + (Ax - A) / max((float)N, 1.0f);
}

HOST_DEVICE ColorXYZf CumulativeMovingAverage(const ColorXYZf& A, const ColorXYZf& Ax, const int& N)
{
	 return A + ((Ax - A) / max((float)N, 1.0f));
}

}
