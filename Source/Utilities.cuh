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

DEV inline Vec3f ToVec3f(const float3& V)
{
	return Vec3f(V.x, V.y, V.z);
}

DEV float GetNormalizedIntensity(const Vec3f& P)
{
	const float Intensity = ((float)SHRT_MAX * tex3D(gTexDensity, P.x * gVolumeInfo.m_InvMaxAABB.x, P.y * gVolumeInfo.m_InvMaxAABB.y, P.z * gVolumeInfo.m_InvMaxAABB.z));

	return (Intensity - gVolumeInfo.m_IntensityMin) * gVolumeInfo.m_IntensityInvRange;
}

DEV float GetOpacity(const float& NormalizedIntensity)
{
	return tex1D(gTexOpacity, NormalizedIntensity);
//	return NormalizedIntensity > 0.1f ? 0.2f : 0.0f;//tex1D(gTexOpacity, NormalizedIntensity);
}

DEV float GetOpacity(const Vec3f& P)
{
	for (int i = 0; i < gSlicing.m_NoSlices; i++)
	{
		Vec3f D = P - ToVec3f(gSlicing.m_Position[i]);

		if (Dot(D, ToVec3f(gSlicing.m_Normal[i]) < 0.0f))
			return 0.0f;
	}

	if (P.x > 0.5f * gVolumeInfo.m_MaxAABB.x)
		return 0.0f;

	return GetOpacity(GetNormalizedIntensity(P));
}

DEV ColorXYZf GetDiffuse(const float& NormalizedIntensity)
{
	float4 Diffuse = tex1D(gTexDiffuse, NormalizedIntensity);
	return ColorXYZf(Diffuse.x, Diffuse.y, Diffuse.z);
}

DEV ColorXYZf GetSpecular(const float& NormalizedIntensity)
{
	float4 Specular = tex1D(gTexSpecular, NormalizedIntensity);
	return ColorXYZf(Specular.x, Specular.y, Specular.z);
}

DEV float GetGlossiness(const float& NormalizedIntensity)
{
	return tex1D(gTexGlossiness, NormalizedIntensity);
}

DEV ColorXYZf GetEmission(const float& NormalizedIntensity)
{
	float4 Emission = tex1D(gTexEmission, NormalizedIntensity);
	return ColorXYZf(Emission.x, Emission.y, Emission.z);
}

DEV inline Vec3f NormalizedGradient(const Vec3f& P)
{
	Vec3f Gradient;

	Gradient.x = (GetNormalizedIntensity(P + ToVec3f(gVolumeInfo.m_GradientDeltaX)) - GetNormalizedIntensity(P - ToVec3f(gVolumeInfo.m_GradientDeltaX))) * gVolumeInfo.m_InvGradientDelta;
	Gradient.y = (GetNormalizedIntensity(P + ToVec3f(gVolumeInfo.m_GradientDeltaY)) - GetNormalizedIntensity(P - ToVec3f(gVolumeInfo.m_GradientDeltaY))) * gVolumeInfo.m_InvGradientDelta;
	Gradient.z = (GetNormalizedIntensity(P + ToVec3f(gVolumeInfo.m_GradientDeltaZ)) - GetNormalizedIntensity(P - ToVec3f(gVolumeInfo.m_GradientDeltaZ))) * gVolumeInfo.m_InvGradientDelta;

	return Normalize(Gradient);
}

DEV float GradientMagnitude(const Vec3f& P)
{
	return ((float)SHRT_MAX * tex3D(gTexGradientMagnitude, P.x * gVolumeInfo.m_InvMaxAABB.x, P.y * gVolumeInfo.m_InvMaxAABB.y, P.z * gVolumeInfo.m_InvMaxAABB.z));
}

DEV bool IntersectBox(const CRay& R, float* pNearT, float* pFarT)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.m_D;
	const Vec3f BottomT		= InvR * (Vec3f(gVolumeInfo.m_MinAABB.x, gVolumeInfo.m_MinAABB.y, gVolumeInfo.m_MinAABB.z) - R.m_O);
	const Vec3f TopT		= InvR * (Vec3f(gVolumeInfo.m_MaxAABB.x, gVolumeInfo.m_MaxAABB.y, gVolumeInfo.m_MaxAABB.z) - R.m_O);
	const Vec3f MinT		= MinVec3f(TopT, BottomT);
	const Vec3f MaxT		= MaxVec3f(TopT, BottomT);
	const float LargestMinT = fmaxf(fmaxf(MinT.x, MinT.y), fmaxf(MinT.x, MinT.z));
	const float LargestMaxT = fminf(fminf(MaxT.x, MaxT.y), fminf(MaxT.x, MaxT.z));

	*pNearT = LargestMinT;
	*pFarT	= LargestMaxT;

	return LargestMaxT > LargestMinT;
}

DEV ColorXYZAf CumulativeMovingAverage(const ColorXYZAf& A, const ColorXYZAf& Ax, const int& N)
{
	 return A + ((Ax - A) / max((float)N, 1.0f));
}

DEV ColorXYZf CumulativeMovingAverage(const ColorXYZf& A, const ColorXYZf& Ax, const int& N)
{
	 return A + ((Ax - A) / max((float)N, 1.0f));
}