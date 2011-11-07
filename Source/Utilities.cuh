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

DEV float GetNormalizedIntensity(const Vec3f& P, VolumeInfo& VI)
{
	const float Intensity = ((float)SHRT_MAX * tex3D(gTexDensity, P.x * VI.m_InvMaxAABB.x, P.y * VI.m_InvMaxAABB.y, P.z * VI.m_InvMaxAABB.z));

	return (Intensity - VI.m_IntensityMin) * VI.m_IntensityInvRange;
}

DEV float GetOpacity(const float& NormalizedIntensity)
{
	return tex1D(gTexOpacity, NormalizedIntensity);
}

DEV float GetOpacity(const Vec3f& P, VolumeInfo& VI)
{
//	if (!gSlicing.Contains(P))
//		return 0.0f;
//	else
		return GetOpacity(GetNormalizedIntensity(P, VI));
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

DEV float GetRoughness(const float& NormalizedIntensity)
{
	return tex1D(gTexRoughness, NormalizedIntensity);
}

DEV ColorXYZf GetEmission(const float& NormalizedIntensity)
{
	float4 Emission = tex1D(gTexEmission, NormalizedIntensity);
	return ColorXYZf(Emission.x, Emission.y, Emission.z);
}

DEV inline Vec3f NormalizedGradient(const Vec3f& P, VolumeInfo& VI)
{
	Vec3f Gradient;

	Gradient.x = (GetNormalizedIntensity(P + ToVec3f(VI.m_GradientDeltaX), VI) - GetNormalizedIntensity(P - ToVec3f(VI.m_GradientDeltaX), VI)) * VI.m_InvGradientDelta;
	Gradient.y = (GetNormalizedIntensity(P + ToVec3f(VI.m_GradientDeltaY), VI) - GetNormalizedIntensity(P - ToVec3f(VI.m_GradientDeltaY), VI)) * VI.m_InvGradientDelta;
	Gradient.z = (GetNormalizedIntensity(P + ToVec3f(VI.m_GradientDeltaZ), VI) - GetNormalizedIntensity(P - ToVec3f(VI.m_GradientDeltaZ), VI)) * VI.m_InvGradientDelta;

	return Normalize(Gradient);
}

DEV float GradientMagnitude(const Vec3f& P, VolumeInfo& VI)
{
	return ((float)SHRT_MAX * tex3D(gTexGradientMagnitude, P.x * VI.m_InvMaxAABB.x, P.y * VI.m_InvMaxAABB.y, P.z * VI.m_InvMaxAABB.z));
}

DEV bool IntersectBox(const CRay& R, float* pNearT, float* pFarT, VolumeInfo& VI)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.m_D;
	const Vec3f BottomT		= InvR * (Vec3f(VI.m_MinAABB.x, VI.m_MinAABB.y, VI.m_MinAABB.z) - R.m_O);
	const Vec3f TopT		= InvR * (Vec3f(VI.m_MaxAABB.x, VI.m_MaxAABB.y, VI.m_MaxAABB.z) - R.m_O);
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