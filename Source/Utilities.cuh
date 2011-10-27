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

#include <cuda_runtime.h>

DEV inline Vec3f ToVec3f(const float3& V)
{
	return Vec3f(V.x, V.y, V.z);
}

DEV float GetNormalizedIntensity(const Vec3f& P)
{
	const float Intensity = ((float)SHRT_MAX * tex3D(gTexDensity, P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));

	return (Intensity - gIntensityMin) * gIntensityInvRange;
}

DEV float GetOpacity(const float& NormalizedIntensity)
{
	return tex1D(gTexOpacity, NormalizedIntensity);
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

DEV inline Vec3f NormalizedGradient(const Vec3f& P)
{
	Vec3f Gradient;

	Gradient.x = (GetNormalizedIntensity(P + ToVec3f(gGradientDeltaX)) - GetNormalizedIntensity(P - ToVec3f(gGradientDeltaX))) * gInvGradientDelta;
	Gradient.y = (GetNormalizedIntensity(P + ToVec3f(gGradientDeltaY)) - GetNormalizedIntensity(P - ToVec3f(gGradientDeltaY))) * gInvGradientDelta;
	Gradient.z = (GetNormalizedIntensity(P + ToVec3f(gGradientDeltaZ)) - GetNormalizedIntensity(P - ToVec3f(gGradientDeltaZ))) * gInvGradientDelta;

	return Normalize(Gradient);
}

DEV float GradientMagnitude(const Vec3f& P)
{
	return ((float)SHRT_MAX * tex3D(gTexGradientMagnitude, P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
}

DEV bool NearestLight(CScene* pScene, CRay R, ColorXYZf& LightColor, Vec3f& Pl, CLight*& pLight, float* pPdf = NULL)
{
	bool Hit = false;
	
	float T = 0.0f;

	CRay RayCopy = R;

	float Pdf = 0.0f;

	for (int i = 0; i < pScene->m_Lighting.m_NoLights; i++)
	{
		if (pScene->m_Lighting.m_Lights[i].Intersect(RayCopy, T, LightColor, NULL, &Pdf))
		{
			Pl		= R(T);
			pLight	= &pScene->m_Lighting.m_Lights[i];
			Hit		= true;
		}
	}
	
	if (pPdf)
		*pPdf = Pdf;

	return Hit;
}

DEV bool IntersectBox(const CRay& R, float* pNearT, float* pFarT)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.m_D;
	const Vec3f BottomT		= InvR * (Vec3f(gAaBbMin.x, gAaBbMin.y, gAaBbMin.z) - R.m_O);
	const Vec3f TopT		= InvR * (Vec3f(gAaBbMax.x, gAaBbMax.y, gAaBbMax.z) - R.m_O);
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