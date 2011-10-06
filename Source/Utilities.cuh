#pragma once

#include "Scene.h"

#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_math.h>

DEV float GetNormalizedIntensity(CScene* pScene, const Vec3f& P)
{
	const float Intensity = ((float)SHRT_MAX * tex3D(gTexDensity, P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));

	return (Intensity - gIntensityMin) * gIntensityInvRange;
}

DEV float GetOpacity(CScene* pScene, const float& NormalizedIntensity)
{
	return tex1D(gTexOpacity, NormalizedIntensity);
}

DEV CColorRgbHdr GetDiffuse(CScene* pScene, const float& NormalizedIntensity)
{
	float4 Diffuse = tex1D(gTexDiffuse, NormalizedIntensity);
	return CColorRgbHdr(Diffuse.x, Diffuse.y, Diffuse.z);
}

DEV CColorRgbHdr GetSpecular(CScene* pScene, const float& NormalizedIntensity)
{
	float4 Specular = tex1D(gTexSpecular, NormalizedIntensity);
	return CColorRgbHdr(Specular.x, Specular.y, Specular.z);
}

DEV float GetRoughness(CScene* pScene, const float& NormalizedIntensity)
{
	return tex1D(gTexRoughness, NormalizedIntensity);
}

DEV CColorRgbHdr GetEmission(CScene* pScene, const float& NormalizedIntensity)
{
	float4 Emission = tex1D(gTexEmission, NormalizedIntensity);
	return CColorRgbHdr(Emission.x, Emission.y, Emission.z);
}

__device__ inline Vec3f NormalizedGradient(CScene* pScene, const Vec3f& P)
{
	Vec3f Gradient;

	const float Delta = gGradientDelta;

	Gradient.x = (GetNormalizedIntensity(pScene, P + Vec3f(Delta, 0.0f, 0.0f)) - GetNormalizedIntensity(pScene, P - Vec3f(Delta, 0.0f, 0.0f))) * gInvGradientDelta;
	Gradient.y = (GetNormalizedIntensity(pScene, P + Vec3f(0.0f, Delta, 0.0f)) - GetNormalizedIntensity(pScene, P - Vec3f(0.0f, Delta, 0.0f))) * gInvGradientDelta;
	Gradient.z = (GetNormalizedIntensity(pScene, P + Vec3f(0.0f, 0.0f, Delta)) - GetNormalizedIntensity(pScene, P - Vec3f(0.0f, 0.0f, Delta))) * gInvGradientDelta;

	Gradient.Normalize();

	return Gradient;
}

DEV float GradientMagnitude(CScene* pScene, const Vec3f& P)
{
	return ((float)SHRT_MAX * tex3D(gTexGradientMagnitude, P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
}

DEV bool NearestLight(CScene* pScene, CRay R, CColorXyz& LightColor, Vec3f& Pl, CLight*& pLight, float* pPdf = NULL)
{
	bool Hit = false;
	
	float T = 0.0f;

	CRay RayCopy = R;

	for (int i = 0; i < pScene->m_Lighting.m_NoLights; i++)
	{
		if (pScene->m_Lighting.m_Lights[i].Intersect(RayCopy, T, LightColor, NULL, pPdf))
		{
			Pl		= R(T);
			pLight	= &pScene->m_Lighting.m_Lights[i];

			Hit = true;
		}
	}
	
	return Hit;
}

DEV bool IntersectBox(CRay R, float* pNearT, float* pFarT)
{
	const float3 InvR		= make_float3(1.0f, 1.0f, 1.0f) / make_float3(R.m_D.x, R.m_D.y, R.m_D.z);
	const float3 BottomT	= InvR * (gAaBbMin - make_float3(R.m_O.x, R.m_O.y, R.m_O.z));
	const float3 TopT		= InvR * (gAaBbMax - make_float3(R.m_O.x, R.m_O.y, R.m_O.z));
	const float3 MinT		= fminf(TopT, BottomT);
	const float3 MaxT		= fmaxf(TopT, BottomT);
	const float LargestMinT = fmaxf(fmaxf(MinT.x, MinT.y), fmaxf(MinT.x, MinT.z));
	const float LargestMaxT = fminf(fminf(MaxT.x, MaxT.y), fminf(MaxT.x, MaxT.z));

	*pNearT = LargestMinT;
	*pFarT	= LargestMaxT;

	return LargestMaxT > LargestMinT;
}