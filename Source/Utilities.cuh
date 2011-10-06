#pragma once

#include "Scene.h"

#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_math.h>

DEV float GetDensity(CScene* pScene, const Vec3f& P)
{
	return (float)SHRT_MAX * tex3D(gTexDensity, P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z);
}

DEV float GetOpacity(CScene* pScene, const Vec3f& P)
{
	return tex1D(gTexOpacity, GetDensity(pScene, P) * gIntensityInvRange);
}

DEV float GetOpacity(CScene* pScene, const float& Density)
{
	return tex1D(gTexOpacity, (Density - gIntensityMin) * gIntensityInvRange);
}

DEV CColorRgbHdr GetDiffuse(CScene* pScene, const Vec3f& P)
{
	float4 Diffuse = tex1D(gTexDiffuse, GetDensity(pScene, P) * gIntensityInvRange);
	return CColorRgbHdr(Diffuse.x, Diffuse.y, Diffuse.z);
}

DEV CColorRgbHdr GetDiffuse(CScene* pScene, const float& Density)
{
	float4 Diffuse = tex1D(gTexDiffuse, (Density - gIntensityMin) * gIntensityInvRange);
	return CColorRgbHdr(Diffuse.x, Diffuse.y, Diffuse.z);
}

DEV CColorRgbHdr GetSpecular(CScene* pScene, const Vec3f& P)
{
	float4 Specular = tex1D(gTexSpecular, GetDensity(pScene, P) * gIntensityInvRange);
	return CColorRgbHdr(Specular.x, Specular.y, Specular.z);
}

DEV CColorRgbHdr GetSpecular(CScene* pScene, const float& Density)
{
	float4 Specular = tex1D(gTexSpecular, (Density - gIntensityMin) * gIntensityInvRange);
	return CColorRgbHdr(Specular.x, Specular.y, Specular.z);
}

DEV float GetRoughness(CScene* pScene, const Vec3f& P)
{
	return tex1D(gTexRoughness, GetDensity(pScene, P) * gIntensityInvRange);
}

DEV float GetRoughness(CScene* pScene, const float& Density)
{
	return tex1D(gTexRoughness, (Density - gIntensityMin) * gIntensityInvRange);
}

DEV CColorRgbHdr GetEmission(CScene* pScene, const Vec3f& P)
{
	float4 Emission = tex1D(gTexEmission, GetDensity(pScene, P) * gIntensityInvRange);
	return CColorRgbHdr(Emission.x, Emission.y, Emission.z);
}

DEV CColorRgbHdr GetEmission(CScene* pScene, const float& Density)
{
	float4 Emission = tex1D(gTexEmission, (Density - gIntensityMin) * gIntensityInvRange);
	return CColorRgbHdr(Emission.x, Emission.y, Emission.z);
}

__device__ inline Vec3f NormalizedGradient(CScene* pScene, const Vec3f& P)
{
	Vec3f Gradient;

	const float Delta = gGradientDelta;

	Gradient.x = (GetDensity(pScene, P + Vec3f(Delta, 0.0f, 0.0f)) - GetDensity(pScene, P - Vec3f(Delta, 0.0f, 0.0f))) / Delta;
	Gradient.y = (GetDensity(pScene, P + Vec3f(0.0f, Delta, 0.0f)) - GetDensity(pScene, P - Vec3f(0.0f, Delta, 0.0f))) / Delta;
	Gradient.z = (GetDensity(pScene, P + Vec3f(0.0f, 0.0f, Delta)) - GetDensity(pScene, P - Vec3f(0.0f, 0.0f, Delta))) / Delta;

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