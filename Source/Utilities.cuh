#pragma once

#include "Scene.h"

#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_math.h>

DEV float GetDensity(CScene* pScene, const Vec3f& P)
{
	return (float)SHRT_MAX * tex3D(gTexDensity, P.x / pScene->m_BoundingBox.m_MaxP.x, P.y / pScene->m_BoundingBox.m_MaxP.y, P.z / pScene->m_BoundingBox.m_MaxP.z);
}

/*
DEV float GetOpacity(CScene* pScene, const Vec3f& P)
{
	return tex1D(gTexOpacity, GetDensity(pScene, P) / pScene->m_IntensityRange.GetLength());
}
*/

DEV float GetOpacity(CScene* pScene, const float& Density)
{
	return tex1D(gTexOpacity, (Density - pScene->m_IntensityRange.GetMin()) / pScene->m_IntensityRange.GetLength());
}

DEV CColorRgbHdr GetDiffuse(CScene* pScene, const Vec3f& P)
{
	float4 Diffuse = tex1D(gTexDiffuse, GetDensity(pScene, P) / pScene->m_IntensityRange.GetLength());
	return CColorRgbHdr(Diffuse.x, Diffuse.y, Diffuse.z);
}

DEV CColorRgbHdr GetDiffuse(CScene* pScene, const float& Density)
{
	float4 Diffuse = tex1D(gTexDiffuse, (Density - pScene->m_IntensityRange.GetMin()) / pScene->m_IntensityRange.GetLength());
	return CColorRgbHdr(Diffuse.x, Diffuse.y, Diffuse.z);
}

DEV CColorRgbHdr GetSpecular(CScene* pScene, const Vec3f& P)
{
	float4 Specular = tex1D(gTexSpecular, GetDensity(pScene, P) / pScene->m_IntensityRange.GetLength());
	return CColorRgbHdr(Specular.x, Specular.y, Specular.z);
}

DEV CColorRgbHdr GetSpecular(CScene* pScene, const float& Density)
{
	float4 Specular = tex1D(gTexSpecular, (Density - pScene->m_IntensityRange.GetMin()) / pScene->m_IntensityRange.GetLength());
	return CColorRgbHdr(Specular.x, Specular.y, Specular.z);
}

DEV float GetRoughness(CScene* pScene, const Vec3f& P)
{
	return tex1D(gTexRoughness, GetDensity(pScene, P) / pScene->m_IntensityRange.GetLength());
}

DEV float GetRoughness(CScene* pScene, const float& Density)
{
	return tex1D(gTexRoughness, (Density - pScene->m_IntensityRange.GetMin()) / pScene->m_IntensityRange.GetLength());
}

DEV CColorRgbHdr GetEmission(CScene* pScene, const Vec3f& P)
{
	float4 Emission = tex1D(gTexEmission, GetDensity(pScene, P) / pScene->m_IntensityRange.GetLength());
	return CColorRgbHdr(Emission.x, Emission.y, Emission.z);
}

DEV CColorRgbHdr GetEmission(CScene* pScene, const float& Density)
{
	float4 Emission = tex1D(gTexEmission, (Density - pScene->m_IntensityRange.GetMin()) / pScene->m_IntensityRange.GetLength());
	return CColorRgbHdr(Emission.x, Emission.y, Emission.z);
}

__device__ inline Vec3f NormalizedGradient(CScene* pScene, const Vec3f& P)
{
	Vec3f Gradient;

	float Delta = pScene->m_GradientDelta;

	Gradient.x = (GetDensity(pScene, P + Vec3f(Delta, 0.0f, 0.0f)) - GetDensity(pScene, P - Vec3f(Delta, 0.0f, 0.0f))) / Delta;
	Gradient.y = (GetDensity(pScene, P + Vec3f(0.0f, Delta, 0.0f)) - GetDensity(pScene, P - Vec3f(0.0f, Delta, 0.0f))) / Delta;
	Gradient.z = (GetDensity(pScene, P + Vec3f(0.0f, 0.0f, Delta)) - GetDensity(pScene, P - Vec3f(0.0f, 0.0f, Delta))) / Delta;

	Gradient.Normalize();

	return Gradient;
}

DEV float GradientMagnitude(CScene* pScene, const Vec3f& P)
{
	return ((float)SHRT_MAX * tex3D(gTexGradientMagnitude, P.x / pScene->m_BoundingBox.m_MaxP.x, P.y / pScene->m_BoundingBox.m_MaxP.y, P.z / pScene->m_BoundingBox.m_MaxP.z));
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

DEV int intersectBox(CRay r, Vec3f boxmin, Vec3f boxmax, float *tnear, float *tfar)
{
  Vec3f invR = Vec3f(1.0f) / r.m_D;
  Vec3f tbot = invR * (boxmin - r.m_O);
  Vec3f ttop = invR * (boxmax - r.m_O);

  Vec3f tmin(fminf(ttop.x, tbot.x), fminf(ttop.y, tbot.y), fminf(ttop.z, tbot.z));
  Vec3f tmax(fmaxf(ttop.x, tbot.x), fmaxf(ttop.y, tbot.y), fmaxf(ttop.z, tbot.z));

  float largest_tmin	= fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
  float smallest_tmax	= fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

  *tnear = largest_tmin;
  *tfar = smallest_tmax;

  return smallest_tmax > largest_tmin;
}