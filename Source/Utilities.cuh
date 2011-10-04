#pragma once

#include "Scene.h"

#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_math.h>

DEV float Density(CScene* pScene, const Vec3f& P)
{
	return (float)SHRT_MAX * tex3D(gTexDensity, P.x / pScene->m_BoundingBox.m_MaxP.x, P.y / pScene->m_BoundingBox.m_MaxP.y, P.z / pScene->m_BoundingBox.m_MaxP.z);
}

__device__ inline Vec3f NormalizedGradient(CScene* pScene, const Vec3f& P)
{
	Vec3f Gradient;

	float Delta = pScene->m_GradientDelta;

	Gradient.x = (Density(pScene, P + Vec3f(Delta, 0.0f, 0.0f)) - Density(pScene, P - Vec3f(Delta, 0.0f, 0.0f))) / Delta;
	Gradient.y = (Density(pScene, P + Vec3f(0.0f, Delta, 0.0f)) - Density(pScene, P - Vec3f(0.0f, Delta, 0.0f))) / Delta;
	Gradient.z = (Density(pScene, P + Vec3f(0.0f, 0.0f, Delta)) - Density(pScene, P - Vec3f(0.0f, 0.0f, Delta))) / Delta;

	Gradient.Normalize();

	return Gradient;
}

DEV float GradientMagnitude(CScene* pScene, const Vec3f& P)
{
	return ((float)SHRT_MAX * tex3D(gTexGradientMagnitude, P.x / pScene->m_BoundingBox.m_MaxP.x, P.y / pScene->m_BoundingBox.m_MaxP.y, P.z / pScene->m_BoundingBox.m_MaxP.z));
}

DEV float GetOpacity(CScene* pScene, const Vec3f& P)
{
	return tex1D(gTexOpacity, 255.0f * tex3D(gTexDensity, P.x / pScene->m_BoundingBox.m_MaxP.x, P.y / pScene->m_BoundingBox.m_MaxP.y, P.z / pScene->m_BoundingBox.m_MaxP.z)).x;
}

DEV CColorRgbHdr GetDiffuse(CScene* pScene, const float& D)
{
	return pScene->m_TransferFunctions.m_Diffuse.F(D);
}

DEV CColorRgbHdr GetSpecular(CScene* pScene, const float& D)
{
	return pScene->m_TransferFunctions.m_Specular.F(D);
}

DEV CColorRgbHdr GetEmission(CScene* pScene, const float& D)
{
	return pScene->m_TransferFunctions.m_Emission.F(D);
}

DEV CColorRgbHdr GetRoughness(CScene* pScene, const float& D)
{
	return pScene->m_TransferFunctions.m_Roughness.F(D);
}

DEV bool NearestLight(CScene* pScene, CRay R, CColorXyz& LightColor, Vec3f& Pl, CLight*& pLight, float* pPdf = NULL)
{
	// Whether a hit with a light was found or not 
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
  // compute intersection of ray with all six bbox planes
  Vec3f invR = Vec3f(1.0f) / r.m_D;
  Vec3f tbot = invR * (boxmin - r.m_O);
  Vec3f ttop = invR * (boxmax - r.m_O);

  // re-order intersections to find smallest and largest on each axis
  Vec3f tmin(fminf(ttop.x, tbot.x), fminf(ttop.y, tbot.y), fminf(ttop.z, tbot.z));
  Vec3f tmax(fmaxf(ttop.x, tbot.x), fmaxf(ttop.y, tbot.y), fmaxf(ttop.z, tbot.z));

  // find the largest tmin and the smallest tmax
  float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
  float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

  *tnear = largest_tmin;
  *tfar = smallest_tmax;

  return smallest_tmax > largest_tmin;
}