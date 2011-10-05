#pragma once

#include "Geometry.h"
#include "Scene.h"

DEV inline bool SampleDistanceRM(CRay& R, CRNG& RNG, Vec3f& P, CScene* pScene)
{
	float MinT = 0.0f, MaxT = 0.0f;

	if (!pScene->m_BoundingBox.Intersect(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
	MaxT = min(MaxT, R.m_MaxT);

	float S			= -log(RNG.Get1()) / pScene->m_IntensityRange.GetLength();
	float Dt		= pScene->m_StepSizeFactor * pScene->m_GradientDelta;
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;
	float D			= 0.0f;

	Vec3f Ps; 

	MinT += RNG.Get1() * Dt;

	while (Sum < S)
	{
		Ps = R.m_O + MinT * R.m_D;

		if (MinT > MaxT)
			return false;
		
		SigmaT	= pScene->m_DensityScale * GetOpacity(pScene, GetDensity(pScene, Ps));

		Sum		+= SigmaT * Dt;
		MinT	+= Dt;
	}

	P = Ps;

	return true;
}

DEV inline bool FreePathRM(CRay R, CRNG& RNG, Vec3f& P, CScene* pScene)
{
	float MinT = 0.0f, MaxT = 0.0f;

	if (!pScene->m_BoundingBox.Intersect(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
	MaxT = min(MaxT, R.m_MaxT);

	float S			= -log(RNG.Get1()) / pScene->m_IntensityRange.GetLength();
	float Dt		= pScene->m_StepSizeFactorShadow * pScene->m_GradientDelta;
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;
	float D			= 0.0f;

	Vec3f Ps; 

	MinT += RNG.Get1() * Dt;

	while (Sum < S)
	{
		Ps = R.m_O + MinT * R.m_D;

		if (MinT > MaxT)
			return false;
		
		SigmaT	= pScene->m_DensityScale * GetOpacity(pScene, GetDensity(pScene, Ps));

		Sum		+= SigmaT * Dt;
		MinT	+= Dt;
	}

	P = Ps;

	return true;
}