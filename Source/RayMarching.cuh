#pragma once

#include "Geometry.h"
#include "Scene.h"

DEV inline bool SampleDistanceRM(CRay& R, CCudaRNG& RNG, Vec3f& P, CScene* pScene, int Component)
{
	float MinT = 0.0f, MaxT = 0.0f;

	if (!pScene->m_BoundingBox.Intersect(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
	MaxT = min(MaxT, R.m_MaxT);

	const float StepSizeFactor = 3.0f;//pScene->GetNoIterations() == 1 ? 2.0f : 2.0f;

	float S			= -log(RNG.Get1()) / pScene->m_IntensityRange.GetLength();
	float Dt		= StepSizeFactor * (1.0f / ((float)pScene->m_Resolution.GetResX()));
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;

	Vec3f samplePos; 

	MinT += RNG.Get1() * Dt;

	while (Sum < S)
	{
		samplePos = R.m_O + MinT * R.m_D;

		if (MinT > MaxT)
			return false;
		
		SigmaT	= GetOpacity(pScene, Density(pScene, samplePos)).r;
		Sum		+= SigmaT * Dt;
		MinT	+= Dt;
	}

	P = samplePos;

	return true;
}

DEV inline bool FreePathRM(CRay R, CCudaRNG& RNG, Vec3f& P, CScene* pScene, int Component)
{
// 	if (pScene->GetNoIterations() == 1)
// 		return false;

	float MinT = 0.0f, MaxT = 0.0f;

	if (!pScene->m_BoundingBox.Intersect(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
	MaxT = min(MaxT, R.m_MaxT);

	const float StepSizeFactor = 5.0f;//pScene->GetNoIterations() == 1 ? 2.0f : 2.0f;

	float S			= -log(RNG.Get1()) / pScene->m_IntensityRange.GetLength();
	float Dt		= StepSizeFactor * (1.0f / ((float)pScene->m_Resolution.GetResX()));
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;

	Vec3f samplePos; 

	MinT += RNG.Get1() * Dt;

	while (Sum < S)
	{
		samplePos = R.m_O + MinT * R.m_D;

		// Free path, no collisions in between
		if (MinT > MaxT)
			return false;
		
		SigmaT	= GetOpacity(pScene, Density(pScene, samplePos)).r;
		Sum		+= SigmaT * Dt;
		MinT	+= Dt;
	}

	P = samplePos;

	return true;
}