#pragma once

#include "Geometry.h"
#include "Scene.h"
#include "CudaUtilities.h"

DEV inline bool SampleDistanceRM(CRay& R, CRNG& RNG, Vec3f& Ps)
{
	float MinT = 0.0f, MaxT = 0.0f;

	if (!IntersectBox(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
	MaxT = min(MaxT, R.m_MaxT);

	const float S	= -log(RNG.Get1()) * gIntensityInvRange;
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;

	MinT += RNG.Get1() * gStepSize;

	while (Sum < S)
	{
		Ps = R.m_O + MinT * R.m_D;

		if (MinT > MaxT)
			return false;
		
		SigmaT	= gDensityScale * GetOpacity(GetNormalizedIntensity(Ps));

		Sum		+= SigmaT * gStepSize;
		MinT	+= gStepSize;
	}

	return true;
}

DEV inline bool FreePathRM(CRay& R, CRNG& RNG)
{
	float MinT = 0.0f, MaxT = 0.0f;

	if (!IntersectBox(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
	MaxT = min(MaxT, R.m_MaxT);

	const float S	= -log(RNG.Get1()) * gIntensityInvRange;
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;

	Vec3f Ps; 

	MinT += RNG.Get1() * gStepSizeShadow;

	while (Sum < S)
	{
		Ps = R.m_O + MinT * R.m_D;

		if (MinT > MaxT)
			return false;
		
		SigmaT	= gDensityScale * GetOpacity(GetNormalizedIntensity(Ps));

		Sum		+= SigmaT * gStepSizeShadow;
		MinT	+= gStepSizeShadow;
	}

	return true;
}

DEV inline bool NearestIntersection(CRay R, CScene* pScene, float& T)
{
	float MinT = 0.0f, MaxT = 0.0f;

	if (!IntersectBox(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
	MaxT = min(MaxT, R.m_MaxT);

	Vec3f Ps; 

	T = MinT;

	while (T < MaxT)
	{
		Ps = R.m_O + T * R.m_D;

		if (GetOpacity(GetNormalizedIntensity(Ps)) > 0.0f)
			return true;

		T += gStepSize;
	}

	return false;
}