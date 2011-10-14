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

DEV CColorRgbHdr GetDiffuse(const float& NormalizedIntensity)
{
	float4 Diffuse = tex1D(gTexDiffuse, NormalizedIntensity);
	return CColorRgbHdr(Diffuse.x, Diffuse.y, Diffuse.z);
}

DEV CColorRgbHdr GetSpecular(const float& NormalizedIntensity)
{
	float4 Specular = tex1D(gTexSpecular, NormalizedIntensity);
	return CColorRgbHdr(Specular.x, Specular.y, Specular.z);
}

DEV float GetRoughness(const float& NormalizedIntensity)
{
	return tex1D(gTexRoughness, NormalizedIntensity);
}

DEV CColorRgbHdr GetEmission(const float& NormalizedIntensity)
{
	float4 Emission = tex1D(gTexEmission, NormalizedIntensity);
	return CColorRgbHdr(Emission.x, Emission.y, Emission.z);
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
			Hit		= true;
		}
	}
	
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

DEV CColorXyza CumulativeMovingAverage(const CColorXyza& A, const CColorXyza& Ax, const int& N)
{
	if (gNoIterations == 0)
		return CColorXyza(0.0f);

	 return A + ((Ax - A) / (float)N);
}