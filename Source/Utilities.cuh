#pragma once

#include "Scene.h"





__device__ inline float Dens(CScene* pScene, const Vec3f& P, texture<short, 3, cudaReadModeNormalizedFloat>* pTexture)
{
	return ((float)(SHRT_MAX) * tex3D(*pTexture, P.x / pScene->m_BoundingBox.LengthX(), P.y / pScene->m_BoundingBox.LengthY(), P.z /  pScene->m_BoundingBox.LengthZ()));
}

__device__ inline Vec3f NormalizedGradient(CScene* pScene, const Vec3f& P, texture<short, 3, cudaReadModeNormalizedFloat>* pTexture)
{
	Vec3f Normal;

	float Delta = 0.01f;

	Normal.x = Dens(pScene, P + Vec3f(Delta, 0.0f, 0.0f), pTexture) - Dens(pScene, P - Vec3f(Delta, 0.0f, 0.0f), pTexture);
	Normal.y = Dens(pScene, P + Vec3f(0.0f, Delta, 0.0f), pTexture) - Dens(pScene, P - Vec3f(0.0f, Delta, 0.0f), pTexture);
	Normal.z = Dens(pScene, P + Vec3f(0.0f, 0.0f, Delta), pTexture) - Dens(pScene, P - Vec3f(0.0f, 0.0f, Delta), pTexture);

	Normal.Normalize();

	return -Normal;
}

DEV float Density(CScene* pScene, const Vec3f& P)
{
	return ((float)(SHRT_MAX) * tex3D(gTexDensity, P.x / pScene->m_BoundingBox.LengthX(), P.y / pScene->m_BoundingBox.LengthY(), P.z /  pScene->m_BoundingBox.LengthZ()));
}

DEV float Extinction(CScene* pScene, const Vec3f& P)
{
	return tex3D(gTexExtinction, P.x / pScene->m_BoundingBox.LengthX(), P.y / pScene->m_BoundingBox.LengthY(), P.z /  pScene->m_BoundingBox.LengthZ());
}

DEV CColorRgbHdr GetOpacity(CScene* pScene, const float& D)
{
	return pScene->m_DensityScale * pScene->m_TransferFunctions.m_Opacity.F(D);
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