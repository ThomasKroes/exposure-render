#pragma once

#include "Scene.h"

__device__ inline float Dens(CScene* pScene, const Vec3f& P, texture<short, 3, cudaReadModeNormalizedFloat>* pTexture)
{
	return ((float)(SHRT_MAX) * tex3D(*pTexture, P.x / pScene->m_BoundingBox.LengthX(), P.y / pScene->m_BoundingBox.LengthY(), P.z /  pScene->m_BoundingBox.LengthZ()));
}

__device__ inline Vec3f Grad(CScene* pScene, const Vec3f& P, texture<short, 3, cudaReadModeNormalizedFloat>* pTexture)
{
	Vec3f Normal;

	float Delta = 0.01f;

	Normal.x = Dens(pScene, P + Vec3f(Delta, 0.0f, 0.0f), pTexture) - Dens(pScene, P - Vec3f(Delta, 0.0f, 0.0f), pTexture);
	Normal.y = Dens(pScene, P + Vec3f(0.0f, Delta, 0.0f), pTexture) - Dens(pScene, P - Vec3f(0.0f, Delta, 0.0f), pTexture);
	Normal.z = Dens(pScene, P + Vec3f(0.0f, 0.0f, Delta), pTexture) - Dens(pScene, P - Vec3f(0.0f, 0.0f, Delta), pTexture);

	Normal.Normalize();

	return -Normal;
}