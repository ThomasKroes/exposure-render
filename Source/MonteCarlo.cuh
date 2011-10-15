#pragma once

#include "Geometry.h"

#include "RNG.cuh"

DEV inline float CumulativeMovingAverage(const float i, const float Ai, const float Xi)
{
	return Ai + (Xi - Ai) / (i+1);
}

/**
	@brief Generate a 2D stratified sample
	@param[in] Pass Pass ID
	@param[in] U Random input
	@param[in] NumX Kernel size X
	@param[in] NumY Kernel size Y
	@return Stratified sample
*/
HOD inline Vec2f StratifiedSample2D(const int& Pass, const Vec2f& U, const int& NumX = 4, const int& NumY = 4)
{
	const float Dx	= 1.0f / (float)NumX;
	const float Dy	= 1.0f / (float)NumY;

	const int Y	= (int)((float)Pass / (float)NumX);
	const int X	= Pass - (Y * NumX);
	
	return Vec2f((float)(X + U.x) * Dx, (float)(Y + U.y) * Dy);
}

/**
	@brief Generate a 2D stratified sample
	@param[in] StratumX Stratum X
	@param[in] StratumY Stratum Y
	@param[in] U Random input
	@param[in] NumX Kernel size X
	@param[in] NumY Kernel size Y
	@return Stratified sample
*/
HOD inline Vec2f StratifiedSample2D(const int& StratumX, const int& StratumY, const Vec2f& U, const int& NumX = 4, const int& NumY = 4)
{
	const float Dx	= 1.0f / ((float)NumX);
	const float Dy	= 1.0f / ((float)NumY);

	return Vec2f((float)(StratumX + U.x) * Dx, (float)(StratumY + U.y) * Dy);
}

/**
	@brief Convert a given vector from world coordinates to local coordinates
	@param[in] W Vector in world coordinates
	@param[in] N Normal vector in world coordinates
	@return Vector in world coordinates
*/
HOD inline Vec3f WorldToLocal(const Vec3f& W, const Vec3f& N)
{
	const Vec3f U = Normalize(Cross(N, Vec3f(0.0072f, 0.0034f, 1.0f)));
	const Vec3f V = Normalize(Cross(N, U));
	
	return Vec3f(Dot(W, U), Dot(W, V), Dot(W, N));
}

/**
	@brief Convert a given vector from local coordinates to world coordinates
	@param[in] W Vector in local coordinates
	@param[in] N Normal vector in world coordinates
	@return Vector in world coordinates
*/
HOD inline Vec3f LocalToWorld(const Vec3f& W, const Vec3f& N)
{
	const Vec3f U = Normalize(Cross(N, Vec3f(0.0072f, 0.0034f, 1.0f)));
	const Vec3f V = Normalize(Cross(N, U));

	return Vec3f(	U.x * W.x + V.x * W.y + N.x * W.z,
						U.y * W.x + V.y * W.y + N.y * W.z,
						U.z * W.x + V.z * W.y + N.z * W.z);
}

/**
	@brief Convert a given vector from world coordinates to local coordinates
	@param[in] W Vector in world coordinates
	@param[in] N Normal vector in world coordinates
	@return Vector in world coordinates
*/
HOD inline Vec3f WorldToLocal(const Vec3f& U, const Vec3f& V, const Vec3f& N, const Vec3f& W)
{
	return Vec3f(Dot(W, U), Dot(W, V), Dot(W, N));
}

/**
	@brief Convert a given vector from local coordinates to world coordinates
	@param[in] W Vector in local coordinates
	@param[in] N Normal vector in world coordinates
	@return Vector in world coordinates
*/
HOD inline Vec3f LocalToWorld(const Vec3f& U, const Vec3f& V, const Vec3f& N, const Vec3f& W)
{
	return Vec3f(U.x * W.x + V.x * W.y + N.x * W.z,	U.y * W.x + V.y * W.y + N.y * W.z, U.z * W.x + V.z * W.y + N.z * W.z);
}

/**
	@brief Computes the spherical theta
	@param[in] Wl Vector in local coordinates
	@return Spherical theta
*/
HOD inline float SphericalTheta(const Vec3f& Wl)
{
	return acosf(Clamp(Wl.y, -1.f, 1.f));
}

/**
	@brief Computes the spherical phi
	@param[in] Wl Vector in local coordinates
	@return Spherical phi
*/
HOD inline float SphericalPhi(const Vec3f& Wl)
{
	float p = atan2f(Wl.z, Wl.x);
	return (p < 0.f) ? p + 2.f * PI_F : p;
}

/**
	@brief Computes the cosine of theta (latitude), given a spherical coordinate
	@param[in] Ws Spherical coordinate
	@return Cosine of theta
*/
HOD inline float CosTheta(const Vec3f& Ws)
{
	return Ws.z;
}

/**
	@brief Computes the absolute cosine of theta (latitude), given a spherical coordinate
	@param[in] Ws Spherical coordinate
	@return Absolute cosine of theta
*/
HOD inline float AbsCosTheta(const Vec3f &Ws)
{
	return fabsf(CosTheta(Ws));
}

/**
	@brief Computes the sine of phi (latitude), given a spherical coordinate
	@param[in] Ws Spherical coordinate
	@return Sine of theta
*/
HOD inline float SinTheta(const Vec3f& Ws)
{
	return sqrtf(max(0.f, 1.f - Ws.z * Ws.z));
}

/**
	@brief Computes the squared cosine of theta (latitude), given a spherical coordinate
	@param[in] Ws Spherical coordinate
	@return Squared cosine of theta
*/
HOD inline float SinTheta2(const Vec3f& Ws)
{
	return 1.f - CosTheta(Ws) * CosTheta(Ws);
}

/**
	@brief Computes the cosine of phi (longitude), given a spherical coordinate
	@param[in] Ws Spherical coordinate
	@return Cosine of phi
*/
HOD inline float CosPhi(const Vec3f& Ws)
{
	return Ws.x / SinTheta(Ws);
}

/**
	@brief Computes the sine of phi (longitude), given a spherical coordinate
	@param[in] Ws Spherical coordinate
	@return Sine of phi
*/
HOD inline float SinPhi(const Vec3f& Ws)
{
	return Ws.y / SinTheta(Ws);
}

/**
	@brief Determines whether two vectors reside in the same hemisphere
	@param[in] Ww1 First vector in world coordinates
	@param[in] Ww2 First vector in world coordinates
	@return Whether two given vectors reside in the same hemisphere
*/
HOD inline bool SameHemisphere(const Vec3f& Ww1, const Vec3f& Ww2)
{
   return Ww1.z * Ww2.z > 0.0f;
}

/**
	@brief Determines whether two vectors reside in the same hemisphere
	@param[in] W1 First vector in world coordinates
	@param[in] W2 First vector in world coordinates
	@param[in] N First vector in world coordinates
	@return Whether two given vectors reside in the same hemisphere
*/
HOD inline bool SameHemisphere(const Vec3f& W1, const Vec3f& W2, const Vec3f& N)
{
   return (Dot(W1, N) * Dot(W2, N)) >= 0.0f;
}

/**
	@brief Determines whether two vectors reside in the same shading hemisphere
	@param[in] W1 First vector in world coordinates
	@param[in] W2 First vector in world coordinates
	@param[in] N Normal vector in world coordinates
	@return Whether two given vectors reside in the same shading hemisphere
*/
HOD inline bool InShadingHemisphere(const Vec3f& W1, const Vec3f& W2, const Vec3f& N)
{
   return Dot(W1, N) >= 0.0f && Dot(W2, N) >= 0.0f;
}

/**
	@brief Generates a uniform sample in a disk
	@param[in] U Random input
	@return Uniform sample in a disk
*/
HOD inline Vec2f UniformSampleDisk(const Vec2f& U)
{
	float r = sqrtf(U.x);
	float theta = 2.0f * PI_F * U.y;
	return Vec2f(r * cosf(theta), r * sinf(theta));
}

/**
	@brief Generates a uniform sample in a disk
	@param[in] U Random input
	@return Uniform sample in a disk
*/
HOD inline Vec3f UniformSampleDisk(const Vec2f& U, const Vec3f& N)
{
	const Vec2f UV = UniformSampleDisk(U);

	Vec3f Ucs, Vcs;

	CreateCS(N, Ucs, Vcs);

	return (UV.x * Ucs) + (UV.y * Vcs);
}

/**
	@brief Generates a concentric sample in a disk
	@param[in] U Random input
	@return Concentric sample in a disk
*/
HOD inline Vec2f ConcentricSampleDisk(const Vec2f& U)
{
	float r, theta;
	// Map uniform random numbers to $[-1,1]^2$
	float sx = 2 * U.x - 1;
	float sy = 2 * U.y - 1;
	// Map square to $(r,\theta)$
	// Handle degeneracy at the origin
	
	if (sx == 0.0 && sy == 0.0)
	{
		return Vec2f(0.0f);
	}

	if (sx >= -sy)
	{
		if (sx > sy)
		{
			// Handle first region of disk
			r = sx;
			if (sy > 0.0)
				theta = sy/r;
			else
				theta = 8.0f + sy/r;
		}
		else
		{
			// Handle second region of disk
			r = sy;
			theta = 2.0f - sx/r;
		}
	}
	else
	{
		if (sx <= sy)
		{
			// Handle third region of disk
			r = -sx;
			theta = 4.0f - sy/r;
		}
		else
		{
			// Handle fourth region of disk
			r = -sy;
			theta = 6.0f + sx/r;
		}
	}
	
	theta *= PI_F / 4.f;

	return Vec2f(r*cosf(theta), r*sinf(theta));
}

/**
	@brief Generates a cosine weighted hemispherical sample
	@param[in] U Random input
	@return Cosine weighted hemispherical sample
*/
HOD inline Vec3f CosineWeightedHemisphere(const Vec2f& U)
{
	const Vec2f ret = ConcentricSampleDisk(U);
	return Vec3f(ret.x, ret.y, sqrtf(max(0.f, 1.f - ret.x * ret.x - ret.y * ret.y)));
}

/**
	@brief Generates a cosine weighted hemispherical sample in world coordinates
	@param[in] U Random input
	@param[in] Wow Vector in world coordinates
	@param[in] Wow Normal in world coordinates
	@return Cosine weighted hemispherical sample in world coordinates
*/
HOD inline Vec3f CosineWeightedHemisphere(const Vec2f& U, const Vec3f& N)
{
	const Vec3f Wl = CosineWeightedHemisphere(U);

	const Vec3f u = Normalize(Cross(N, N));
	const Vec3f v = Normalize(Cross(N, u));

	return Vec3f(	u.x * Wl.x + v.x * Wl.y + N.x * Wl.z,
						u.y * Wl.x + v.y * Wl.y + N.y * Wl.z,
						u.z * Wl.x + v.z * Wl.y + N.z * Wl.z);
}

/**
	@brief Computes the probability from a cosine weighted hemispherical sample
	@param[in] CosTheta Cosine of theta (Latitude)
	@param[in] Phi Phi (Longitude)
	@return Probability from a cosine weighted hemispherical sample
*/
HOD inline float CosineWeightedHemispherePdf(const float& CosTheta, const float& Phi)
{
	return CosTheta * INV_PI_F;
}

/**
	@brief Generates a spherical sample
	@param[in] SinTheta Sine of theta (Latitude)
	@param[in] CosTheta Cosine of theta (Latitude)
	@param[in] Phi Phi (Longitude)
	@return Spherical sample
*/
HOD inline Vec3f SphericalDirection(const float& SinTheta, const float& CosTheta, const float& Phi)
{
	return Vec3f(SinTheta * cosf(Phi), SinTheta * sinf(Phi), CosTheta);
}

HOD inline Vec3f SphericalDirection(float sintheta, float costheta, float phi, const Vec3f& x, const Vec3f& y, const Vec3f& z)
{
	return sintheta * cosf(phi) * x + sintheta * sinf(phi) * y + costheta * z;
}

/**
	@brief Generates a cosine weighted hemispherical sample in world coordinates
	@param[in] U Random input
	@param[in] Wow Vector in world coordinates
	@param[in] Wow Normal in world coordinates
	@return Cosine weighted hemispherical sample in world coordinates
*/
HOD inline Vec3f SphericalDirection(const float& SinTheta, const float& CosTheta, const float& Phi, const Vec3f& N)
{
	const Vec3f Wl = SphericalDirection(SinTheta, CosTheta, Phi);

	const Vec3f u = Normalize(Cross(N, Vec3f(0.0072f, 1.0f, 0.0034f)));
	const Vec3f v = Normalize(Cross(N, u));

	return Vec3f(	u.x * Wl.x + v.x * Wl.y + N.x * Wl.z,
						u.y * Wl.x + v.y * Wl.y + N.y * Wl.z,
						u.z * Wl.x + v.z * Wl.y + N.z * Wl.z);
}

/**
	@brief Generates a sample in a triangle
	@param[in] U Random input
	@return Sample in a triangle
*/
HOD inline Vec2f UniformSampleTriangle(const Vec2f& U)
{
	float su1 = sqrtf(U.x);

	return Vec2f(1.0f - su1, U.y * su1);
}

/**
	@brief Generates a sample in a sphere
	@param[in] U Random input
	@return Sample in a sphere
*/
HOD inline Vec3f UniformSampleSphere(const Vec2f& U)
{
	float z = 1.f - 2.f * U.x;
	float r = sqrtf(max(0.f, 1.f - z*z));
	float phi = 2.f * PI_F * U.y;
	float x = r * cosf(phi);
	float y = r * sinf(phi);
	return Vec3f(x, y, z);
}

/**
	@brief Generates a hemispherical sample
	@param[in] U Random input
	@return Sample in a hemisphere
*/
HOD inline Vec3f UniformSampleHemisphere(const Vec2f& U)
{
	float z = U.x;
	float r = sqrtf(max(0.f, 1.f - z*z));
	float phi = 2 * PI_F * U.y;
	float x = r * cosf(phi);
	float y = r * sinf(phi);
	return Vec3f(x, y, z);
}

/**
	@brief Generates a hemispherical sample in world coordinates
	@param[in] U Random input
	@param[in] N Normal in world coordinates
	@return Hemispherical sample in world coordinates
*/
DEV inline Vec3f UniformSampleHemisphere(const Vec2f& U, const Vec3f& N)
{
	const Vec3f Wl = UniformSampleHemisphere(U);

	const Vec3f u = Normalize(Cross(N, Vec3f(0.0072f, 1.0f, 0.0034f)));
	const Vec3f v = Normalize(Cross(N, u));

	return Vec3f(	u.x * Wl.x + v.x * Wl.y + N.x * Wl.z,
						u.y * Wl.x + v.y * Wl.y + N.y * Wl.z,
						u.z * Wl.x + v.z * Wl.y + N.z * Wl.z);
}

/**
	@brief Generates a sample in a cone
	@param[in] U Random input
	@param[in] CosThetaMax Maximum cone angle
	@return Sample in a cone
*/
HOD inline Vec3f UniformSampleCone(const Vec2f& U, const float& CosThetaMax)
{
	float costheta = Lerp(U.x, CosThetaMax, 1.f);
	float sintheta = sqrtf(1.f - costheta*costheta);
	float phi = U.y * 2.f * PI_F;
	return Vec3f(cosf(phi) * sintheta, sinf(phi) * sintheta, costheta);
}

/**
	@brief Generates a sample in a cone
	@param[in] U Random input
	@param[in] CosThetaMax Maximum cone angle
	@param[in] N Normal
	@return Sample in a cone
*/
HOD inline Vec3f UniformSampleCone(const Vec2f& U, const float& CosThetaMax, const Vec3f& N)
{
	const Vec3f Wl = UniformSampleCone(U, CosThetaMax);

	const Vec3f u = Normalize(Cross(N, Vec3f(0.0072f, 1.0f, 0.0034f)));
	const Vec3f v = Normalize(Cross(N, u));

	return Vec3f(	u.x * Wl.x + v.x * Wl.y + N.x * Wl.z,
						u.y * Wl.x + v.y * Wl.y + N.y * Wl.z,
						u.z * Wl.x + v.z * Wl.y + N.z * Wl.z);
}

/**
	@brief Computes the PDF of a sample in a cone
	@param[in] CosThetaMax Maximum cone angle
	@return PDF of a sample in a cone
*/
HOD inline float UniformConePdf(float CosThetaMax)
{
	return 1.f / (2.f * PI_F * (1.f - CosThetaMax));
}

/**
	@brief Computes the probability of a spherical sample
	@return Probability of a spherical sample
*/
HOD inline float UniformSpherePdf(void)
{
	return 1.0f / (4.0f * PI_F);
}

/**
	@brief Uniformly samples a point on a triangle
	@param[in] pIndicesV Vertex indices
	@param[in] pVertices Vertices
	@param[in] pIndicesVN Vertex normal indices
	@param[in] pVertexNormals Vertex normals
	@param[in] SampleTriangleIndex Index of the triangle to sample
	@param[in] U Random input
	@param[in][out] N Sampled normal 
	@param[in] UV Sampled texture coordinates
	@return Probability of a spherical sample
*/
HOD inline Vec3f UniformSampleTriangle(Vec4i* pIndicesV, Vec3f* pVertices, Vec4i* pIndicesVN, Vec3f* pVertexNormals, int SampleTriangleIndex, Vec2f U, Vec3f& N, Vec2f& UV)
{
	const Vec4i Face = pIndicesV[SampleTriangleIndex];

	const Vec3f P[3] = { pVertices[Face.x], pVertices[Face.y], pVertices[Face.z] };

	UV = UniformSampleTriangle(U);

	const float B0 = 1.0f - UV.x - UV.y;

	const Vec3f VN[3] = 
	{
		pVertexNormals[pIndicesVN[SampleTriangleIndex].x],
		pVertexNormals[pIndicesVN[SampleTriangleIndex].y],
		pVertexNormals[pIndicesVN[SampleTriangleIndex].z]	
	};

	N = Normalize(B0 * VN[0] + UV.x * VN[1] + UV.y * VN[2]);

	return B0 * P[0] + UV.x * P[1] + UV.y * P[2];
}


/**
	@brief P. Shirley's concentric disk algorithm, maps square to disk
	@param[in] U Random input
	@param[out] u Output u coordinate
	@param[out] v Output v coordinate
*/
HOD inline void ShirleyDisk(const Vec2f& U, float& u, float& v)
{
	float phi = 0, r = 0, a = 2 * U.x - 1, b = 2 * U.y - 1;
	
	if (a >- b)
	{
		if (a > b)
		{	
			// Reg.1
			r = a;
			phi = QUARTER_PI_F * (b / a);
		}
		else
		{			
			// Reg.2
			r = b;
			phi = QUARTER_PI_F * (2 - a / b);
		}
	}
	else
	{
		if (a < b)
		{	
			// Reg.3
			r = -a;
			phi = QUARTER_PI_F * (4 + b / a);
		}
		else
		{			
			// Reg.4
			r = -b;

			if (b != 0)
				phi = QUARTER_PI_F * (6 - a / b);
			else
				phi = 0;
		}
	}

	u = r * cos(phi);
	v = r * sin(phi);
}

/**
	@brief P. Shirley's concentric disk algorithm, maps square to disk
	@param[in] N Normal
	@param[in] U Random input
*/
HOD inline Vec3f ShirleyDisk(const Vec3f& N, const Vec2f& U)
{
	float u, v;
	float phi = 0, r = 0, a = 2 * U.x - 1, b = 2 * U.y - 1;
	
	if (a >- b)
	{
		if (a > b)
		{	
			// Reg.1
			r = a;
			phi = QUARTER_PI_F * (b / a);
		}
		else
		{			
			// Reg.2
			r = b;
			phi = QUARTER_PI_F * (2 - a / b);
		}
	}
	else
	{
		if (a < b)
		{	
			// Reg.3
			r = -a;
			phi = QUARTER_PI_F * (4 + b / a);
		}
		else
		{			
			// Reg.4
			r = -b;

			if (b != 0)
				phi = QUARTER_PI_F * (6 - a / b);
			else
				phi = 0;
		}
	}

	u = r * cos(phi);
	v = r * sin(phi);

	Vec3f Ucs, Vcs;

	CreateCS(N, Ucs, Vcs);

	return (u * Ucs) + (v * Vcs);
}

DEV inline float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
{
	float f = nf * fPdf, g = ng * gPdf;
	return (f * f) / (f * f + g * g); 
}