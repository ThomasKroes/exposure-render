
#include "VolumeTracer.cuh"

#include "Filter.h"
#include "Scene.h"
#include "Material.h"

texture<short, 3, cudaReadModeNormalizedFloat>	gTexDensity;
texture<short, 3, cudaReadModeNormalizedFloat>	gTexExtinction;

void BindDensityVolume(short* pDensityBuffer, cudaExtent Size)
{
	cudaArray* gpDensity = NULL;

	// create 3D array
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<short>();
	cudaMalloc3DArray(&gpDensity, &ChannelDesc, Size);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams	= {0};
	copyParams.srcPtr				= make_cudaPitchedPtr(pDensityBuffer, Size.width * sizeof(short), Size.width, Size.height);
	copyParams.dstArray				= gpDensity;
	copyParams.extent				= Size;
	copyParams.kind					= cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Set texture parameters
	gTexDensity.normalized		= true;
	gTexDensity.filterMode		= cudaFilterModeLinear;      
	gTexDensity.addressMode[0]	= cudaAddressModeClamp;  
	gTexDensity.addressMode[1]	= cudaAddressModeClamp;
 	gTexDensity.addressMode[2]	= cudaAddressModeClamp;

	// Bind array to 3D texture
	cudaBindTextureToArray(gTexDensity, gpDensity, ChannelDesc);
}

void BindExtinctionVolume(short* pExtinctionBuffer, cudaExtent Size)
{
	cudaArray* gpExtinction = NULL;

	// create 3D array
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<short>();
	cudaMalloc3DArray(&gpExtinction, &ChannelDesc, Size);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams	= {0};
	copyParams.srcPtr				= make_cudaPitchedPtr(pExtinctionBuffer, Size.width * sizeof(short), Size.width, Size.height);
	copyParams.dstArray				= gpExtinction;
	copyParams.extent				= Size;
	copyParams.kind					= cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Set texture parameters
	gTexExtinction.normalized		= true;
	gTexExtinction.filterMode		= cudaFilterModePoint;      
	gTexExtinction.addressMode[0]	= cudaAddressModeClamp;  
	gTexExtinction.addressMode[1]	= cudaAddressModeClamp;
// 	gTexExtinction.addressMode[2]	= cudaAddressModeClamp;

	// Bind array to 3D texture
	cudaBindTextureToArray(gTexExtinction, gpExtinction, ChannelDesc);
}

DEV float Density(CScene* pDevScene, const Vec3f& P)
{
	return ((float)(SHRT_MAX) * tex3D(gTexDensity, P.x / pDevScene->m_BoundingBox.LengthX(), P.y / pDevScene->m_BoundingBox.LengthY(), P.z /  pDevScene->m_BoundingBox.LengthZ()));
}

DEV float Extinction(CScene* pDevScene, const Vec3f& P)
{
	return tex3D(gTexExtinction, P.x / pDevScene->m_BoundingBox.LengthX(), P.y / pDevScene->m_BoundingBox.LengthY(), P.z /  pDevScene->m_BoundingBox.LengthZ());
}

DEV CColorRgbHdr GetOpacity(CScene* pDevScene, const float& D)
{
	return pDevScene->m_DensityScale * pDevScene->m_TransferFunctions.m_Opacity.F(D);
}

DEV CColorRgbHdr GetDiffuse(CScene* pDevScene, const float& D)
{
	return pDevScene->m_TransferFunctions.m_Diffuse.F(D);
}

DEV CColorRgbHdr GetSpecular(CScene* pDevScene, const float& D)
{
	return pDevScene->m_TransferFunctions.m_Specular.F(D);
}

DEV CColorRgbHdr GetEmission(CScene* pDevScene, const float& D)
{
	return pDevScene->m_TransferFunctions.m_Emission.F(D);
}

DEV CColorRgbHdr GetRoughness(CScene* pDevScene, const float& D)
{
	return pDevScene->m_TransferFunctions.m_Roughness.F(D);
}

// Determines whether ray is blocked by lights
DEV bool IntersectP(CScene* pDevScene, const Vec3f& P, const Vec3f& W, const float& MinT, const float& MaxT)
{
	/*
	// Ray for intersection
	CRay R(P, W, MinT, MaxT);

	// Hit distance
	float T = 0.0f; 
	 
	for (int i = 0; i < pScene->m_NoLights; i++) 
	{
		if (pScene->m_Lights[i].Intersect(P, W, MinT, MaxT, T) && T >= MinT && T <= MaxT)
			return true;
	}
	*/

	return false;
}

// ToDo: Add description
DEV bool NearestLight(CScene* pDevScene, CRay& R, CColorXyz& LightColor)
{
	// Whether a hit with a light was found or not 
	bool Hit = false;
	
	float T = 0.0f;

	CRay RayCopy = R;

	for (int i = 0; i < pDevScene->m_Lighting.m_NoLights; i++)
	{
		if (pDevScene->m_Lighting.m_Lights[i].Intersect(RayCopy, T, LightColor))
			Hit = true;
	}
	
	return Hit;
}

// Exitant radiance from nearest light source in scene
DEV CColorXyz Le(CScene* pDevScene, const Vec3f& P, const Vec3f& N, const Vec3f& W, const float& MinT, const float& MaxT)
{
	/*
	// Ray for intersection
	CRay R(P, W, MinT, MaxT);
	 
	// Hit distance
	float HitT = 0.0, T = INF_MAX;

	// Direct light from lights
	CColorXyz Ld = SPEC_BLACK;

	for (int i = 0; i < pScene->m_NoLights; i++)
	{
		Vec2f UV(0.0f); 

		if (pScene->m_Lights[i].Intersect(P, W, MinT, MaxT, HitT, NULL, &UV) && HitT > MinT && HitT <= T)
		{
			Ld	= pScene->m_Lights[i].Le(UV, pMaterials, pTextures, pBitmaps);
			T	= HitT; 
		}
	}
	
	return Ld;
	*/

	return SPEC_BLACK;
}

// Computes the power heuristic
DEV inline float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
{
	float f = nf * fPdf, g = ng * gPdf;
	return (f * f) / (f * f + g * g); 
}

// Find the nearest non-empty voxel in the volume
DEV inline bool NearestIntersection(CScene* pDevScene, CRay& R, const float& StepSize, const float& U, float* pBoxMinT = NULL, float* pBoxMaxT = NULL)
{
	float MinT;
	float MaxT;

	// Intersect the eye ray with bounding box, if it does not intersect then return the environment
	if (!pDevScene->m_BoundingBox.Intersect(R, &MinT, &MaxT))
		return false;

	bool Hit = false;

	if (pBoxMinT)
		*pBoxMinT = MinT;

	if (pBoxMaxT)
		*pBoxMaxT = MaxT;

	MinT += U * StepSize;

	// Step through the volume and stop as soon as we come across a non-empty voxel
	while (MinT < MaxT)
	{
		if (GetOpacity(pDevScene, Density(pDevScene, R(MinT))).r > 0.0f)
		{
			Hit = true;
			break;
		}
		else
		{
			MinT += StepSize;
		}
	}

	if (Hit)
	{
		R.m_MinT = MinT;
		R.m_MaxT = MaxT;
	}

	return Hit;
}

// Computes the local gradient
DEV Vec3f ComputeGradient(CScene* pDevScene, const Vec3f& P, const Vec3f& D)
{
	Vec3f Normal;

	Normal.x = Density(pDevScene, P + 1 * Vec3f(1.0f, 0.0f, 0.0f)) - Density(pDevScene, P - 1 * Vec3f(1.0f, 0.0f, 0.0f));
	Normal.y = Density(pDevScene, P + 1 * Vec3f(0.0f, 1.0f, 0.0f)) - Density(pDevScene, P - 1 * Vec3f(0.0f, 1.0f, 0.0f));
	Normal.z = Density(pDevScene, P + 1 * Vec3f(0.0f, 0.0f, 1.0f)) - Density(pDevScene, P - 1 * Vec3f(0.0f, 0.0f, 1.0f));

	Normal.Normalize();

	return -Normal;
}

// Computes the attenuation through the volume
DEV inline CColorXyz Transmittance(CScene* pDevScene, const Vec3f& P, const Vec3f& D, const float& MaxT, const float& StepSize, CCudaRNG& Rnd)
{
	// Near and far intersections with volume axis aligned bounding box
	float NearT = 0.0f, FarT = 0.0f;

	// Intersect with volume axis aligned bounding box
	if (!pDevScene->m_BoundingBox.Intersect(CRay(P, D, 0.0f, FLT_MAX), &NearT, &FarT))
		return SPEC_BLACK;

	// Clamp to near plane if necessary
	if (NearT < 0.0f) 
		NearT = 0.0f;     

	CColorXyz Lt = SPEC_WHITE;

	NearT += Rnd.Get1() * StepSize;

	// Accumulate
	while (NearT < MaxT)
	{
		// Determine sample point
		const Vec3f SP = P + D * (NearT);

		// Fetch density
		const float D = Density(pDevScene, SP);
		
		// We ignore air density
		if (D == 0)
		{
			// Increase extent
			NearT += StepSize;
			continue;
		}

		// Get shadow opacity
		const float	Opacity = GetOpacity(pDevScene, D).r;

		if (Opacity > 0.0f)
		{
			// Compute eye transmittance
			Lt *= expf(-(Opacity * StepSize));

			// Exit if eye transmittance is very small
			if (Lt.y() < 0.05f)
				break;
		}

		// Increase extent
		NearT += StepSize;
	}

	return Lt;
}

// Estimates direct lighting
DEV CColorXyz EstimateDirectLight(CScene* pDevScene, CLight& Light, CLightingSample& LS, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CCudaRNG& Rnd, const float& StepSize)
{
// 	if (Dot(Wo, N) < 0.0f)
// 		return SPEC_BLACK;

	// Accumulated radiance
	CColorXyz Ld = SPEC_BLACK;
	
	// Radiance from light source
	CColorXyz Li = SPEC_BLACK;

	// Attenuation
	CColorXyz Tr = SPEC_BLACK;

	const float D = Density(pDevScene, Pe);

	CBSDF Bsdf(N, Wo, GetDiffuse(pDevScene, D).ToXYZ(), GetSpecular(pDevScene, D).ToXYZ(), 50.0f, GetRoughness(pDevScene, D).r);
	// Light/shadow ray
	CRay R; 

	// Light probability
	float LightPdf = 1.0f, BsdfPdf = 1.0f;
	
	// Incident light direction
	Vec3f Wi;

	CColorXyz F = SPEC_BLACK;
	
	CSurfacePoint SPe, SPl;

	SPe.m_P		= Pe;
	SPe.m_Ng	= N; 

	// Sample the light source
 	Li = Light.SampleL(Pe, R, LightPdf, LS);
	
	Wi = -R.m_D; 

	F = Bsdf.F(Wo, Wi); 

	BsdfPdf	= Bsdf.Pdf(Wo, Wi);
	
	// Sample the light with MIS
	if (!Li.IsBlack() && LightPdf > 0.0f && BsdfPdf > 0.0f)
	{
		// Compute tau
		const CColorXyz Tr = Transmittance(pDevScene, R.m_O, R.m_D, Length(R.m_O - Pe), StepSize, Rnd);
		
		// Attenuation due to volume
		Li *= Tr;

		// Compute MIS weight
		const float Weight = PowerHeuristic(1.0f, LightPdf, 1.0f, BsdfPdf);
 
		// Add contribution
		Ld += F * Li * (AbsDot(Wi, N) * Weight / LightPdf);
	}

	return Ld;

	/*
	// Sample the BRDF with MIS
	F = Bsdf.SampleF(Wo, Wi, BsdfPdf, LS.m_BsdfSample);
	
	CLight* pNearestLight = NULL;

	Vec2f UV;
	
	if (!F.IsBlack())
	{
		float MaxT = 1000000000.0f; 

		// Compute virtual light point
		const Vec3f Pl = Pe + (MaxT * Wi);

		if (NearestLight(pDevScene, Pe, Wi, 0.0f, MaxT, pNearestLight, NULL, &UV, &LightPdf))
		{
			if (LightPdf > 0.0f && BsdfPdf > 0.0f) 
			{
				// Add light contribution from BSDF sampling
				const float Weight = PowerHeuristic(1.0f, BsdfPdf, 1.0f, LightPdf);
				 
				// Get exitant radiance from light source
// 				Li = pNearestLight->Le(UV, pScene->m_Materials, pScene->m_Textures, pScene->m_Bitmaps);

				if (!Li.IsBlack())
				{
					// Scale incident radiance by attenuation through volume
					Tr = Transmittance(pDevScene, Pe, Wi, 1.0f, StepSize, Rnd);

					// Attenuation due to volume
					Li *= Tr;

					// Contribute
					Ld += F * Li * AbsDot(Wi, N) * Weight / BsdfPdf;
				}
			}
		}
	}
	*/

	return Ld;
}

// Uniformly samples one light
DEV CColorXyz UniformSampleOneLight(CScene* pDevScene, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CCudaRNG& Rnd, const float& StepSize)
{
	// Determine no. lights
	const int NumLights = pDevScene->m_Lighting.m_NoLights;

	// Exit return zero radiance if no light
 	if (NumLights == 0)
 		return SPEC_BLACK;

	CLightingSample LS;

	// Create light sampler
	LS.LargeStep(Rnd);

	// Choose which light to sample
	const int WhichLight = (int)floorf(LS.m_LightNum * (float)NumLights);

	// Get the light
	CLight& Light = pDevScene->m_Lighting.m_Lights[WhichLight];

	// Return estimated direct light
	return (float)NumLights * EstimateDirectLight(pDevScene, Light, LS, Wo, Pe, N, Rnd, StepSize);
}

// Trace volume with single scattering
KERNEL void KrnlSS(CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates, CColorXyz* pDevEstFrameXyz)
{
	const int X = (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global y
	const int Y	= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global x
	
	// Compute sample ID
	const int SID = (Y * (gridDim.x * blockDim.x)) + X;

	float StepSize = 0.02f;

	// Exit if beyond kernel boundaries
	if (X >= pDevScene->m_Camera.m_Film.m_Resolution.GetResX() || Y >= pDevScene->m_Camera.m_Film.m_Resolution.GetResY())
		return;
	
	// Init random number generator
	CCudaRNG RNG(&pDevRandomStates[SID]);

	// Transmittance
	CColorXyz 	EyeTr	= SPEC_WHITE;		// Eye transmittance
	CColorXyz	L		= SPEC_BLACK;		// Measured volume radiance

	// Continue
	bool Continue = true;

	CRay EyeRay, RayCopy;


	float BoxMinT = 0.0f, BoxMaxT = 0.0f;


 	// Generate the camera ray
 	pDevScene->m_Camera.GenerateRay(Vec2f(X, Y), RNG.Get2(), EyeRay.m_O, EyeRay.m_D);

	EyeRay.m_MinT = 0.0f; 
	EyeRay.m_MaxT = FLT_MAX;

	// Check if ray passes through volume, if it doesn't, evaluate scene lights and stop tracing 
 	if (!NearestIntersection(pDevScene, EyeRay, StepSize, RNG.Get1(), &BoxMinT, &BoxMaxT))
 		Continue = false;

	CColorXyz Li = SPEC_BLACK;
	RayCopy = CRay(EyeRay.m_O, EyeRay.m_D, 0.0f, BoxMinT);

	if (NearestLight(pDevScene, RayCopy, Li))
	{
		pDevEstFrameXyz[Y * (int)pDevScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = Li;
		return;
	}
/*

	// Check if there is a light between the observer and the volume, if there is, evaluate it, contribute to image and stop tracing
	if (NearestLight(pDevScene, EyeRay.m_O, EyeRay.m_D, 0.0f, MinT, pNearestLight, &Front, &UV, NULL))
	{
// 		L = Front ? EyeTr * pNearestLight->Le(UV, pDevScene->m_Materials, pDevScene->m_Textures, pDevScene->m_Bitmaps) : SPEC_BLACK;

		// Stop
		Continue = false;
	}
	*/

// 	if (EyeRay.m_MaxT == INF_MAX)
//  		Continue = false;
	
	float EyeT	= EyeRay.m_MinT;

	Vec3f EyeP, Normal;
	
	// Walk along the eye ray with ray marching
	while (Continue && EyeT < EyeRay.m_MaxT)
	{
		// Determine new point on eye ray
		EyeP = EyeRay(EyeT);

		// Increase parametric range
		EyeT += StepSize;

		// Fetch density
		const float D = Density(pDevScene, EyeP);

		// We ignore air density
// 		if (Density == 0) 
// 			continue;
		 
		// Get opacity at eye point
		const float		Tr = GetOpacity(pDevScene, D).r;
		const CColorXyz	Ke = GetEmission(pDevScene, D).ToXYZ();
		
		// Add emission
		EyeTr += Ke; 
		
		// Compute outgoing direction
		const Vec3f Wo = Normalize(-EyeRay.m_D);

		// Obtain normal
		Normal = ComputeGradient(pDevScene, EyeP, Wo);

		// Exit if air, or not within hemisphere
		if (Tr < 0.05f)// || Dot(Wo, Normal[TID]) < 0.0f)
			continue;

		// Estimate direct light at eye point
	 	L += EyeTr * UniformSampleOneLight(pDevScene, Wo, EyeP, Normal, RNG, StepSize);

		// Compute eye transmittance
		EyeTr *= expf(-(Tr * StepSize));

		/*
		// Russian roulette
		if (EyeTr.y() < 0.5f)
		{
			const float DieP = 1.0f - (EyeTr.y() / Threshold);

			if (DieP > RNG.Get1())
			{
				break;
			}
			else
			{
				EyeTr *= 1.0f / (1.0f - DieP);
			}
		}
		*/

		if (EyeTr.y() < 0.05f)
			break;
	}

	RayCopy = CRay(EyeRay(BoxMaxT), EyeRay.m_D, 0.0f);

// 	if (NearestLight(pDevScene, RayCopy, Li))
// 		pDevEstFrameXyz[Y * (int)pDevScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = Li;
// 		return;
//		L += Li;

	// Contribute
	pDevEstFrameXyz[Y * (int)pDevScene->m_Camera.m_Film.m_Resolution.GetResX() + X] = L;
}

// Traces the volume
void RenderVolume(CScene* pScene, CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates, CColorXyz* pDevEstFrameXyz)
{
	const dim3 KernelBlock(32, 8);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	
	// Execute kernel
	KrnlSS<<<KernelGrid, KernelBlock>>>(pDevScene, pDevRandomStates, pDevEstFrameXyz);
}
