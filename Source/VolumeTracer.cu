
#include "VolumeTracer.cuh"

#include "Filter.h"
#include "Scene.h"
#include "Material.h"

texture<short, 3, cudaReadModeNormalizedFloat>	gTexDensity;

cudaArray* gpI = NULL;

void BindVolumeData(short* pDensity, CResolution3D& Resolution)
{
	cudaExtent ExtentGridI;

	ExtentGridI.width	= Resolution.m_XYZ.x;
	ExtentGridI.depth	= Resolution.m_XYZ.z;
	ExtentGridI.height	= Resolution.m_XYZ.y; 

	// create 3D array
	cudaChannelFormatDesc ChannelDescDensity = cudaCreateChannelDesc<short>();
	cudaMalloc3DArray(&gpI, &ChannelDescDensity, ExtentGridI);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(pDensity, ExtentGridI.width * sizeof(short), ExtentGridI.width, ExtentGridI.height);
	copyParams.dstArray = gpI;
	copyParams.extent   = ExtentGridI;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Set texture parameters
	gTexDensity.normalized		= true;
	gTexDensity.filterMode		= cudaFilterModeLinear;      
	gTexDensity.addressMode[0]	= cudaAddressModeClamp;  
	gTexDensity.addressMode[1]	= cudaAddressModeClamp;
	gTexDensity.addressMode[2]	= cudaAddressModeClamp;

	// Bind array to 3D texture
	cudaBindTextureToArray(gTexDensity, gpI, ChannelDescDensity);
}

void UnbindVolumeData(void)
{
	cudaFree(gpI);
}

DEV float Density(const Vec3f& P)
{
	return (float)(SHRT_MAX * tex3D(gTexDensity, P.x, P.y, P.z));
}

// Computes the attenuation through the volume
DEV CColorXyz Transmittance(CScene* pDevScene, const Vec3f& P, const Vec3f& D, const float& MaxT, const float& StepSize, CCudaRNG& RNG)
{
	// Near and far intersections with volume axis aligned bounding box
	float NearT = 0.0f, FarT = FLT_MAX;

	if (!pDevScene->m_BoundingBox.Intersect(CRay(P, D, 0.0f, MaxT), &NearT, &FarT))
		return SPEC_WHITE;

	CColorXyz Lt = SPEC_WHITE;

	NearT += RNG.Get1() * StepSize;

	// Accumulate
	while (NearT < MaxT)
	{
		// Determine sample point
		const Vec3f SP = P + D * (NearT);

		// Fetch density
		const float D = Density(SP);
		
		// We ignore air density
		if (D == 0)
		{
			// Increase extent
			NearT += StepSize;
			continue;
		}

		// Get shadow opacity
		const float		Opacity = pDevScene->m_TransferFunctions.m_Opacity.F(D).r * 100.0f;
		const CColorXyz	Color	= pDevScene->m_TransferFunctions.m_DiffuseColor.F(D).ToXYZ();

		if (Opacity > 0.0f)
		{
			// Compute chromatic attenuation
			Lt.c[0] *= expf(-(Opacity * (1.0f - Color.c[0]) * StepSize));
			Lt.c[1] *= expf(-(Opacity * (1.0f - Color.c[1]) * StepSize));
			Lt.c[2] *= expf(-(Opacity * (1.0f - Color.c[2]) * StepSize));

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
	
	if (Dot(Wo, N) < 0.0f)
		return SPEC_BLACK;

	// Accumulated radiance
	CColorXyz Ld = SPEC_BLACK;
	
	// Radiance from light source
	CColorXyz Li = SPEC_BLACK;

	// Attenuation
	CColorXyz Tr = SPEC_BLACK;

	float D = Density(Pe);

	CBSDF Bsdf(N, Wo, pDevScene->m_TransferFunctions.m_DiffuseColor.F(D).ToXYZ(), pDevScene->m_TransferFunctions.m_SpecularColor.F(D).ToXYZ(), 1.0f, 1.0f);
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
 	Li = Light.SampleL(SPe, SPl, LS, LightPdf, 0.1f);
	
	R.m_O		= SPl.m_P;
	R.m_D		= Normalize(SPe.m_P - SPl.m_P);
	R.m_MinT	= 0.0f;
	R.m_MaxT	= (SPl.m_P - SPe.m_P).Length();
	
	Wi = -R.m_D; 

	F = Bsdf.F(Wo, Wi); 

	BsdfPdf	= Bsdf.Pdf(Wo, Wi);
//	BsdfPdf = Dot(Wi, N);

	/*
	// Sample the light with MIS
	if (!Li.IsBlack() && LightPdf > 0.0f && BsdfPdf > 0.0f)
	{
		// Compute tau
		const CColorXyz Tr = Transmittance(pDevScene, R.m_O, R.m_D, Length(R.m_O - Pe), StepSize, Rnd);
		
		// Attenuation due to volume
		Li *= Tr;

		// Compute MIS weight
		const float Weight = 1.0f;//PowerHeuristic(1.0f, LightPdf, 1.0f, BsdfPdf);
 
		// Add contribution
		Ld += F * Li * (AbsDot(Wi, N) * Weight / LightPdf);
	}
	
	// Compute tau
	const CColorXyz Tr = ;
	*/	
	// Attenuation due to volume
	

	Ld = Li * Transmittance(pDevScene, R.m_O, R.m_D, Length(R.m_O - Pe), StepSize, Rnd);

	/**/

	/*
	// Sample the BRDF with MIS
	F = Bsdf.SampleF(Wo, Wi, BsdfPdf, LS.m_BsdfSample);
	
//	Wi = CosineWeightedHemisphere(Rnd.Get2(), N);

//	BsdfPdf = Dot(Wi, N);

	CLight* pNearestLight = NULL;

	Vec2f UV;

	if (!F.IsBlack())
	{
		float MaxT = INF_MAX;

		// Compute virtual light point
		const Vec3f Pl = Pe + (MaxT * Wi);

		if (NearestLight(pScene, Pe, Wi, 0.0f, MaxT, pNearestLight, NULL, &UV, &LightPdf))
		{
			if (LightPdf > 0.0f && BsdfPdf > 0.0f) 
			{
				// Add light contribution from BSDF sampling
				const float Weight = PowerHeuristic(1.0f, BsdfPdf, 1.0f, LightPdf);
				 
				// Get exitant radiance from light source
				Li = pNearestLight->Le(UV, pScene->m_Materials, pScene->m_Textures, pScene->m_Bitmaps);

				if (!Li.IsBlack())
				{
					// Scale incident radiance by attenuation through volume
					Tr = Transmittance(pScene, Pe, Wi, 1.0f, StepSize, Rnd);

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
 	if (pDevScene->m_Lighting.m_NoLights == 0)
 		return SPEC_BLACK;

	CLightingSample LS;

	// Create light sampler
	LS.LargeStep(Rnd);

	// Choose which light to sample
	const int WhichLight = (int)floorf(LS.m_LightNum * (float)pDevScene->m_Lighting.m_NoLights);

	// Get the light
	CLight& Light = pDevScene->m_Lighting.m_Lights[WhichLight];

	// Return estimated direct light
	return (float)pDevScene->m_Lighting.m_NoLights * EstimateDirectLight(pDevScene, Light, LS, Wo, Pe, N, Rnd, StepSize);
}

HOD float PhaseHG(const Vec3f& W, const Vec3f& Wp, float G)
{
	float CosTheta = Dot(W, Wp);
	return 1.0f / (4.0f * PI_F) * (1.0f - G * G) / powf(1.0f + G * G - 2.0f * G * CosTheta, 1.5f);
}

HOD Vec3f SampleHG(const Vec3f& W, float G, const Vec2f& U)
{
	float CosTheta;

	if (fabsf(G) < 1e-3)
	{
		CosTheta = 1.0f - 2.0f * U.x;
	}
	else
	{
		float SqrtTerm = (1.0f - G * G) / (1.0f - G + 2.0f * G * U.x);
		CosTheta = (1.0f + G * G - SqrtTerm * SqrtTerm) / (2.0f * G);
	}

	float SinTheta = sqrtf(max(0.f, 1.f - CosTheta * CosTheta));
	float Phi = 2.f * PI_F * U.y;
	Vec3f V1, V2;
	CoordinateSystem(W, &V1, &V2);
	return SphericalDirection(SinTheta, CosTheta, Phi, V1, V2, W);
}

HOD float PdfHG(const Vec3f& W, const Vec3f& Wp, float G)
{
	return PhaseHG(W, Wp, G);
}

// Fetches the density from the texture
DEV inline float LookupDensity(const Vec3f& P)
{
	return (float)(SHRT_MAX * tex3D(gTexDensity, P.x, P.y, P.z));
}

// Computes the local gradient
DEV Vec3f ComputeGradient(const Vec3f& P)
{
	Vec3f Normal;

	Vec3f X(1.0f, 0.0f, 0.0f), Y(0.0f, 1.0f, 0.0f), Z(0.0f, 0.0f, 1.0f);

	Normal.x = 0.5f * (float)(Density(P + X) - Density(P - X));
	Normal.y = 0.5f * (float)(Density(P + Y) - Density(P - Y));
	Normal.z = 0.5f * (float)(Density(P + Z) - Density(P - Z));

	return -Normal;
}

DEV inline bool SampleDistanceRM(CRay& R, CCudaRNG& RNG, CVolumePoint& VP, CScene* pDevScene, int Component)
{
	float MinT = 0.0f, MaxT = 0.0f;

	if (!pDevScene->m_BoundingBox.Intersect(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
	MaxT = min(MaxT, R.m_MaxT);

	float S = -log(RNG.Get1()) / pDevScene->m_MaxD, Dt = 1.0f * (1.0f / (float)pDevScene->m_Resolution.m_XYZ.Max()), Sum = 0.0f, SigmaT = 0.0f, D = 0.0f;

	Vec3f samplePos; 

	MinT += RNG.Get1() * Dt;

	while (Sum < S)
	{
		samplePos = R.m_O + MinT * R.m_D;

		if (MinT > MaxT)
			return false;
		
		D = (float)(SHRT_MAX * tex3D(gTexDensity, pDevScene->m_BoundingBox.m_MinP.x + (samplePos.x / pDevScene->m_BoundingBox.m_MaxP.x), pDevScene->m_BoundingBox.m_MinP.y + (samplePos.y / pDevScene->m_BoundingBox.m_MaxP.y), pDevScene->m_BoundingBox.m_MinP.z + (samplePos.z / pDevScene->m_BoundingBox.m_MaxP.z)));

		SigmaT	= 10.0f * pDevScene->m_TransferFunctions.m_Opacity.F(D)[Component] * pDevScene->m_TransferFunctions.m_DiffuseColor.F(D)[Component];
		Sum		+= SigmaT * Dt;
		MinT	+= Dt;
	}

	VP.m_Transmittance.c[Component]	= 0.5f;
	VP.m_P							= samplePos;
	VP.m_D							= D;

	return true;
}

DEV inline bool FreePathRM(CRay& R, CCudaRNG& RNG, CVolumePoint& VP, CScene* pDevScene, int Component)
{
	float MinT = 0.0f, MaxT = 0.0f;

	if (!pDevScene->m_BoundingBox.Intersect(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
//	MaxT = min(MaxT, R.m_MaxT);

	float S = -log(RNG.Get1()) / pDevScene->m_MaxD, Dt = 1.0f * (1.0f / (float)pDevScene->m_Resolution.m_XYZ.Max()), Sum = 0.0f, SigmaT = 0.0f, D = 0.0f;

	Vec3f samplePos; 

	MinT += RNG.Get1() * Dt;

	while (Sum < S)
	{
		samplePos = R.m_O + MinT * R.m_D;

		// Free path, no collisions in between
		if (MinT > R.m_MaxT)
			break;
		
		D = (float)(SHRT_MAX * tex3D(gTexDensity, pDevScene->m_BoundingBox.m_MinP.x + (samplePos.x / pDevScene->m_BoundingBox.m_MaxP.x), pDevScene->m_BoundingBox.m_MinP.y + (samplePos.y / pDevScene->m_BoundingBox.m_MaxP.y), pDevScene->m_BoundingBox.m_MinP.z + (samplePos.z / pDevScene->m_BoundingBox.m_MaxP.z)));

		SigmaT	= 10.0f * pDevScene->m_TransferFunctions.m_Opacity.F(D)[Component] * pDevScene->m_TransferFunctions.m_DiffuseColor.F(D)[Component];
		Sum		+= SigmaT * Dt;
		MinT	+= Dt;
	}

	if (MinT < R.m_MaxT)
		return false;

	VP.m_Transmittance.c[Component]	= 0.5f;
	VP.m_P							= samplePos;
	VP.m_D							= D;

	return true;
}

// Trace volume with single scattering
KERNEL void KrnlSS(CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates, CColorXyz* pDevEstFrameXyz)
{
	const int X = (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global y
	const int Y	= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global x

	// Compute sample ID
	const int SID = (Y * (gridDim.x * blockDim.x)) + X;

	// Exit if beyond kernel boundaries
	if (X >= pDevScene->m_Camera.m_Film.m_Resolution.Width() || Y >= pDevScene->m_Camera.m_Film.m_Resolution.Height())
		return;

	// Init random number generator
	CCudaRNG RNG(&pDevRandomStates[SID]);

	// Eye ray (Re), Light ray (Rl)
	CRay Re, Rl;

	// Generate the camera ray
	pDevScene->m_Camera.GenerateRay(Vec2f(X, Y), RNG.Get2(), Re.m_O, Re.m_D);

	// Distance towards nearest intersection with bounding box (MinT), distance to furthest intersection with bounding box (MaxT)
	float MinT = 0.0f, MaxT = INF_MAX;

	// Early ray termination if ray does not intersect with bounding box
	if (!pDevScene->m_BoundingBox.Intersect(Re, &MinT, &MaxT))
	{
		pDevEstFrameXyz[Y * (int)pDevScene->m_Camera.m_Film.m_Resolution.Width() + X] = SPEC_BLACK;
		return;
	}

	// Eye attenuation (Le), accumulated color through volume (Lv), unattenuated light from light source (Li), attenuated light from light source (Ld), and BSDF value (F)
	CColorXyz Ltr	= SPEC_WHITE;
	CColorXyz Li	= SPEC_BLACK;
	CColorXyz Lv	= SPEC_BLACK;
	CColorXyz F		= SPEC_BLACK;

	int NoScatteringEvents = 0, RussianRouletteDepth = 2; 

	Re.m_MinT	= 0.0f;
	Re.m_MaxT	= RAY_MAX;

	// Continue probability (Pc) Light probability (LightPdf) Bsdf probability (BsdfPdf)
	float Pc = 0.5f, LightPdf = 1.0f, BsdfPdf = 1.0f;

	// Eye point (Pe), light sample point (Pl), Gradient (G), normalized gradient (Gn), reversed eye direction (Wo), incident direction (Wi), new direction (W)
	Vec3f Pe, Pl, G, Gn, Wo, Wi, W;

	// Distance along eye ray (Te), step size (Ss), density (D)
	float Ss = pDevScene->m_Spacing.Min(), Te = MinT + RNG.Get1() * Ss, D = 0.0f;

	// Choose color component to sample
	int CC1 = floorf(RNG.Get1() * 3.0f);

	bool Hit = false;

	// Walk along the eye ray with ray marching
	while (Te < MaxT)
	{
		// Determine new point on eye ray
		Pe = Re(Te);

		// Increase parametric range
		Te += Ss;

		// Fetch density
		const short D = Density(Pe);

		// We ignore air density
		if (D == 0.0f)
			continue;

		// Get opacity at eye point
		const float Tr = pDevScene->m_TransferFunctions.m_Opacity.F(D).r;
//		const CColorXyz	Ke = pDevScene->m_Volume.Ke(D);
		
		// Add emission
//		Ltr += Ke;

		// Compute outgoing direction
		Wo = Normalize(-Re.m_D);

		// Obtain normal
		Gn = ComputeGradient(Pe);

		// Exit if air, or not within hemisphere
		if (Tr < 0.01f)
			continue;

		// Estimate direct light at eye point
	 	Lv += Ltr * UniformSampleOneLight(pDevScene, Wo, Pe, Gn, RNG, Ss);
//		Lv += Ltr * T * SPEC_WHITE;

		// Compute eye transmittance
		Ltr *= expf(-(Tr * Ss));

		// Exit if eye transmittance is very small
// 		if (EyeTr.y() < gScene.m_Volume.m_TauThreshold)
// 			break;

		if (Ltr.y() < 0.1f)
		{
// 			EyeTr = 
			break;
		}
	}

	pDevEstFrameXyz[Y * (int)pDevScene->m_Camera.m_Film.m_Resolution.Width() + X] = Lv;
}

// Traces the volume
void RenderVolume(CScene* pScene, CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates, CColorXyz* pDevEstFrameXyz)
{
	const dim3 KernelBlock(pScene->m_KernelSize.x, pScene->m_KernelSize.y);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.Width() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.Height() / (float)KernelBlock.y));
	
	// Execute kernel
//	KrnlRenderVolume<<<KernelGrid, KernelBlock>>>(pDevScene, pDevRandomStates, pDevEstFrameXyz);
	KrnlSS<<<KernelGrid, KernelBlock>>>(pDevScene, pDevRandomStates, pDevEstFrameXyz);
}