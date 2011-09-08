
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

DEV float Density(CScene* pDevScene, const Vec3f& P)
{
	return (float)(SHRT_MAX * tex3D(gTexDensity, P.x / pDevScene->m_BoundingBox.LengthX(), P.y / pDevScene->m_BoundingBox.LengthY(), P.z /  pDevScene->m_BoundingBox.LengthZ()));
}

DEV CColorRgbHdr GetOpacity(CScene* pDevScene, const float& D)
{
	return pDevScene->m_TransferFunctions.m_Opacity.F(D).r * 100000000.0f;
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
		const float D = Density(pDevScene, SP);
		
		// We ignore air density
		if (D == 0)
		{
			// Increase extent
			NearT += StepSize;
			continue;
		}

		// Get shadow opacity
		const float		Opacity = GetOpacity(pDevScene, D).r;
		const CColorXyz	Color	= GetDiffuse(pDevScene, D).r;

		if (Opacity > 0.0f)
		{
			// Compute chromatic attenuation
// 			Lt.c[0] *= expf(-(Opacity * (1.0f - Color.c[0]) * StepSize));
// 			Lt.c[1] *= expf(-(Opacity * (1.0f - Color.c[1]) * StepSize));
// 			Lt.c[2] *= expf(-(Opacity * (1.0f - Color.c[2]) * StepSize));

			Lt.c[0] *= expf(-(Opacity * StepSize));
			Lt.c[1] *= expf(-(Opacity * StepSize));
			Lt.c[2] *= expf(-(Opacity * StepSize));

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
	// Accumulated radiance
	CColorXyz Ld = SPEC_BLACK;
	
	// Radiance from light source
	CColorXyz Li = SPEC_BLACK;

	// Attenuation
	CColorXyz Tr = SPEC_BLACK;

	float D = Density(pDevScene, Pe);

	CBSDF Bsdf(N, Wo, GetDiffuse(pDevScene, D).ToXYZ(), GetSpecular(pDevScene, D).ToXYZ(), 500.0f, 0.1f);//pDevScene->m_TransferFunctions.m_Roughness.F(D).r);

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
	
	// Sample the light with MIS
	if (!Li.IsBlack() && LightPdf > 0.0f && BsdfPdf > 0.0f)
	{
		// Compute tau
		Tr = Transmittance(pDevScene, R.m_O, R.m_D, Length(R.m_O - Pe), StepSize, Rnd);
		
		// Attenuation due to volume
		Li *= Tr;

		// Compute MIS weight
		const float Weight = 1.0f;//PowerHeuristic(1.0f, LightPdf, 1.0f, BsdfPdf);
 
		// Add contribution
		Ld += Li * (Weight / LightPdf);
	}
	
	// Compute tau

	/**/	
	// Attenuation due to volume
	

//	Ld = Li * Transmittance(pDevScene, R.m_O, R.m_D, Length(R.m_O - Pe), StepSize, Rnd);

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

// Computes the local gradient
DEV Vec3f ComputeGradient(CScene* pDevScene, const Vec3f& P)
{
	Vec3f Normal;

	const float Delta = pDevScene->m_Spacing.Min();

	Vec3f X(Delta, 0.0f, 0.0f), Y(0.0f, Delta, 0.0f), Z(0.0f, 0.0f, Delta);

	Normal.x = 0.5f * (float)(Density(pDevScene, P + X) - Density(pDevScene, P - X));
	Normal.y = 0.5f * (float)(Density(pDevScene, P + Y) - Density(pDevScene, P - Y));
	Normal.z = 0.5f * (float)(Density(pDevScene, P + Z) - Density(pDevScene, P - Z));

	return -Normal;
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

	Re.m_MinT	= MinT;
	Re.m_MaxT	= MaxT;

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
		const float D = Density(pDevScene, Pe);

		// Get opacity at eye point
		const float Tr = GetOpacity(pDevScene, D).r;
//		const CColorXyz	Ke = pDevScene->m_Volume.Ke(D);
		
		// Add emission
//		Lv += Ltr * GetDiffuse(pDevScene, D).ToXYZ();

		// Compute outgoing direction
		Wo = Normalize(-Re.m_D);

		// Obtain normal
		Gn = ComputeGradient(pDevScene, Pe);

		// Exit if air, or not within hemisphere
		if (Tr < 0.01f)
			continue;

		// Estimate direct light at eye point
	 	Lv += Ltr * UniformSampleOneLight(pDevScene, Wo, Pe, Gn, RNG, Ss);

		// Compute eye transmittance
		Ltr *= expf(-(Tr * Ss));

		// Exit if eye transmittance is very small
// 		if (EyeTr.y() < gScene.m_Volume.m_TauThreshold)
// 			break;

		if (Ltr.y() < 0.1f)
			break;
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