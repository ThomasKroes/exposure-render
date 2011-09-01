
#include "VolumeTracer.cuh"

#include "Filter.h"
#include "Scene.h"

texture<short, 3, cudaReadModeNormalizedFloat>	gTexDensity;

cudaArray* gpI = NULL;

KERNEL void KrnlSetupRNG(CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates)
{
	const int X		= (blockIdx.x * blockDim.x) + threadIdx.x;
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;

	// Exit if beyond canvas boundaries
	if (X >= pDevScene->m_Camera.m_Film.m_Resolution.Width() || Y >= pDevScene->m_Camera.m_Film.m_Resolution.Height())
		return;

	// Initialize
	curand_init(Y * (int)pDevScene->m_Camera.m_Film.m_Resolution.Width() + X, 1234, 0, &pDevRandomStates[Y * (int)pDevScene->m_Camera.m_Film.m_Resolution.Width() + X]);
}

extern "C" void SetupRNG(CScene* pScene, CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates)
{
	const dim3 KernelBlock(pScene->m_KernelSize.x, pScene->m_KernelSize.y);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.Width() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.Height() / (float)KernelBlock.y));

	KrnlSetupRNG<<<KernelGrid, KernelBlock>>>(pDevScene, pDevRandomStates);

	cudaError_t Error = cudaGetLastError();
}

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

	Normal.x = 0.5f * (float)(LookupDensity(P + X) - LookupDensity(P - X));
	Normal.y = 0.5f * (float)(LookupDensity(P + Y) - LookupDensity(P - Y));
	Normal.z = 0.5f * (float)(LookupDensity(P + Z) - LookupDensity(P - Z));

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

		SigmaT	= 10.0f * pDevScene->m_TransferFunctions.m_Kt.F(D)[Component] * pDevScene->m_TransferFunctions.m_Kd.F(D)[Component];
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

		SigmaT	= 10.0f * pDevScene->m_TransferFunctions.m_Kt.F(D)[Component] * pDevScene->m_TransferFunctions.m_Kd.F(D)[Component];
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
KERNEL void KrnlRenderVolume(CScene* pDevScene, curandStateXORWOW_t* pDevRandomStates, CColorXyz* pDevEstFrameXyz)
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

	CRay Re, Rl;
	
	// Generate the camera ray
	pDevScene->m_Camera.GenerateRay(Vec2f(X, Y), RNG.Get2(), Re.m_O, Re.m_D);

	// Eye attenuation (Le), accumulated color through volume (Lv), unattenuated light from light source (Li), attenuated light from light source (Ld), and BSDF value (F)
	CColorXyz PathThroughput	= SPEC_WHITE;
	CColorXyz Li				= SPEC_BLACK;
	CColorXyz Lv				= SPEC_BLACK;
	CColorXyz F					= SPEC_BLACK;

	int NoScatteringEvents = 0, RussianRouletteDepth = 2; 

	Re.m_MinT	= 0.0f;
	Re.m_MaxT	= RAY_MAX;

	// Continue probability (Pc) Light probability (LightPdf) Bsdf probability (BsdfPdf)
	float Pc = 0.5f, LightPdf = 1.0f, BsdfPdf = 1.0f;

	// Eye point (Pe), light sample point (Pl), Gradient (G), normalized gradient (Gn), reversed eye direction (Wo), incident direction (Wi), new direction (W)
	Vec3f Pe, Pl, G, Gn, Wo, Wi, W;

	// Choose color component to sample
	int CC1 = floorf(RNG.Get1() * 3.0f);

	// Walk along the eye ray with ray marching
	while (NoScatteringEvents < pDevScene->m_MaxNoBounces)
	{
		CVolumePoint VP;

		// Sample distance
		if (SampleDistanceRM(Re, RNG, VP, pDevScene, CC1))
		{
// 			if (VP.m_Transmittance.y() > 0.0f)
//			PathThroughput.c[CC1] *= VP.m_Transmittance.c[CC1];

			// Compute gradient (G) and normalized gradient (Gn)
  			G	= ComputeGradient(VP.m_P);
  			Gn	= Normalize(G);
 			Wo	= Normalize(-Re.m_D);

			// Choose random light and compute the amount of light that reaches the scattering point
//			Li = SampleRandomLight(pScene, RNG, Pe, Pl, LightPdf);
//			Li = 1000.0f * CColorXyz(0.9f, 0.6f, 0.2f);
			Li = 500.0f * CColorXyz(1.0f);

			Pe = VP.m_P;



			Pl = pDevScene->m_BoundingBox.GetCenter() + pDevScene->m_Light.m_Distance * Vec3f(sinf(pDevScene->m_Light.m_Theta), sinf(pDevScene->m_Light.m_Phi), cosf(pDevScene->m_Light.m_Theta));
//			Pl = pBoundingBox->GetCenter() + UniformSampleSphere(RNG.Get2()) * Vec3f(1000.0f);

			// LightPdf = powf((Pe - Pl).Length(), 2.0f);

			Rl = CRay(Pl, Normalize(Pe - Pl), 0.0f, (Pe - Pl).Length());

			if (!Li.IsBlack() && LightPdf > 0.0f && FreePathRM(Rl, RNG, VP, pDevScene, CC1))
			{
				Li /= LightPdf;
				Lv.c[CC1] += PathThroughput.c[CC1] * Li.c[CC1] * PhaseHG(Wo, Rl.m_D, pDevScene->m_PhaseG);// * VP.m_Transmittance.c[CC1];// * ;
			}

			W = Normalize(SampleHG(Wo, pDevScene->m_PhaseG, RNG.Get2()));
//			W = UniformSampleSphere(RNG.Get2());
//			W = UniformSampleHemisphere(RNG.Get2(), Gn);

			// Configure eye ray
			Re = CRay(VP.m_P, W, 0.0f, RAY_MAX);

			// Russian roulette
			if (NoScatteringEvents >= RussianRouletteDepth)
			{
				if (RNG.Get1() > Pc)
					break;
				else
					PathThroughput.c[CC1] /= Pc;
			}

//			PathThroughput.c[CC1] /= 4.0f * PI_F;
//			PathThroughput.c[CC1] /= PhaseHG(Wo, Rl.m_D, PhaseG);

			// Add scattering event
			NoScatteringEvents++;
		}
		else
		{
			break;
		}
	}

//  	if (pBoundingBox->Intersect(Re))
//  		Lv += SPEC_WHITE;


	pDevEstFrameXyz[Y * (int)pDevScene->m_Camera.m_Film.m_Resolution.Width() + X].c[CC1] = Lv.c[CC1] / fmaxf(1.0f, NoScatteringEvents);
}


struct CVolumeMaterial
{
	Vec3f		m_N;
	CColorXyz	m_Kd;
	CColorXyz	m_Ks;
	float		m_Ni;
	float		m_Ns;
	
	HOD void Init(const Vec3f& N, const CColorXyz& Kd, const CColorXyz& Ks, const float& Ni, const float& Ns)
	{
		m_N		= N;
		m_Kd	= Kd;
		m_Ks	= Ks;
		m_Ni	= Ni;
		m_Ns	= Ns;
	}

	HOD void BlinnSampleF(const Vec3f& Wo, Vec3f& Wi, Vec2f& U, float& Pdf)
	{
		// Compute sampled half-angle vector $\wh$ for Blinn distribution
		float costheta = powf(U.x, 1.f / (m_Ns + 1));
		float sintheta = sqrtf(max(0.f, 1.f - costheta*costheta));
		float phi = U.y * 2.f * PI_F;
		
		Vec3f wh = SphericalDirection(sintheta, costheta, phi);
		
		if (!SameHemisphere(Wo, wh))
			wh = -wh;

		// Compute incident direction by reflecting about $\wh$
		Wi = -Wo + 2.f * Dot(Wo, wh) * wh;

		// Compute PDF for $\wi$ from Blinn distribution
		float blinn_pdf = ((m_Ns + 1.f) * powf(costheta, m_Ns)) / (2.f * PI_F * 4.f * Dot(Wo, wh));

		if (Dot(Wo, wh) <= 0.f) 
			blinn_pdf = 0.f;

		Pdf = blinn_pdf;
	}

	HOD float BlinnD(const Vec3f& Wh)
	{
		float costhetah = AbsCosTheta(Wh);
		return (m_Ns + 2) * INV_TWO_PI_F * powf(costhetah, m_Ns);
	}

	HOD float BlinnPdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		Vec3f wh = Normalize(Wo + Wi);
		float costheta = AbsCosTheta(wh);
		// Compute PDF for $\wi$ from Blinn distribution
		
		float blinn_pdf = ((m_Ns + 1.f) * powf(costheta, m_Ns)) / (2.f * PI_F * 4.f * Dot(Wo, wh));
		
		if (Dot(Wo, wh) <= 0.f) 
			blinn_pdf = 0.f;

		return blinn_pdf;
	}

	HOD CColorXyz SchlickFresnel(const float& CosTheta)
	{
		return m_Ks + powf(1.0f - CosTheta, 5.0f) * (SPEC_WHITE - m_Ks);
	}

	HOD CColorXyz F(const Vec3f& Wo, const Vec3f& Wi)
	{
		if (!SameHemisphere(Wo, Wi)) 
			return SPEC_BLACK;

		CColorXyz diffuse = (28.0f / (23.0f * PI_F)) * m_Kd * (SPEC_WHITE - m_Ks) * (1.0f - powf(1.0f - 0.5f * AbsCosTheta(Wi), 5.0f)) * (1.0f - powf(1.0f - 0.5f * AbsCosTheta(Wo), 5.0f));
		Vec3f wh = Wi + Wo;
		
		if (wh.x == 0. && wh.y == 0. && wh.z == 0.) 
			return SPEC_BLACK;

		wh = Normalize(wh);
		
		CColorXyz specular = BlinnD(wh) / (4.f * AbsDot(Wi, wh) * max(AbsCosTheta(Wi), AbsCosTheta(Wo))) * 	SchlickFresnel(Dot(Wi, wh));
		
		return diffuse + specular;
	}

	HOD CColorXyz SampleF(const Vec3f& Wo, Vec3f& Wi, Vec2f U, float& Pdf)
	{
		if (U.x < .5)
		{
			U.x = 2.f * U.x;
			
			Wi = CosineWeightedHemisphere(U);
			
			if (Wo.z < 0.0f)
				Wi.z *= -1.f;
		}
		else
		{
			U.x = 2.f * (U.x - .5f);
			
			BlinnSampleF(Wo, Wi, U, Pdf);
			
			if (!SameHemisphere(Wo, Wi)) 
				return SPEC_BLACK;
		}
		
		Pdf = this->Pdf(Wo, Wi);

		return F(Wo, Wi);
	}

	HOD float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		if (!SameHemisphere(Wo, Wi)) 
			return 0.f;

		return .5f * (AbsCosTheta(Wi) * INV_PI_F + BlinnPdf(Wo, Wi));
	}
};

struct CVolumeMatLambert
{
	Vec3f		m_N;
	CColorXyz	m_Kd;
	
	HOD void Init(const Vec3f& N, const CColorXyz& Kd)
	{
		m_N		= N;
		m_Kd	= Kd;
	}

	HOD CColorXyz F(const Vec3f& Wo, const Vec3f& Wi)
	{
		return m_Kd * INV_PI_F;
	}

	HOD CColorXyz SampleF(const Vec3f& Wo, Vec3f& Wi, Vec2f U, float& Pdf)
	{
		Wi = CosineWeightedHemisphere(U);
		
		Pdf = this->Pdf(Wo, Wi);

		return F(Wo, Wi);
	}

	HOD float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		if (!SameHemisphere(Wo, Wi)) 
			return 0.f;

		return Wi.z * INV_PI_F;
	}
};


// Computes the attenuation through the volume
DEV inline CColorXyz TransmittanceRM(CScene* pDevScene, CCudaRNG& RNG, CRay& R,const float& StepSize)
{
	// Near and far intersections with volume axis aligned bounding box
	float NearT = 0.0f, FarT = 0.0f;

	// Intersect with volume axis aligned bounding box
	if (!pDevScene->m_BoundingBox.Intersect(R, &NearT, &FarT))
		return SPEC_BLACK;

	// Clamp to near plane if necessary
	if (NearT < 0.0f) 
		NearT = 0.0f;     

	CColorXyz Lt = SPEC_WHITE;

	NearT += RNG.Get1() * StepSize;

	// Accumulate
	while (NearT < R.m_MaxT)
	{
		// Determine sample point
		const Vec3f SP = R(NearT);

		// Fetch density
		const short D = (float)(SHRT_MAX * tex3D(gTexDensity, pDevScene->m_BoundingBox.m_MinP.x + (SP.x / pDevScene->m_BoundingBox.m_MaxP.x), pDevScene->m_BoundingBox.m_MinP.y + (SP.y / pDevScene->m_BoundingBox.m_MaxP.y), pDevScene->m_BoundingBox.m_MinP.z + (SP.z / pDevScene->m_BoundingBox.m_MaxP.z)));
		
		// We ignore air density
		if (D == 0)
		{
			// Increase extent
			NearT += StepSize;
			continue;
		}

		// Get shadow opacity
		const float	Opacity = pDevScene->m_TransferFunctions.m_Kt.F(D).r;

		if (Opacity > 0.0f)
		{
			// Compute eye transmittance
			Lt *= expf(-(Opacity * StepSize));

			// Exit if eye transmittance is very small
			if (Lt.y() < 0.1f)
				break;
		}

		// Increase extent
		NearT += StepSize;
	}

	return Lt;
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
	float Ss = 0.2f, Te = MinT + RNG.Get1() * Ss, D = 0.0f;

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
		const short D = (float)(SHRT_MAX * tex3D(gTexDensity, pDevScene->m_BoundingBox.m_MinP.x + (Pe.x / pDevScene->m_BoundingBox.m_MaxP.x), pDevScene->m_BoundingBox.m_MinP.y + (Pe.y / pDevScene->m_BoundingBox.m_MaxP.y), pDevScene->m_BoundingBox.m_MinP.z + (Pe.z / pDevScene->m_BoundingBox.m_MaxP.z)));

		// We ignore air density
		if (D == 0.0f)
			continue;

		// Get opacity at eye point
		const float Tr = pDevScene->m_TransferFunctions.m_Kt.F(D).r;
//		const CColorXyz	Ke = pDevScene->m_Volume.Ke(D);
		
		// Add emission
//		Ltr += Ke;

		// Compute outgoing direction
//		const Vec3f Wo = Normalize(-Re.m_D);

		// Obtain normal
//		Gn = ComputeGradient(Pe, Wo);

		// Exit if air, or not within hemisphere
		if (Tr < 0.01f)
			continue;

		Vec3f Pl(150, 150, 150);

		Rl.m_O = Pl;
		Rl.m_D = Normalize((Vec3f(0.0f) - Pl));
		Rl.m_MinT = 0.0f;
		Rl.m_MaxT = (Vec3f(0.0f) - Pl).Length();

		CColorXyz T = TransmittanceRM(pDevScene, RNG, Rl, Ss);
		// Estimate direct light at eye point
//	 	Lv += EyeTr * UniformSampleOneLight(Wo, EyeP[TID] + Normal[TID] * gScene.m_Volume.m_RayEpsilon, Normal[TID], Rnd, StepSize);
		Lv += Ltr * T * SPEC_WHITE;

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

KERNEL void KrnlBlurXyzH(CColorXyz* pImage, CColorXyz* pTempImage, CResolution2D Resolution, CGaussianFilter GaussianFilter)
{
	const int X 	= (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global y
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global x
	const int PID	= (Y * Resolution.m_XY.x) + X;					// Get pixel ID	

	// Exit if beyond image boundaries
	if (X >= Resolution.m_XY.x || Y >= Resolution.m_XY.y)
		return;

	// Compute filter extent
	const int X0 = max((int)ceilf(X - GaussianFilter.xWidth), 0);
	const int X1 = min((int)floorf(X + GaussianFilter.xWidth), Resolution.m_XY.x - 1);

	// Accumulated color
	CColorXyz Sum;

	// Weights
	float FW = 1.0f, SumW = 0.0f;

	for (int x = X0; x <= X1; x++)
	{
		// Compute filter weight
		FW = GaussianFilter.Evaluate(fabs((float)(x - X) / (0.5f * GaussianFilter.xWidth)), 0.0f);

		Sum		+= FW * pImage[(Y * Resolution.m_XY.x) + x];
		SumW	+= FW;
	}

	__syncthreads();

	// Write to temporary image
	pTempImage[PID] = Sum / SumW;
}

// ToDo: Add description
KERNEL void KrnlBlurXyzV(CColorXyz* pImage, CColorXyz* pTempImage, CResolution2D Resolution, CGaussianFilter GaussianFilter)
{
	const int X 	= (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global y
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global x
	const int PID	= (Y * Resolution.m_XY.x) + X;					// Get pixel ID	

	// Exit if beyond image boundaries
	if (X >= Resolution.m_XY.x || Y >= Resolution.m_XY.y)
		return;

	// Compute filter extent
	const int Y0 = max((int)ceilf (Y - GaussianFilter.yWidth), 0);
	const int Y1 = min((int)floorf(Y + GaussianFilter.yWidth), Resolution.m_XY.y - 1);

	// Accumulated color
	CColorXyz Sum;

	// Weights
	float FW = 1.0f, SumW = 0.0f;

	for (int y = Y0; y <= Y1; y++)
	{
		// Compute filter weight
		FW = GaussianFilter.Evaluate(0.0f, fabs((float)(y - Y) / (0.5f * GaussianFilter.yWidth)));

		Sum		+= FW * pTempImage[(y * Resolution.m_XY.x) + X];
		SumW	+= FW;
	}

	__syncthreads();

	// Write to image
	pImage[PID]	= Sum / SumW;
}

// ToDo: Add description
void BlurImageXyz(CColorXyz* pImage, CColorXyz* pTempImage, const CResolution2D& Resolution, const float& Radius)
{
	const dim3 KernelBlock(32, 8);
	const dim3 KernelGrid((int)ceilf((float)Resolution.m_XY.x / (float)KernelBlock.x), (int)ceilf((float)Resolution.m_XY.y / (float)KernelBlock.y));

	// Create gaussian filter
	CGaussianFilter GaussianFilter(2.0f * Radius, 2.0f * Radius, 2.0f);

	KrnlBlurXyzH<<<KernelGrid, KernelBlock>>>(pImage, pTempImage, Resolution, GaussianFilter); 
	KrnlBlurXyzV<<<KernelGrid, KernelBlock>>>(pImage, pTempImage, Resolution, GaussianFilter); 
}

// ToDo: Add description
KERNEL void KrnlComputeEstimate(int Width, int Height, CColorXyz* gpEstFrameXyz, CColorXyz* pAccEstXyz, float N, float Exposure, unsigned char* pPixels)
{
	const int X 	= (blockIdx.x * blockDim.x) + threadIdx.x;		// Get global Y
	const int Y		= (blockIdx.y * blockDim.y) + threadIdx.y;		// Get global X
	const int PID	= (Y * Width) + X;								// Get pixel ID	

	// Exit if beyond image boundaries
	if (X >= Width || Y >= Height)
		return;

	pAccEstXyz[PID] += gpEstFrameXyz[PID];

	const CColorXyz L = pAccEstXyz[PID] / (float)__max(1.0f, N);

	float InvGamma = 1.0f / 2.2f;

	CColorRgbHdr RgbHdr = CColorRgbHdr(L.c[0], L.c[1], L.c[2]);

	RgbHdr.r = Clamp(1.0f - expf(-(RgbHdr.r / Exposure)), 0.0, 1.0f);
	RgbHdr.g = Clamp(1.0f - expf(-(RgbHdr.g / Exposure)), 0.0, 1.0f);
	RgbHdr.b = Clamp(1.0f - expf(-(RgbHdr.b / Exposure)), 0.0, 1.0f);

	pPixels[(3 * (Y * Width + X)) + 0] = (unsigned char)Clamp((255.0f * powf(RgbHdr.r, InvGamma)), 0.0f, 255.0f);
	pPixels[(3 * (Y * Width + X)) + 1] = (unsigned char)Clamp((255.0f * powf(RgbHdr.g, InvGamma)), 0.0f, 255.0f);
	pPixels[(3 * (Y * Width + X)) + 2] = (unsigned char)Clamp((255.0f * powf(RgbHdr.b, InvGamma)), 0.0f, 255.0f);
}

void ComputeEstimate(int Width, int Height, CColorXyz* pEstFrameXyz, CColorXyz* pAccEstXyz, float N, float Exposure, unsigned char* pPixels)
{
	const dim3 KernelBlock(8, 8);
	const dim3 KernelGrid((int)ceilf((float)Width / (float)KernelBlock.x), (int)ceilf((float)Height / (float)KernelBlock.y));

	KrnlComputeEstimate<<<KernelGrid, KernelBlock>>>(Width, Height, pEstFrameXyz, pAccEstXyz, N, Exposure, pPixels); 
}