/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Geometry.cuh"
#include "MonteCarlo.cuh"
#include "Sample.cuh"
#include "Texture.cuh"

namespace ExposureRender
{

struct Lambertian
{
	DEVICE Lambertian(void)
	{
	}

	DEVICE Lambertian(const ColorXYZf& Kd)
	{
		this->Kd = Kd;
	}

	DEVICE ColorXYZf F(const Vec3f& Wo, const Vec3f& Wi)
	{
		return Kd * INV_PI_F;
	}

	DEVICE ColorXYZf SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const Vec2f& U)
	{
		Wi = CosineWeightedHemisphere(U);

		if (Wo[2] < 0.0f)
			Wi[2] *= -1.0f;

		Pdf = this->Pdf(Wo, Wi);

		return this->F(Wo, Wi);
	}

	DEVICE float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		return SameHemisphere(Wo, Wi) ? AbsCosTheta(Wi) * INV_PI_F : 0.0f;
	}

	DEVICE Lambertian& operator = (const Lambertian& Other)
	{
		this->Kd = Other.Kd;
		return *this;
	}

	ColorXYZf	Kd;
};

DEVICE inline ColorXYZf FrDiel(float cosi, float cost, const ColorXYZf &etai, const ColorXYZf &etat)
{
	ColorXYZf Rparl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
	ColorXYZf Rperp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
	return (Rparl*Rparl + Rperp*Rperp) / 2.f;
}

struct Fresnel
{
	DEVICE Fresnel(void)
	{
	}

	DEVICE Fresnel(float ei, float et) :
		EtaI(ei),
		EtaT(et)
	{
	}

	DEVICE ColorXYZf Evaluate(float cosi)
	{
		// Compute Fresnel reflectance for dielectric
		cosi = Clamp(cosi, -1.0f, 1.0f);

		// Compute indices of refraction for dielectric
		bool entering = cosi > 0.0f;
		float ei = EtaI, et = EtaT;

		if (!entering)
		swap(ei, et);

		// Compute _sint_ using Snell's law
		float sint = ei/et * sqrtf(max(0.f, 1.f - cosi*cosi));

		if (sint >= 1.0f)
		{
			// Handle total internal reflection
			return 1.0f;
		}
		else
		{
			float cost = sqrtf(max(0.f, 1.0f - sint * sint));
			return FrDiel(fabsf(cosi), cost, ei, et);
		}
	}

	DEVICE Fresnel& operator = (const Fresnel& Other)
	{
		this->EtaI = Other.EtaI;
		this->EtaT = Other.EtaT;

		return *this;
	}

	float EtaI, EtaT;
};

struct Blinn
{
	DEVICE Blinn(void)
	{
	}

	DEVICE Blinn(const float& Exponent) :
		Exponent(Exponent)
	{
	}

	DEVICE void SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const Vec2f& U)
	{
		// Compute sampled half-angle vector $\wh$ for Blinn distribution
		float costheta = powf(U[0], 1.f / (this->Exponent+1));
		float sintheta = sqrtf(max(0.f, 1.f - costheta*costheta));
		float phi = U[1] * 2.f * PI_F;

		Vec3f wh = SphericalDirection(sintheta, costheta, phi);

		if (!SameHemisphere(Wo, wh))
			wh = -wh;

		// Compute incident direction by reflecting about $\wh$
		Wi = -Wo + 2.f * Dot(Wo, wh) * wh;

		// Compute PDF for $\wi$ from Blinn distribution
		float blinn_pdf = ((Exponent + 1.f) * powf(costheta, this->Exponent)) / (2.f * PI_F * 4.f * Dot(Wo, wh));

		if (Dot(Wo, wh) <= 0.f)
			blinn_pdf = 0.f;

		Pdf = blinn_pdf;
	}

	DEVICE float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		// Compute half angle vector
		const Vec3f Wh = Normalize(Wo + Wi);

		const float CosTheta = AbsCosTheta(Wh);

		// Compute PDF for $\wi$ from Blinn distribution
		float Pdf = ((this->Exponent + 1.0f) * powf(CosTheta, this->Exponent)) / (2.0f * PI_F * 4.0f * Dot(Wo, Wh));

		if (Dot(Wo, Wh) <= 0.0f)
			Pdf = 0.0f;

		return Pdf;
	}

	DEVICE float D(const Vec3f& Wh)
	{
		float CosThetaH = AbsCosTheta(Wh);
		return (this->Exponent + 2) * INV_TWO_PI_F * powf(CosThetaH, this->Exponent);
	}

	DEVICE Blinn& operator = (const Blinn& Other)
	{
		this->Exponent = Other.Exponent;

		return *this;
	}

	float	Exponent;
};

struct Microfacet
{
	DEVICE Microfacet(void)
	{
	}

	DEVICE Microfacet(const ColorXYZf& Reflectance, const float& Ior, const float& Exponent) :
		R(Reflectance),
		Fresnel(1.0f, Ior),
		Blinn(Exponent)
	{
	}

	DEVICE ColorXYZf F(const Vec3f& Wo, const Vec3f& Wi)
	{
		float cosThetaO = AbsCosTheta(Wo);
		float cosThetaI = AbsCosTheta(Wi);

		if (cosThetaI == 0.f || cosThetaO == 0.f)
			return ColorXYZf(0.0f);

		Vec3f Wh = Wi + Wo;

		if (Wh[0] == 0. && Wh[1] == 0. && Wh[2] == 0.)
		return ColorXYZf(0.0f);

		Wh = Normalize(Wh);
		float cosThetaH = Dot(Wi, Wh);

		ColorXYZf F = this->Fresnel.Evaluate(cosThetaH);

		return this->R * this->Blinn.D(Wh) * G(Wo, Wi, Wh) * F / (4.0f * cosThetaI * cosThetaO);
	}

	DEVICE ColorXYZf SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const Vec2f& U)
	{
		this->Blinn.SampleF(Wo, Wi, Pdf, U);

		if (!SameHemisphere(Wo, Wi))
			return SPEC_BLACK;

		return this->F(Wo, Wi);
	}

	DEVICE float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		if (!SameHemisphere(Wo, Wi))
			return 0.0f;

		return Blinn.Pdf(Wo, Wi);
	}

	DEVICE float G(const Vec3f& Wo, const Vec3f& Wi, const Vec3f& Wh)
	{
		const float NdotWh 	= AbsCosTheta(Wh);
		const float NdotWo 	= AbsCosTheta(Wo);
		const float NdotWi 	= AbsCosTheta(Wi);
		const float WOdotWh = AbsDot(Wo, Wh);

		return min(1.0f, min((2.0f * NdotWh * NdotWo / WOdotWh), (2.0f * NdotWh * NdotWi / WOdotWh)));
	}

	DEVICE Microfacet& operator = (const Microfacet& Other)
	{
		this->R			= Other.R;
		this->Fresnel	= Other.Fresnel;
		this->Blinn		= Other.Blinn;

		return *this;
	}

	ColorXYZf	R;
	Fresnel	Fresnel;
	Blinn		Blinn;

};

struct IsotropicPhase
{
	DEVICE IsotropicPhase(void)
	{
	}

	DEVICE IsotropicPhase(const ColorXYZf& Kd) :
		Kd(Kd)
	{
	}

	DEVICE ColorXYZf F(const Vec3f& Wo, const Vec3f& Wi)
	{
		return Kd * INV_PI_F;
	}

	DEVICE ColorXYZf SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const Vec2f& U)
	{
		Wi	= UniformSampleSphereSurface(U);
		Pdf	= this->Pdf(Wo, Wi);

		return F(Wo, Wi);
	}

	DEVICE float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		return INV_FOUR_PI_F;
	}

	DEVICE IsotropicPhase& operator = (const IsotropicPhase& Other)
	{
		this->Kd = Other.Kd;

		return *this;
	}

	ColorXYZf	Kd;
};

struct BRDF
{
	DEVICE BRDF(void)
	{
	}

	DEVICE BRDF(const Vec3f& N, const Vec3f& Wo, const ColorXYZf& Kd, const ColorXYZf& Ks, const float& Ior, const float& Exponent) :
		Lambertian(Kd),
		Microfacet(Ks, Ior, Exponent),
		Nn(Normalize(N)),
		Nu(Normalize(Cross(N, Wo))),
		Nv(Normalize(Cross(N, Nu)))
	{
	}

	DEVICE Vec3f WorldToLocal(const Vec3f& W)
	{
		return Vec3f(Dot(W, this->Nu), Dot(W, this->Nv), Dot(W, this->Nn));
	}

	DEVICE Vec3f LocalToWorld(const Vec3f& W)
	{
		return Vec3f(	this->Nu[0] * W[0] + this->Nv[0] * W[1] + this->Nn[0] * W[2],
						this->Nu[1] * W[0] + this->Nv[1] * W[1] + this->Nn[1] * W[2],
						this->Nu[2] * W[0] + this->Nv[2] * W[1] + this->Nn[2] * W[2]);
	}

	DEVICE ColorXYZf F(const Vec3f& Wo, const Vec3f& Wi)
	{
		const Vec3f Wol = WorldToLocal(Wo);
		const Vec3f Wil = WorldToLocal(Wi);

		ColorXYZf R;

		R += this->Lambertian.F(Wol, Wil);
		R += this->Microfacet.F(Wol, Wil);

		return R;
	}

	DEVICE ColorXYZf SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const BrdfSample& S)
	{
		const Vec3f Wol = WorldToLocal(Wo);
		Vec3f Wil;

		ColorXYZf R;

		if (S.Component <= 0.5f)
		{
			this->Lambertian.SampleF(Wol, Wil, Pdf, S.Dir);
		}
		else
		{
			this->Microfacet.SampleF(Wol, Wil, Pdf, S.Dir);
		}

		Pdf += this->Lambertian.Pdf(Wol, Wil);
		Pdf += this->Microfacet.Pdf(Wol, Wil);

		R += this->Lambertian.F(Wol, Wil);
		R += this->Microfacet.F(Wol, Wil);

		Wi = LocalToWorld(Wil);

		return R;
	}

	DEVICE float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		const Vec3f Wol = WorldToLocal(Wo);
		const Vec3f Wil = WorldToLocal(Wi);

		float Pdf = 0.0f;

		Pdf += this->Lambertian.Pdf(Wol, Wil);
		Pdf += this->Microfacet.Pdf(Wol, Wil);

		return Pdf;
	}

	DEVICE BRDF& operator = (const BRDF& Other)
	{
		this->Nn 			= Other.Nn;
		this->Nu 			= Other.Nu;
		this->Nv 			= Other.Nv;
		this->Lambertian 	= Other.Lambertian;
		this->Microfacet 	= Other.Microfacet;

		return *this;
	}

	Vec3f			Nn;
	Vec3f			Nu;
	Vec3f			Nv;
	Lambertian		Lambertian;
	Microfacet		Microfacet;
};

struct VolumeShader
{
	enum EType
	{
		Brdf,
		Phase
	};

	DEVICE VolumeShader(void)
	{
	}

	DEVICE VolumeShader(const EType& Type, const Vec3f& N, const Vec3f& Wo, const ColorXYZf& Kd, const ColorXYZf& Ks, const float& Ior, const float& Exponent) :
		Type(Type),
		BRDF(N, Wo, Kd, Ks, Ior, Exponent),
		IsotropicPhase(Kd)
	{
	}

	DEVICE ColorXYZf F(Vec3f Wo, Vec3f Wi)
	{
		switch (this->Type)
		{
			case Brdf:
				return this->BRDF.F(Wo, Wi);

			case Phase:
				return this->IsotropicPhase.F(Wo, Wi);
		}

		return 1.0f;
	}

	DEVICE ColorXYZf SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const BrdfSample& S)
	{
		switch (this->Type)
		{
			case Brdf:
				return this->BRDF.SampleF(Wo, Wi, Pdf, S);

			case Phase:
				return this->IsotropicPhase.SampleF(Wo, Wi, Pdf, S.Dir);
		}

		return ColorXYZf(0.0f);
	}

	DEVICE float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		switch (this->Type)
		{
			case Brdf:
				return this->BRDF.Pdf(Wo, Wi);

			case Phase:
				return this->IsotropicPhase.Pdf(Wo, Wi);
		}

		return 1.0f;
	}

	DEVICE VolumeShader& operator = (const VolumeShader& Other)
	{
		this->Type 				= Other.Type;
		this->BRDF 				= Other.BRDF;
		this->IsotropicPhase	= Other.IsotropicPhase;

		return *this;
	}

	EType				Type;
	BRDF				BRDF;
	IsotropicPhase		IsotropicPhase;
};

DEVICE_NI VolumeShader GetVolumeShader(ScatterEvent& SE, CRNG& RNG)
{
	const float I = GetIntensity(SE.P);

	bool BRDF = false;

	float PdfBrdf = 1.0f;

	switch (gRenderSettings.Shading.Type)
	{
		case 0:
		{
			BRDF = true;
			break;
		}
	
		case 1:
		{
			BRDF = false;
			break;
		}

		case 2:
		{
			const float NGM			= GradientMagnitude(SE.P) * gpTracer->Volume.GradientMagnitudeRange.Inv;
			const float Sensitivity	= 25;
			const float ExpGF		= 3;
			const float Exponent	= Sensitivity * powf(gRenderSettings.Shading.GradientFactor, ExpGF) * NGM;
			
			PdfBrdf = gRenderSettings.Shading.OpacityModulated ? GetOpacity(SE.P) * (1.0f - __expf(-Exponent)) : 1.0f - __expf(-Exponent);
			BRDF = RNG.Get1() <= PdfBrdf;
			break;
		}

		case 3:
		{
			const float NGM = GradientMagnitude(SE.P) * gpTracer->Volume.GradientMagnitudeRange.Inv;
			
			PdfBrdf = 1.0f - powf(1.0f - NGM, 2.0f);
			BRDF = RNG.Get1() < PdfBrdf;
			break;
		}

		case 4:
		{
			const float NGM = GradientMagnitude(SE.P) * gpTracer->Volume.GradientMagnitudeRange.Inv;

			if (NGM > gRenderSettings.Shading.GradientThreshold)
				BRDF = true;
			else
				BRDF = false;
		}
	}

	if (BRDF)
		return VolumeShader(VolumeShader::Brdf, SE.N, SE.Wo, GetDiffuse(I), GetSpecular(I), gRenderSettings.Shading.IndexOfReflection, GetGlossiness(I));
	else
		return VolumeShader(VolumeShader::Phase, SE.N, SE.Wo, GetDiffuse(I), GetSpecular(I), gRenderSettings.Shading.IndexOfReflection, GetGlossiness(I));
}

DEVICE_NI VolumeShader GetLightShader(ScatterEvent& SE, CRNG& RNG)
{
	return VolumeShader(VolumeShader::Brdf, SE.N, SE.Wo, ColorXYZf(0.0f), ColorXYZf(0.0f), 5.0f, 0.0f);
}

DEVICE_NI VolumeShader GetReflectorShader(ScatterEvent& SE, CRNG& RNG)
{
	return VolumeShader(VolumeShader::Brdf, SE.N, SE.Wo, EvaluateTexture2D(gObjects.List[SE.ObjectID].DiffuseTextureID, SE.UV), EvaluateTexture2D(gObjects.List[SE.ObjectID].SpecularTextureID, SE.UV), 10.0f, GlossinessExponent(EvaluateTexture2D(gObjects.List[SE.ObjectID].GlossinessTextureID, SE.UV).Y()));
}

}
