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

class CLambertian
{
public:
	DEV CLambertian(const ColorXYZf& Kd)
	{
		m_Kd = Kd;
	}

	DEV ~CLambertian(void)
	{
	}

	DEV ColorXYZf F(const Vec3f& Wo, const Vec3f& Wi)
	{
		return m_Kd * INV_PI_F;
	}

	DEV ColorXYZf SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const Vec2f& U)
	{
		Wi = CosineWeightedHemisphere(U);

		if (Wo[2] < 0.0f)
			Wi[2] *= -1.0f;

		Pdf = this->Pdf(Wo, Wi);

		return this->F(Wo, Wi);
	}

	DEV float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		return SameHemisphere(Wo, Wi) ? AbsCosTheta(Wi) * INV_PI_F : 0.0f;
	}

	ColorXYZf	m_Kd;
};

DEV inline ColorXYZf FrDiel(float cosi, float cost, const ColorXYZf &etai, const ColorXYZf &etat)
{
	ColorXYZf Rparl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
	ColorXYZf Rperp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
	return (Rparl*Rparl + Rperp*Rperp) / 2.f;
}

class CFresnel
{
public:
	DEV CFresnel(float ei, float et) :
	  eta_i(ei),
		  eta_t(et)
	  {
	  }

	  DEV  ~CFresnel(void)
	  {
	  }

	  DEV ColorXYZf Evaluate(float cosi)
	  {
		  // Compute Fresnel reflectance for dielectric
		  cosi = Clamp(cosi, -1.0f, 1.0f);

		  // Compute indices of refraction for dielectric
		  bool entering = cosi > 0.0f;
		  float ei = eta_i, et = eta_t;

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

	  float eta_i, eta_t;
};

class CBlinn
{
public:
	DEV CBlinn(const float& Exponent) :
	  m_Exponent(Exponent)
	  {
	  }

	  DEV ~CBlinn(void)
	  {
	  }

	  DEV void SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const Vec2f& U)
	  {
		  // Compute sampled half-angle vector $\wh$ for Blinn distribution
		  float costheta = powf(U[0], 1.f / (m_Exponent+1));
		  float sintheta = sqrtf(max(0.f, 1.f - costheta*costheta));
		  float phi = U[1] * 2.f * PI_F;

		  Vec3f wh = SphericalDirection(sintheta, costheta, phi);

		  if (!SameHemisphere(Wo, wh))
			  wh = -wh;

		  // Compute incident direction by reflecting about $\wh$
		  Wi = -Wo + 2.f * Dot(Wo, wh) * wh;

		  // Compute PDF for $\wi$ from Blinn distribution
		  float blinn_pdf = ((m_Exponent + 1.f) * powf(costheta, m_Exponent)) / (2.f * PI_F * 4.f * Dot(Wo, wh));

		  if (Dot(Wo, wh) <= 0.f)
			  blinn_pdf = 0.f;

		  Pdf = blinn_pdf;
	  }

	  DEV float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	  {
		  Vec3f wh = Normalize(Wo + Wi);

		  float costheta = AbsCosTheta(wh);
		  // Compute PDF for $\wi$ from Blinn distribution
		  float blinn_pdf = ((m_Exponent + 1.f) * powf(costheta, m_Exponent)) / (2.f * PI_F * 4.f * Dot(Wo, wh));

		  if (Dot(Wo, wh) <= 0.0f)
			  blinn_pdf = 0.0f;

		  return blinn_pdf;
	  }

	  DEV float D(const Vec3f& wh)
	  {
		  float costhetah = AbsCosTheta(wh);
		  return (m_Exponent+2) * INV_TWO_PI_F * powf(costhetah, m_Exponent);
	  }

	  float	m_Exponent;
};

class CMicrofacet
{
public:
	DEV CMicrofacet(const ColorXYZf& Reflectance, const float& Ior, const float& Exponent) :
	  m_R(Reflectance),
		  m_Fresnel(Ior, 1.0f),
		  m_Blinn(Exponent)
	  {
	  }

	  DEV ~CMicrofacet(void)
	  {
	  }

	  DEV ColorXYZf F(const Vec3f& wo, const Vec3f& wi)
	  {
		  float cosThetaO = AbsCosTheta(wo);
		  float cosThetaI = AbsCosTheta(wi);

		  if (cosThetaI == 0.f || cosThetaO == 0.f)
			  return ColorXYZf(0.0f);

		  Vec3f wh = wi + wo;

		  if (wh[0] == 0. && wh[1] == 0. && wh[2] == 0.)
			  return ColorXYZf(0.0f);

		  wh = Normalize(wh);
		  float cosThetaH = Dot(wi, wh);

		  ColorXYZf F(1.0f);// = m_Fresnel.Evaluate(cosThetaH);

		  return m_R * m_Blinn.D(wh) * G(wo, wi, wh) * F / (4.0f * cosThetaI * cosThetaO);
	  }

	  DEV ColorXYZf SampleF(const Vec3f& wo, Vec3f& wi, float& Pdf, const Vec2f& U)
	  {
		  m_Blinn.SampleF(wo, wi, Pdf, U);

		  if (!SameHemisphere(wo, wi))
			  return SPEC_BLACK;

		  return this->F(wo, wi);
	  }

	  DEV float Pdf(const Vec3f& wo, const Vec3f& wi)
	  {
		  if (!SameHemisphere(wo, wi))
			  return 0.0f;

		  return m_Blinn.Pdf(wo, wi);
	  }

	  DEV float G(const Vec3f& wo, const Vec3f& wi, const Vec3f& wh)
	  {
		  float NdotWh = AbsCosTheta(wh);
		  float NdotWo = AbsCosTheta(wo);
		  float NdotWi = AbsCosTheta(wi);
		  float WOdotWh = AbsDot(wo, wh);

		  return min(1.f, min((2.f * NdotWh * NdotWo / WOdotWh), (2.f * NdotWh * NdotWi / WOdotWh)));
	  }

	  ColorXYZf		m_R;
	  CFresnel		m_Fresnel;
	  CBlinn		m_Blinn;

};

class CIsotropicPhase
{
public:
	DEV CIsotropicPhase(const ColorXYZf& Kd) :
		m_Kd(Kd)
	{
	}

	DEV ~CIsotropicPhase(void)
	{
	}

	DEV ColorXYZf F(const Vec3f& Wo, const Vec3f& Wi)
	{
		return m_Kd * INV_PI_F;
	}

	DEV ColorXYZf SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const Vec2f& U)
	{
		Wi	= UniformSampleSphereSurface(U);
		Pdf	= this->Pdf(Wo, Wi);

		return F(Wo, Wi);
	}

	DEV float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		return INV_4_PI_F;
	}

	ColorXYZf	m_Kd;
};

class CBRDF
{
public:
	DEV CBRDF(const Vec3f& N, const Vec3f& Wo, const ColorXYZf& Kd, const ColorXYZf& Ks, const float& Ior, const float& Exponent) :
		m_Lambertian(Kd),
		m_Microfacet(Ks, Ior, Exponent),
		m_Nn(Normalize(N)),
		m_Nu(Normalize(Cross(N, Wo))),
		m_Nv(Normalize(Cross(N, m_Nu)))
	{
	}

	DEV ~CBRDF(void)
	{
	}

	DEV Vec3f WorldToLocal(const Vec3f& W)
	{
		return Vec3f(Dot(W, m_Nu), Dot(W, m_Nv), Dot(W, m_Nn));
	}

	DEV Vec3f LocalToWorld(const Vec3f& W)
	{
		return Vec3f(	m_Nu[0] * W[0] + m_Nv[0] * W[1] + m_Nn[0] * W[2],
						m_Nu[1] * W[0] + m_Nv[1] * W[1] + m_Nn[1] * W[2],
						m_Nu[2] * W[0] + m_Nv[2] * W[1] + m_Nn[2] * W[2]);
	}

	DEV ColorXYZf F(const Vec3f& Wo, const Vec3f& Wi)
	{
		const Vec3f Wol = WorldToLocal(Wo);
		const Vec3f Wil = WorldToLocal(Wi);

		ColorXYZf R;

		R += m_Lambertian.F(Wol, Wil);
		R += m_Microfacet.F(Wol, Wil);

		return R;
	}

	DEV ColorXYZf SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const BrdfSample& S)
	{
		const Vec3f Wol = WorldToLocal(Wo);
		Vec3f Wil;

		ColorXYZf R;

		if (S.m_Component <= 0.5f)
		{
			m_Lambertian.SampleF(Wol, Wil, Pdf, S.m_Dir);
		}
		else
		{
			m_Microfacet.SampleF(Wol, Wil, Pdf, S.m_Dir);
		}

		Pdf += m_Lambertian.Pdf(Wol, Wil);
		Pdf += m_Microfacet.Pdf(Wol, Wil);

		R += m_Lambertian.F(Wol, Wil);
		R += m_Microfacet.F(Wol, Wil);

		Wi = LocalToWorld(Wil);

		return R;
	}

	DEV float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		const Vec3f Wol = WorldToLocal(Wo);
		const Vec3f Wil = WorldToLocal(Wi);

		float Pdf = 0.0f;

		Pdf += m_Lambertian.Pdf(Wol, Wil);
		Pdf += m_Microfacet.Pdf(Wol, Wil);

		return Pdf;
	}

	Vec3f			m_Nn;
	Vec3f			m_Nu;
	Vec3f			m_Nv;
	CLambertian		m_Lambertian;
	CMicrofacet		m_Microfacet;
};

class CVolumeShader
{
public:
	enum EType
	{
		Brdf,
		Phase
	};

	DEV CVolumeShader(const EType& Type, const Vec3f& N, const Vec3f& Wo, const ColorXYZf& Kd, const ColorXYZf& Ks, const float& Ior, const float& Exponent) :
		m_Type(Type),
		m_Brdf(N, Wo, Kd, Ks, Ior, Exponent),
		m_IsotropicPhase(Kd)
	{
	}

	DEV ~CVolumeShader(void)
	{
	}

	DEV ColorXYZf F(Vec3f Wo, Vec3f Wi)
	{
		switch (m_Type)
		{
			case Brdf:
				return m_Brdf.F(Wo, Wi);

			case Phase:
				return m_IsotropicPhase.F(Wo, Wi);
		}

		return 1.0f;
	}

	DEV ColorXYZf SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const BrdfSample& S)
	{
		switch (m_Type)
		{
			case Brdf:
				return m_Brdf.SampleF(Wo, Wi, Pdf, S);

			case Phase:
				return m_IsotropicPhase.SampleF(Wo, Wi, Pdf, S.m_Dir);
		}

		return ColorXYZf(0.0f);
	}

	DEV float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		switch (m_Type)
		{
			case Brdf:
				return m_Brdf.Pdf(Wo, Wi);

			case Phase:
				return m_IsotropicPhase.Pdf(Wo, Wi);
		}

		return 1.0f;
	}

	EType				m_Type;
	CBRDF				m_Brdf;
	CIsotropicPhase		m_IsotropicPhase;
};