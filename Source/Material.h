
#include "Geometry.h"
#include "MonteCarlo.h"
#include "Light.h"

class CLambertian
{
public:
	HOD CLambertian(const CColorXyz& Kd)
	{
		m_Kd = Kd;
	}

	HOD ~CLambertian(void)
	{
	}

	HOD CColorXyz F(const Vec3f& Wo, const Vec3f& Wi)
	{
		return m_Kd * INV_PI_F;
	}

	HOD CColorXyz SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const Vec2f& U)
	{
		Wi = CosineWeightedHemisphere(U);

		if (Wo.z < 0.0f)
			Wi.z *= -1.0f;

		Pdf = this->Pdf(Wo, Wi);

		return this->F(Wo, Wi);
	}

	HOD float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	{
		return SameHemisphere(Wo, Wi) ? AbsCosTheta(Wi) * INV_PI_F : 0.0f;
	}

	CColorXyz	m_Kd;
};

HOD inline CColorXyz FrDiel(float cosi, float cost, const CColorXyz &etai, const CColorXyz &etat)
{
	CColorXyz Rparl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
	CColorXyz Rperp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
	return (Rparl*Rparl + Rperp*Rperp) / 2.f;
}

class CFresnel
{
public:
	HOD CFresnel(float ei, float et) :
	  eta_i(ei),
		  eta_t(et)
	  {
	  }

	  HOD  ~CFresnel(void)
	  {
	  }

	  HOD CColorXyz Evaluate(float cosi)
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
	HOD CBlinn(const float& Exponent) :
	  m_Exponent(Exponent)
	  {
	  }

	  HOD ~CBlinn(void)
	  {
	  }

	  HOD void SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const Vec2f& U)
	  {
		  // Compute sampled half-angle vector $\wh$ for Blinn distribution
		  float costheta = powf(U.x, 1.f / (m_Exponent+1));
		  float sintheta = sqrtf(max(0.f, 1.f - costheta*costheta));
		  float phi = U.y * 2.f * PI_F;

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

	  HOD float Pdf(const Vec3f& Wo, const Vec3f& Wi)
	  {
		  Vec3f wh = Normalize(Wo + Wi);

		  float costheta = AbsCosTheta(wh);
		  // Compute PDF for $\wi$ from Blinn distribution
		  float blinn_pdf = ((m_Exponent + 1.f) * powf(costheta, m_Exponent)) / (2.f * PI_F * 4.f * Dot(Wo, wh));

		  if (Dot(Wo, wh) <= 0.0f)
			  blinn_pdf = 0.0f;

		  return blinn_pdf;
	  }

	  HOD float D(const Vec3f& wh)
	  {
		  float costhetah = AbsCosTheta(wh);
		  return (m_Exponent+2) * INV_TWO_PI_F * powf(costhetah, m_Exponent);
	  }

	  float	m_Exponent;
};

class CMicrofacet
{
public:
	HOD CMicrofacet(const CColorXyz& Reflectance, const float& Ior, const float& Exponent) :
	  m_R(Reflectance),
		  m_Fresnel(Ior, 1.0f),
		  m_Blinn(Exponent)
	  {
	  }

	  HOD ~CMicrofacet(void)
	  {
	  }

	  HOD CColorXyz F(const Vec3f& wo, const Vec3f& wi)
	  {
		  float cosThetaO = AbsCosTheta(wo);
		  float cosThetaI = AbsCosTheta(wi);

		  if (cosThetaI == 0.f || cosThetaO == 0.f)
			  return SPEC_BLACK;

		  Vec3f wh = wi + wo;

		  if (wh.x == 0. && wh.y == 0. && wh.z == 0.)
			  return SPEC_BLACK;

		  wh = Normalize(wh);
		  float cosThetaH = Dot(wi, wh);

		  CColorXyz F = m_Fresnel.Evaluate(cosThetaH);

		  return m_R * m_Blinn.D(wh) * G(wo, wi, wh) * F / (4.f * cosThetaI * cosThetaO);
	  }

	  HOD CColorXyz SampleF(const Vec3f& wo, Vec3f& wi, float& Pdf, const Vec2f& U)
	  {
		  m_Blinn.SampleF(wo, wi, Pdf, U);

		  if (!SameHemisphere(wo, wi))
			  return SPEC_BLACK;

		  return this->F(wo, wi);
	  }

	  HOD float Pdf(const Vec3f& wo, const Vec3f& wi)
	  {
		  if (!SameHemisphere(wo, wi))
			  return 0.0f;

		  return m_Blinn.Pdf(wo, wi);
	  }

	  HOD float G(const Vec3f& wo, const Vec3f& wi, const Vec3f& wh)
	  {
		  float NdotWh = AbsCosTheta(wh);
		  float NdotWo = AbsCosTheta(wo);
		  float NdotWi = AbsCosTheta(wi);
		  float WOdotWh = AbsDot(wo, wh);

		  return min(1.f, min((2.f * NdotWh * NdotWo / WOdotWh), (2.f * NdotWh * NdotWi / WOdotWh)));
	  }

	  CColorXyz	m_R;
	  CFresnel	m_Fresnel;
	  CBlinn		m_Blinn;

};

class CBSDF
{
public:
	HOD CBSDF(const Vec3f& N, const Vec3f& W, const CColorXyz& Kd, const CColorXyz& Ks, const float& Ior, const float& Exponent) :
	  m_Lambertian(Kd),
		  m_Microfacet(Ks, Ior, Exponent),
		  m_Nn(N),
		  m_Nu(Normalize(Cross(N, W))),
		  m_Nv(Normalize(Cross(N, m_Nu)))
	  {
	  }

	  HOD ~CBSDF(void)
	  {
	  }

	  HOD Vec3f WorldToLocal(const Vec3f& W)
	  {
		  return Vec3f(Dot(W, m_Nu), Dot(W, m_Nv), Dot(W, m_Nn));
	  }

	  HOD Vec3f LocalToWorld(const Vec3f& W)
	  {
		  return Vec3f(	m_Nu.x * W.x + m_Nv.x * W.y + m_Nn.x * W.z,
			  m_Nu.y * W.x + m_Nv.y * W.y + m_Nn.y * W.z,
			  m_Nu.z * W.x + m_Nv.z * W.y + m_Nn.z * W.z);
	  }

	  HOD CColorXyz F(const Vec3f& Wo, const Vec3f& Wi)
	  {
		  const Vec3f Wol = WorldToLocal(Wo);
		  const Vec3f Wil = WorldToLocal(Wi);

		  CColorXyz R;

		  R += m_Lambertian.F(Wol, Wil);
		  R += m_Microfacet.F(Wol, Wil);

		  return R;
	  }

	  HOD CColorXyz SampleF(const Vec3f& Wo, Vec3f& Wi, float& Pdf, const CBsdfSample& S)
	  {
		  const Vec3f Wol = WorldToLocal(Wo);
		  Vec3f Wil;

		  CColorXyz R;

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

	  HOD float Pdf(const Vec3f& Wo, const Vec3f& Wi)
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