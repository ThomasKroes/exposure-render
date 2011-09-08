#pragma once

#include "Geometry.h"

// ToDo: Add description
class EXPOSURE_RENDER_DLL CFilter
{
public:
	HOD CFilter(float xw, float yw)
		: xWidth(xw), yWidth(yw), invXWidth(1.f/xw), invYWidth(1.f/yw)
	{
	}

	const float xWidth, yWidth;
	const float invXWidth, invYWidth;
};

// ToDo: Add description
class EXPOSURE_RENDER_DLL CGaussianFilter : public CFilter
{
public:
	HOD CGaussianFilter(float xw, float yw, float a)
		: CFilter(xw, yw), alpha(a), expX(expf(-alpha * xWidth * xWidth)),
		expY(expf(-alpha * yWidth * yWidth))
	{
		m_FilterTable33[0] = 0.2f;
		m_FilterTable33[1] = 0.1f;

		m_FilterTable55[0] = 0.5f;
		m_FilterTable55[1] = 0.4f;
		m_FilterTable55[2] = 0.3f;
		m_FilterTable55[3] = 0.15f;
		m_FilterTable55[4] = 0.1f;
	}

	// ToDo: Add description
	HOD float Evaluate(float x, float y)
	{
		return Gaussian(x, expX) * Gaussian(y, expY);
	}

	// GaussianFilter Private Data
	const float alpha;
	const float expX, expY;

	float		m_FilterTable33[2];
	float		m_FilterTable55[5];

	// GaussianFilter Utility Functions
	HOD float Gaussian(float d, float expv) const
	{
		return max(0.f, float(expf(-alpha * d * d) - expv));
	}
};

// ToDo: Add description
class EXPOSURE_RENDER_DLL CMitchellFilter : public CFilter
{
public:
	HOD CMitchellFilter(float b, float c, float xw, float yw)
		: CFilter(xw, yw), B(b), C(c)
	{
	}

	HOD float Evaluate(float x, float y)
	{
		return Mitchell1D(x * invXWidth) * Mitchell1D(y * invYWidth);
	}

	HOD float Mitchell1D(float x) const
	{
		x = fabsf(2.f * x);
		if (x > 1.f)
			return ((-B - 6*C) * x*x*x + (6*B + 30*C) * x*x +
			(-12*B - 48*C) * x + (8*B + 24*C)) * (1.f/6.f);
		else
			return ((12 - 9*B - 6*C) * x*x*x +
			(-18 + 12*B + 6*C) * x*x +
			(6 - 2*B)) * (1.f/6.f);
	}

private:
	const float B, C;
};