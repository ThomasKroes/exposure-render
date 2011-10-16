/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Defines.h"
#include "Enumerations.h"
#include "Dll.h"

#include <math.h>
#include <algorithm>

using namespace std;

HOD inline float Lerp(float t, float v1, float v2)
{
	return (1.f - t) * v1 + t * v2;
}

HOD inline float clamp2(float v, float a, float b)
{
	return max(a, min(v, b));
}

HOD  inline void swap(int& a, int& b)
{
	int t = a; a = b; b = t;
}

HOD  inline void swap(float& a, float& b)
{
	float t = a; a = b; b = t;
}

HOD inline void Swap(float* pF1, float* pF2)
{
	const float TempFloat = *pF1;

	*pF1 = *pF2;
	*pF2 = TempFloat;
}

HOD inline void Swap(float& F1, float& F2)
{
	const float TempFloat = F1;

	F1 = F2;
	F2 = TempFloat;
}

HOD inline void Swap(int* pI1, int* pI2)
{
	const int TempInt = *pI1;

	*pI1 = *pI2;
	*pI2 = TempInt;
}

HOD inline void Swap(int& I1, int& I2)
{
	const int TempInt = I1;

	I1 = I2;
	I2 = TempInt;

}
class CColorXyz;

HOD inline void XYZToRGB(const float xyz[3], float rgb[3]) {
	rgb[0] =  3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2];
	rgb[1] = -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2];
	rgb[2] =  0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2];
}


HOD inline void RGBToXYZ(const float rgb[3], float xyz[3]) {
	xyz[0] = 0.412453f*rgb[0] + 0.357580f*rgb[1] + 0.180423f*rgb[2];
	xyz[1] = 0.212671f*rgb[0] + 0.715160f*rgb[1] + 0.072169f*rgb[2];
	xyz[2] = 0.019334f*rgb[0] + 0.119193f*rgb[1] + 0.950227f*rgb[2];
}

static const int nSpectralSamples = 30;

static float CIE_X[nSpectralSamples] = {
	0.0253f,	0.0815f,	0.2125f,	0.3243f,	0.3461f,	0.3168f,
	0.2480f,	0.1432f,	0.0600f,	0.0160f,	0.0040f,	0.0316f,
	0.1113f,	0.2265f,	0.3604f,	0.5127f,	0.6784f,	0.8414f,
	0.9761f,	1.0523f,	1.0409f,	0.9351f,	0.7504f,	0.5429f,
	0.3625f,	0.2206f,	0.1229f,	0.0648f,	0.0336f,	0.0162f
};

static float CIE_Y[nSpectralSamples] = {
	0.0007f,	0.0023f,	0.0075f,	0.0170f,	0.0300f,	0.0483f,
	0.0744f,	0.1134f,	0.1708f,	0.2608f,	0.4091f,	0.6077f,
	0.7909f,	0.9126f,	0.9783f,	0.9983f,	0.9769f,	0.9139f,
	0.8153f,	0.6946f,	0.5668f,	0.4415f,	0.3217f,	0.2180f,
	0.1392f,	0.0824f,	0.0452f,	0.0236f,	0.0122f,	0.0059f
};

static float CIE_Z[nSpectralSamples] = {
	0.1202f,	0.3901f,	1.0293f,	1.6036f,	1.7748f,	1.7358f,
	1.5092f,	1.0446f,	0.6242f,	0.3585f,	0.2130f,	0.1140f,
	0.0582f,	0.0303f,	0.0138f,	0.0059f,	0.0028f,	0.0018f,
	0.0014f,	0.0010f,	0.0006f,	0.0002f,	0.0001f,	0.0000f,
	0.0000f,	0.0000f,	0.0000f,	0.0000f,	0.0000f,	0.0000f
};

static float CIE_lambda[nSpectralSamples] = {
	405.0000f,	415.0000f,	425.0000f,	435.0000f,	445.0000f,	455.0000f,
	465.0000f,	475.0000f,	485.0000f,	495.0000f,	505.0000f,	515.0000f,
	525.0000f,	535.0000f,	545.0000f,	555.0000f,	565.0000f,	575.0000f,
	585.0000f,	595.0000f,	605.0000f,	615.0000f,	625.0000f,	635.0000f,
	645.0000f,	655.0000f,	665.0000f,	675.0000f,	685.0000f,	695.0000f
};

static float RGB2SpectLambda[nSpectralSamples] = {
	405.0000f,	415.0000f,	425.0000f,	435.0000f,	445.0000f,	455.0000f,
	465.0000f,	475.0000f,	485.0000f,	495.0000f,	505.0000f,	515.0000f,
	525.0000f,	535.0000f,	545.0000f,	555.0000f,	565.0000f,	575.0000f,
	585.0000f,	595.0000f,	605.0000f,	615.0000f,	625.0000f,	635.0000f,
	645.0000f,	655.0000f,	665.0000f,	675.0000f,	685.0000f,	695.0000f
};

static float RGBRefl2SpectWhite[nSpectralSamples] = {
	1.0617f,	1.0622f,	1.0623f,	1.0625f,	1.0624f,	1.0625f,
	1.0625f,	1.0625f,	1.0622f,	1.0617f,	1.0612f,	1.0612f,
	1.0614f,	1.0615f,	1.0620f,	1.0625f,	1.0625f,	1.0625f,
	1.0625f,	1.0625f,	1.0625f,	1.0625f,	1.0625f,	1.0624f,
	1.0624f,	1.0623f,	1.0612f,	1.0598f,	1.0599f,	1.0602f
};

static float RGBRefl2SpectCyan[nSpectralSamples] = {
	1.0196f,	    1.0279f,	    1.0156f,	    1.0388f,	    1.0447f,	    1.0499f,
	1.0284f,	    1.0353f,	    1.0492f,	    1.0533f,	    1.0536f,	    1.0535f,
	1.0535f,	    1.0528f,	    1.0533f,	    1.0548f,	    1.0547f,	    1.0351f,
	0.7535f,	    0.3568f,	    0.0836f,	   -0.0043f,	   -0.0028f,	   -0.0059f,
   -0.0018f,	    0.0022f,	    0.0091f,	   -0.0001f,	    0.0117f,	    0.0086f
};

static float RGBRefl2SpectMagenta[nSpectralSamples] = {
	0.9870f,	    1.0012f,	    1.0177f,	    1.0176f,	    1.0192f,	    1.0025f,
	1.0064f,	    1.0146f,	    0.8028f,	    0.3308f,	    0.0053f,	    0.0053f,
	0.0022f,	   -0.0016f,	   -0.0065f,	    0.0011f,	    0.0111f,	    0.1780f,
	0.5042f,	    0.8378f,	    0.9734f,	    0.9914f,	    1.0106f,	    0.9850f,
	0.9297f,	    0.8750f,	    0.9371f,	    0.9511f,	    0.9798f,	    0.9029f
};

static float RGBRefl2SpectYellow[nSpectralSamples] = {
   -0.0056f,	   -0.0063f,	   -0.0054f,	   -0.0003f,	    0.0202f,	    0.0850f,
	0.1839f,	    0.3113f,	    0.4637f,	    0.6388f,	    0.8147f,	    0.9597f,
	1.0436f,	    1.0510f,	    1.0512f,	    1.0511f,	    1.0516f,	    1.0516f,
	1.0513f,	    1.0512f,	    1.0514f,	    1.0516f,	    1.0515f,	    1.0515f,
	1.0512f,	    1.0514f,	    1.0510f,	    1.0507f,	    1.0485f,	    1.0488f
};

static float RGBRefl2SpectRed[nSpectralSamples] = {
	0.1209f,	    0.1061f,	    0.0734f,	    0.0320f,	   -0.0019f,	    0.0114f,
	0.0090f,	    0.0106f,	    0.0024f,	   -0.0040f,	   -0.0053f,	   -0.0080f,
   -0.0051f,	   -0.0098f,	   -0.0075f,	   -0.0022f,	    0.0044f,	    0.0144f,
	0.4147f,	    0.8365f,	    0.9912f,	    0.9982f,	    0.9998f,	    0.9945f,
	1.0009f,	    1.0039f,	    0.9893f,	    1.0019f,	    0.9827f,	    0.9813f
};

static float RGBRefl2SpectGreen[nSpectralSamples] = {
   -0.0115f,	   -0.0103f,	   -0.0115f,	   -0.0084f,	   -0.0081f,	   -0.0055f,
	0.0527f,	    0.2842f,	    0.6002f,	    0.8550f,	    0.9772f,	    0.9986f,
	0.9998f,	    0.9995f,	    0.9998f,	    0.9994f,	    0.9969f,	    0.9600f,
	0.7327f,	    0.4067f,	    0.1300f,	    0.0042f,	   -0.0035f,	   -0.0051f,
   -0.0072f,	   -0.0088f,	   -0.0086f,	   -0.0084f,	   -0.0077f,	   -0.0022f
};

static float RGBRefl2SpectBlue[nSpectralSamples] = {
	0.9952f,	    0.9945f,	    0.9935f,	    0.9993f,	    0.9998f,	    0.9991f,
	0.9846f,	    0.8559f,	    0.6587f,	    0.4495f,	    0.2542f,	    0.1014f,
	0.0177f,	    0.0010f,	   -0.0004f,	   -0.0002f,	    0.0015f,	    0.0032f,
	0.0009f,	   -0.0002f,	    0.0039f,	    0.0154f,	    0.0299f,	    0.0410f,
	0.0490f,	    0.0496f,	    0.0487f,	    0.0409f,	    0.0323f,	    0.0237f
};

static float RGBIllum2SpectWhite[nSpectralSamples] = {
	1.1563f,	    1.1558f,	    1.1563f,	    1.1567f,	    1.1568f,	    1.1568f,
	1.1565f,	    1.1566f,	    1.1566f,	    1.1565f,	    1.1566f,	    1.1538f,
	1.1442f,	    1.1338f,	    1.1298f,	    1.1218f,	    1.0651f,	    1.0455f,
	1.0100f,	    0.9710f,	    0.9399f,	    0.9206f,	    0.9097f,	    0.8987f,
	0.8942f,	    0.8882f,	    0.8828f,	    0.8801f,	    0.8773f,	    0.8789f
};

static float RGBIllum2SpectCyan[nSpectralSamples] = {
	1.1349f,	    1.1357f,	    1.1357f,	    1.1361f,	    1.1362f,	    1.1364f,
	1.1358f,	    1.1361f,	    1.1362f,	    1.1360f,	    1.1358f,	    1.1357f,
	1.1361f,	    1.1356f,	    1.1353f,	    1.1328f,	    1.1039f,	    0.9485f,
	0.7023f,	    0.4212f,	    0.1927f,	    0.0501f,	   -0.0110f,	   -0.0119f,
    -0.0114f,	   -0.0109f,	   -0.0062f,	   -0.0076f,	   -0.0090f,	   -0.0067f
};

static float RGBIllum2SpectMagenta[nSpectralSamples] = {
	1.0763f,	    1.0770f,	    1.0784f,	    1.0747f,	    1.0730f,	    1.0736f,
	1.0799f,	    1.0825f,	    1.0105f,	    0.7600f,	    0.3661f,	    0.0628f,
	0.0020f,	   -0.0019f,	   -0.0011f,	   -0.0002f,	    0.0006f,	    0.0182f,
	0.1837f,	    0.4220f,	    0.7250f,	    0.9775f,	    1.0747f,	    1.0815f,
	1.0558f,	    1.0246f,	    1.0310f,	    1.0629f,	    1.0085f,	    1.0447f
};

static float RGBIllum2SpectYellow[nSpectralSamples] = {
	0.0001f,	    0.0002f,	   -0.0002f,	   -0.0001f,	   -0.0002f,	    0.0022f,
	0.0462f,	    0.3387f,	    0.7971f,	    1.0314f,	    1.0347f,	    1.0367f,
	1.0365f,	    1.0366f,	    1.0368f,	    1.0366f,	    1.0364f,	    1.0366f,
	1.0366f,	    1.0363f,	    1.0355f,	    1.0218f,	    0.9484f,	    0.8174f,
	0.7260f,	    0.6567f,	    0.6107f,	    0.5971f,	    0.5934f,	    0.5737f
};

static float RGBIllum2SpectRed[nSpectralSamples] = {
	0.0593f,	    0.0541f,	    0.0455f,	    0.0372f,	    0.0249f,	    0.0080f,
	0.0007f,	    0.0004f,	    0.0006f,	   -0.0000f,	   -0.0003f,	   -0.0001f,
   -0.0001f,	   -0.0002f,	   -0.0002f,	    0.0021f,	    0.0296f,	    0.1330f,
	0.2924f,	    0.4850f,	    0.6716f,	    0.8183f,	    0.9156f,	    0.9691f,
	0.9897f,	    0.9962f,	    0.9886f,	    0.9923f,	    0.9798f,	    0.9863f
};

static float RGBIllum2SpectGreen[nSpectralSamples] = {
	0.0070f,	    0.0055f,	    0.0007f,	   -0.0026f,	   -0.0153f,	    0.0073f,
	0.0138f,	    0.2188f,	    0.7212f,	    1.0245f,	    1.0326f,	    1.0334f,
	1.0305f,	    1.0199f,	    1.0325f,	    1.0366f,	    1.0356f,	    1.0246f,
	0.9748f,	    0.3835f,	   -0.0019f,	    0.0035f,	    0.0046f,	    0.0066f,
	0.0172f,	    0.0059f,	    0.0018f,	   -0.0001f,	   -0.0043f,	    0.0058f
};

static float RGBIllum2SpectBlue[nSpectralSamples] = {
	1.0544f,	    1.0543f,	    1.0576f,	    1.0579f,	    1.0582f,	    1.0580f,
	1.0567f,	    1.0567f,	    1.0485f,	    0.6948f,	    0.1966f,	    0.0022f,
   -0.0014f,	   -0.0014f,	   -0.0014f,	   -0.0015f,	    0.0005f,	   -0.0009f,
   -0.0014f,	   -0.0016f,	   -0.0015f,	    0.0037f,	    0.0175f,	    0.0465f,
	0.0965f,	    0.1373f,	    0.1526f,	    0.1511f,	    0.1624f,	    0.1687f
};

static const int gNoSamplesSpectrumXYZ = 3;

CD static float YWeight[gNoSamplesSpectrumXYZ] =
{
	0.212671f, 0.715160f, 0.072169f
};

class EXPOSURE_RENDER_DLL CColorXyz
{
public:
	enum EType
	{
		Reflectance,
		Illuminant
	};

	// SampledSpectrum Public Methods
	HOD CColorXyz(float v = 0.f)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i) c[i] = v;
	}

	HOD CColorXyz(float x, float y, float z)
	{
		c[0] = x;
		c[1] = y;
		c[2] = z;
	}

	HOD CColorXyz &operator+=(const CColorXyz &s2)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			c[i] += s2.c[i];
		return *this;
	}

	HOD CColorXyz operator+(const CColorXyz &s2) const
	{
		CColorXyz ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] += s2.c[i];
		return ret;
	}

	HOD CColorXyz operator-(const CColorXyz &s2) const
	{
		CColorXyz ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] -= s2.c[i];
		return ret;
	}

	HOD CColorXyz operator/(const CColorXyz &s2) const
	{
		CColorXyz ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] /= s2.c[i];
		return ret;
	}

	HOD CColorXyz operator*(const CColorXyz &sp) const
	{
		CColorXyz ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] *= sp.c[i];
		return ret;
	}

	HOD CColorXyz &operator*=(const CColorXyz &sp)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			c[i] *= sp.c[i];
		return *this;
	}

	HOD CColorXyz operator*(float a) const
	{
		CColorXyz ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] *= a;
		return ret;
	}

	HOD CColorXyz &operator*=(float a)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			c[i] *= a;
		return *this;
	}

	HOD friend inline CColorXyz operator*(float a, const CColorXyz &s)
	{
		return s * a;
	}

	HOD CColorXyz operator/(float a) const
	{
		CColorXyz ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] /= a;
		return ret;
	}

	HOD CColorXyz &operator/=(float a)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			c[i] /= a;
		return *this;
	}

	HOD bool operator==(const CColorXyz &sp) const
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			if (c[i] != sp.c[i]) return false;
		return true;
	}

	HOD bool operator!=(const CColorXyz &sp) const
	{
		return !(*this == sp);
	}

	HOD float operator[](int i) const
	{
		return c[i];
	}

	HOD float operator[](int i)
	{
		return c[i];
	}

	// ToDo: Add description
	HOD CColorXyz& CColorXyz::operator=(const CColorXyz& Other)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			c[i] = Other.c[i];

		// By convention, always return *this
		return *this;
	}

	HOD bool IsBlack() const
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			if (c[i] != 0.) return false;
		return true;
	}

	HOD CColorXyz Clamp(float low = 0, float high = INF_MAX) const
	{
		CColorXyz ret;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] = clamp2(c[i], low, high);
		return ret;
	}

	HOD float y() const
	{
		float v = 0.;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; i++)
			v += YWeight[i] * c[i];
		return v;
	}

	HOD void ToRGB(float rgb[3], float* pCIEX, float* pCIEY, float* pCIEZ) const
	{
		rgb[0] = c[0];
		rgb[1] = c[1];
		rgb[2] = c[2];

		XYZToRGB(c, rgb);
	}

	// 	RGBSpectrum ToRGBSpectrum() const;

	HOD static CColorXyz FromXYZ(float r, float g, float b)
	{
		CColorXyz L;

		L.c[0] = r;
		L.c[1] = g;
		L.c[2] = b;

		return L;
	}

	HOD static CColorXyz FromRGB(float r, float g, float b)
	{
		const float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
		const float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
		const float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

		float XYZ[3];

		XYZ[0] =	CoeffX[0] * r +
					CoeffX[1] * g +
					CoeffX[2] * b;

		XYZ[1] =	CoeffY[0] * r +
					CoeffY[1] * g +
					CoeffY[2] * b;

		XYZ[2] =	CoeffZ[0] * r +
					CoeffZ[1] * g +
					CoeffZ[2] * b;

		return CColorXyz::FromXYZ(XYZ[0], XYZ[1], XYZ[2]);
	}

	// 	static CSampledSpectrum FromXYZ(const float xyz[3], SpectrumType type = SPECTRUM_REFLECTANCE)
	// 	{
	// 		float rgb[3];
	// 		XYZToRGB(xyz, rgb);
	// 		return FromRGB(rgb, type);
	// 	}

	// 	CSampledSpectrum(const RGBSpectrum &r, SpectrumType type = SPECTRUM_REFLECTANCE);

public:
	float c[3];
};

class EXPOSURE_RENDER_DLL CSpectrumSample
{
public:
	float	m_C;
	int		m_Index;

	// ToDo: Add description
	HOD CSpectrumSample(void)
	{
		m_C		= 0.0f;
		m_Index	= 0;
	};

	// ToDo: Add description
	HOD ~CSpectrumSample(void)
	{
	}

	// ToDo: Add description
	HOD CSpectrumSample& operator=(const CSpectrumSample& Other)
	{
		m_C		= Other.m_C;
		m_Index	= Other.m_Index;

		// By convention, always return *this
		return *this;
	}
};


/*
Spectrum FromXYZ(float x, float y, float z) {
	float c[3];
	c[0] =  3.240479f * x + -1.537150f * y + -0.498535f * z;
	c[1] = -0.969256f * x +  1.875991f * y +  0.041556f * z;
	c[2] =  0.055648f * x + -0.204043f * y +  1.057311f * z;
	return Spectrum(c);
}*/

// static inline HOD SpectrumXYZ MakeSpectrum(void)																			{ SpectrumXYZ s; s.c[0] = 0.0f; s.c[1] = 0.0f; s.c[2] = 0.0f; return s;				}
// static inline HOD SpectrumXYZ MakeSpectrum(const float& r, const float& g, const float& b)									{ SpectrumXYZ s; s.c[0] = r; s.c[1] = g; s.c[2] = b; return s;							}
// static inline HOD SpectrumXYZ MakeSpectrum(const float& rgb)																{ SpectrumXYZ s; s.c[0] = rgb; s.c[1] = rgb; s.c[2] = rgb; return s;					}

class EXPOSURE_RENDER_DLL CColorXyza
{
public:
	enum EType
	{
		Reflectance,
		Illuminant
	};

	// SampledSpectrum Public Methods
	HOD CColorXyza(float v = 0.f)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i) c[i] = v;
	}

	HOD CColorXyza(float x, float y, float z)
	{
		c[0] = x;
		c[1] = y;
		c[2] = z;
	}

	HOD CColorXyza &operator+=(const CColorXyza &s2)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			c[i] += s2.c[i];
		return *this;
	}

	HOD CColorXyza operator+(const CColorXyza &s2) const
	{
		CColorXyza ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] += s2.c[i];
		return ret;
	}

	HOD CColorXyza operator-(const CColorXyza &s2) const
	{
		CColorXyza ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] -= s2.c[i];
		return ret;
	}

	HOD CColorXyza operator/(const CColorXyza &s2) const
	{
		CColorXyza ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] /= s2.c[i];
		return ret;
	}

	HOD CColorXyza operator*(const CColorXyza &sp) const
	{
		CColorXyza ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] *= sp.c[i];
		return ret;
	}

	HOD CColorXyza &operator*=(const CColorXyza &sp)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			c[i] *= sp.c[i];
		return *this;
	}

	HOD CColorXyza operator*(float a) const
	{
		CColorXyza ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] *= a;
		return ret;
	}

	HOD CColorXyza &operator*=(float a)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			c[i] *= a;
		return *this;
	}

	HOD friend inline CColorXyza operator*(float a, const CColorXyza &s)
	{
		return s * a;
	}

	HOD CColorXyza operator/(float a) const
	{
		CColorXyza ret = *this;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] /= a;
		return ret;
	}

	HOD CColorXyza &operator/=(float a)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			c[i] /= a;
		return *this;
	}

	HOD bool operator==(const CColorXyza &sp) const
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			if (c[i] != sp.c[i]) return false;
		return true;
	}

	HOD bool operator!=(const CColorXyza &sp) const
	{
		return !(*this == sp);
	}

	HOD float operator[](int i) const
	{
		return c[i];
	}

	HOD float operator[](int i)
	{
		return c[i];
	}

	// ToDo: Add description
	HOD CColorXyza& CColorXyza::operator=(const CColorXyza& Other)
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			c[i] = Other.c[i];

		// By convention, always return *this
		return *this;
	}

	HOD bool IsBlack() const
	{
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			if (c[i] != 0.) return false;
		return true;
	}

	HOD CColorXyza Clamp(float low = 0, float high = INF_MAX) const
	{
		CColorXyza ret;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
			ret.c[i] = clamp2(c[i], low, high);
		return ret;
	}

	HOD float y() const
	{
		float v = 0.;
		for (int i = 0; i < gNoSamplesSpectrumXYZ; i++)
			v += YWeight[i] * c[i];
		return v;
	}

	HOD void ToRGB(float rgb[3], float* pCIEX, float* pCIEY, float* pCIEZ) const
	{
		rgb[0] = c[0];
		rgb[1] = c[1];
		rgb[2] = c[2];

		XYZToRGB(c, rgb);
	}

	// 	RGBSpectrum ToRGBSpectrum() const;

	HOD static CColorXyza FromXYZ(float r, float g, float b)
	{
		CColorXyza L;

		L.c[0] = r;
		L.c[1] = g;
		L.c[2] = b;

		return L;
	}

	HOD static CColorXyza FromRGB(float r, float g, float b)
	{
		const float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
		const float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
		const float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

		float XYZ[3];

		XYZ[0] =	CoeffX[0] * r +
					CoeffX[1] * g +
					CoeffX[2] * b;

		XYZ[1] =	CoeffY[0] * r +
					CoeffY[1] * g +
					CoeffY[2] * b;

		XYZ[2] =	CoeffZ[0] * r +
					CoeffZ[1] * g +
					CoeffZ[2] * b;

		return CColorXyza::FromXYZ(XYZ[0], XYZ[1], XYZ[2]);
	}

	// 	static CSampledSpectrum FromXYZ(const float xyz[3], SpectrumType type = SPECTRUM_REFLECTANCE)
	// 	{
	// 		float rgb[3];
	// 		XYZToRGB(xyz, rgb);
	// 		return FromRGB(rgb, type);
	// 	}

	// 	CSampledSpectrum(const RGBSpectrum &r, SpectrumType type = SPECTRUM_REFLECTANCE);

public:
	float c[3];
};

// Colors
#define CLR_RAD_BLACK										CColorXyz(0.0f)
#define CLR_RAD_WHITE										CColorXyz(1.0f)
#define CLR_RAD_RED											CColorXyz(1.0f, 0.0f, 0.0)
#define CLR_RAD_GREEN										CColorXyz(0.0f, 1.0f, 0.0)
#define CLR_RAD_BLUE										CColorXyz(1.0f)
#define SPEC_BLACK											CColorXyz(0.0f)
#define SPEC_GRAY_10										CColorXyz(1.0f)
#define SPEC_GRAY_20										CColorXyz(1.0f)
#define SPEC_GRAY_30										CColorXyz(1.0f)
#define SPEC_GRAY_40										CColorXyz(1.0f)
#define SPEC_GRAY_50										CColorXyz(0.5f)
#define SPEC_GRAY_60										CColorXyz(1.0f)
#define SPEC_GRAY_70										CColorXyz(1.0f)
#define SPEC_GRAY_80										CColorXyz(1.0f)
#define SPEC_GRAY_90										CColorXyz(1.0f)
#define SPEC_WHITE											CColorXyz(1.0f)
#define SPEC_CYAN											CColorXyz(1.0f)
#define SPEC_RED											CColorXyz(1.0f, 0.0f, 0.0f)


#define	XYZA_BLACK			CColorXyza(0.0f)		