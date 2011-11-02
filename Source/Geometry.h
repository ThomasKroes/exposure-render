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

#include <algorithm>
#include <math.h>

using namespace std;

class CColorRgbHdr;
class CColorRgbLdr;
class Vec2i;
class Vec2f;
class Vec3i;
class Vec3f;
class Vec4i;
class Vec4f;

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

class EXPOSURE_RENDER_DLL Vec2f
{
public:
	HOD Vec2f(void)
	{
		this->x = 0.0f;
		this->y = 0.0f;
	}

	HOD Vec2f(const float& x, const float& y)
	{
		this->x = x;
		this->y = y;
	}

	HOD Vec2f(const float& xy)
	{
		this->x = xy;
		this->y = xy;
	}

	HOD Vec2f(const Vec2f& v)
	{
		this->x = v.x;
		this->y = v.y;
	}

	HOD float operator[](int i) const
	{
		return (&x)[i];
	}

	HOD float& operator[](int i)
	{
		return (&x)[i];
	}

	HOD Vec2f& operator = (const Vec2f& v)
	{
		x = v.x; 
		y = v.y; 

		return *this;
	}

	HOD Vec2f operator + (const Vec2f& v) const
	{
		return Vec2f(x + v.x, y + v.y);
	}

	HOD Vec2f& operator += (const Vec2f& v)
	{
		x += v.x; y += v.y;
		return *this;
	}

	HOD Vec2f operator - (const Vec2f& v) const
	{
		return Vec2f(x - v.x, y - v.y);
	}

	HOD Vec2f& operator -= (const Vec2f& v)
	{
		x -= v.x; y -= v.y;
		return *this;
	}

	HOD Vec2f operator * (float f) const
	{
		return Vec2f(x * f, y * f);
	}

	HOD Vec2f& operator *= (float f)
	{
		x *= f; 
		y *= f; 

		return *this;
	}

	HOD bool operator < (const Vec2f& V) const
	{
		return V.x < x && V.y < y;
	}

	HOD bool operator > (const Vec2f& V) const
	{
		return V.x > x && V.y > y;
	}

	HOD bool operator == (const Vec2f& V) const
	{
		return V.x == x && V.y == y;
	}

	HOD float LengthSquared(void) const
	{
		return x * x + y * y;
	}

	HOD float Length(void) const
	{
		return sqrtf(LengthSquared());
	}

	void PrintSelf(void)
	{
		printf("[%g, %g]\n", x, y);
	}

	float x, y;
};

class EXPOSURE_RENDER_DLL  Vec2i
{
public:
	HOD Vec2i(void)
	{
		this->x = 0;
		this->y = 0;
	}

	HOD Vec2i(const int& x, const int& y)
	{
		this->x = x;
		this->y = y;
	}

	HOD Vec2i(int& x, int& y)
	{
		this->x = x;
		this->y = y;
	}

	HOD Vec2i(const Vec2f& V)
	{
		this->x = (int)V.x;
		this->y = (int)V.y;
	}

	HOD Vec2i(const int& xy)
	{
		this->x = xy;
		this->y = xy;
	}

	HOD int operator[](int i) const
	{
		return (&x)[i];
	}

	HOD int& operator[](int i)
	{
		return (&x)[i];
	}

	HOD Vec2i& operator = (const Vec2i& v)
	{
		x = v.x; 
		y = v.y; 

		return *this;
	}

	HOD bool operator < (const Vec2i& V) const
	{
		return V.x < x && V.y < y;
	}

	HOD bool operator > (const Vec2i& V) const
	{
		return V.x > x && V.y > y;
	}

	HOD bool operator == (const Vec2i& V) const
	{
		return V.x == x && V.y == y;
	}

	void PrintSelf(void)
	{
		printf("[%d, %d]\n", x, y);
	}

	int x, y;
};

class EXPOSURE_RENDER_DLL Vec3i
{
public:
	HOD Vec3i(void)
	{
		this->x = 0;
		this->y = 0;
		this->z = 0;
	}

	HOD Vec3i(const int& x, const int& y, const int& z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	HOD Vec3i(const int& xyz)
	{
		this->x = xyz;
		this->y = xyz;
		this->z = xyz;
	}

	HOD int operator[](int i) const
	{
		return (&x)[i];
	}

	HOD int& operator[](int i)
	{
		return (&x)[i];
	}

	HOD Vec3i& operator = (const Vec3i &v)
	{
		x = v.x; 
		y = v.y; 
		z = v.z;

		return *this;
	}

	HOD bool operator < (const Vec3i& V) const
	{
		return V.x < x && V.y < y && V.z < z;
	}

	HOD bool operator > (const Vec3i& V) const
	{
		return V.x > x && V.y > y && V.z > z;
	}

	HOD bool operator == (const Vec3i& V) const
	{
		return V.x == x && V.y == y && V.z == z;
	}

	HOD float LengthSquared(void) const
	{
		return x * x + y * y;
	}

	HOD float Length(void) const
	{
		return sqrtf(LengthSquared());
	}

	HOD int Max(void)
	{
		if (x >= y && x >= z)
		{
			return x;
		}		
		else
		{
			if (y >= x && y >= z)
				return y;
			else
				return z;
		}
	}

	HOD int Min(void)
	{
		if (x <= y && x <= z)
		{
			return x;
		}		
		else
		{
			if (y <= x && y <= z)
				return y;
			else
				return z;
		}
	}

	void PrintSelf(void)
	{
		printf("[%d, %d, %d]\n", x, y, z);
	}

	int x, y, z;
};

class EXPOSURE_RENDER_DLL Vec3f
{
public:
#if defined __CUDACC__

	HOD Vec3f(void)
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}

	HOD Vec3f(const Vec3f& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
	}

	HOD Vec3f(const float& x, const float& y, const float& z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	HOD Vec3f(const float& xyz)
	{
		this->x = xyz;
		this->y = xyz;
		this->z = xyz;
	}
#endif
	HOD float operator[](int i) const
	{
		return (&x)[i];
	}

	HOD float& operator[](int i)
	{
		return (&x)[i];
	}

	HOD Vec3f& operator = (const Vec3f &v)
	{
		x = v.x; 
		y = v.y; 
		z = v.z;

		return *this;
	}

	HOD Vec3f operator + (const Vec3f& v) const
	{
		return Vec3f(x + v.x, y + v.y, z + v.z);
	}

	HOD Vec3f& operator += (const Vec3f& v)
	{
		x += v.x; y += v.y; z += v.z;
		return *this;
	}

	HOD Vec3f operator - (const Vec3f& v) const
	{
		return Vec3f(x - v.x, y - v.y, z - v.z);
	}

	HOD Vec3f& operator -= (const Vec3f& v)
	{
		x -= v.x; y -= v.y; z -= v.z;
		return *this;
	}

	HOD Vec3f operator * (float f) const
	{
		return Vec3f(x * f, y * f, z * f);
	}

	HOD Vec3f& operator *= (float f)
	{
		x *= f; 
		y *= f; 
		z *= f;

		return *this;
	}

	HOD Vec3f operator / (float f) const
	{
		float inv = 1.f / f;
		return Vec3f(x * inv, y * inv, z * inv);
	}

	HOD Vec3f& operator /= (float f)
	{
		float inv = 1.f / f;
		x *= inv; y *= inv; z *= inv;
		return *this;
	}

	HOD Vec3f operator / (Vec3i V)
	{
		return Vec3f(x / (float)V.x, y / (float)V.y, z / (float)V.z);
	}

	HOD Vec3f& operator /= (Vec3i V)
	{
		x /= (float)V.x;
		y /= (float)V.y;
		z /= (float)V.z;

		return *this;
	}

	HOD Vec3f operator-() const
	{
		return Vec3f(-x, -y, -z);
	}

	HOD bool operator < (const Vec3f& V) const
	{
		return V.x < x && V.y < y && V.z < z;
	}

	HOD bool operator > (const Vec3f& V) const
	{
		return V.x > x && V.y > y && V.z > z;
	}

	HOD bool operator == (const Vec3f& V) const
	{
		return V.x == x && V.y == y && V.z == z;
	}

	HOD float Max(void) const
	{
		if (x >= y && x >= z)
		{
			return x;
		}		
		else
		{
			if (y >= x && y >= z)
				return y;
			else
				return z;
		}
	}

	HOD float Min(void) const
	{
		if (x <= y && x <= z)
		{
			return x;
		}		
		else
		{
			if (y <= x && y <= z)
				return y;
			else
				return z;
		}
	}

	HOD float LengthSquared(void) const
	{
		return x * x + y * y + z * z;
	}

	HOD float Length(void) const
	{
		return sqrtf(LengthSquared());
	}

	HOD void Normalize(void)
	{
		const float L = Length();
		x /= L;
		y /= L;
		z /= L;
	}

	HOD float Dot(const Vec3f& rhs) const
	{
		return (x * rhs.x + y * rhs.y + z * rhs.z);
	}

	HOD Vec3f Cross(const Vec3f& rhs) const
	{
		return Vec3f(	(y * rhs.z) - (z * rhs.y),
							(z * rhs.x) - (x * rhs.z),
							(x * rhs.y) - (y * rhs.x));
	}

	HOD void ScaleBy(const float& factor)
	{
		x *= factor;
		y *= factor;
		z *= factor;
	}

	HOD void RotateAxis(const Vec3f& axis, const float degrees)
	{
		RadianRotateAxis(axis, degrees * DEG_TO_RAD);
	}

	HOD void RadianRotateAxis(const Vec3f& axis, const float radians)
	{
		// Formula goes CW around axis. I prefer to think in terms of CCW
		// rotations, to be consistant with the other rotation methods.
		float cosAngle = cos(-radians);
		float sinAngle = sin(-radians);

		Vec3f w = axis;
		w.Normalize();
		float vDotW = this->Dot(w);
		Vec3f vCrossW = this->Cross(w);
		w.ScaleBy(vDotW); // w * (v . w)

		x = w.x + (this->x - w.x) * cosAngle + vCrossW.x * sinAngle;
		y = w.y + (this->y - w.y) * cosAngle + vCrossW.y * sinAngle;
		z = w.z + (this->z - w.z) * cosAngle + vCrossW.z * sinAngle;
	}

	HOD float NormLengthSquared(void)
	{
		float vl = x * x + y * y + z * z;
		
		if (vl != 0.0f)
		{
			const float d = 1.0f / sqrt(vl);
			x *= d; y *= d; z *= d;
		}

		return vl;
	}

	void PrintSelf(void)
	{
		printf("[%g, %g, %g]\n", x, y, z);
	}

	float x, y, z;
};

class EXPOSURE_RENDER_DLL Vec4i
{
public:
	HOD Vec4i(void)
	{
		this->x = 0;
		this->y = 0;
		this->z = 0;
		this->w = 0;
	}

	HOD Vec4i(const int& x, const int& y, const int& z, const int& w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	HOD Vec4i(const int& xyzw)
	{
		this->x = xyzw;
		this->y = xyzw;
		this->z = xyzw;
		this->w = xyzw;
	}

	int x, y, z, w;
};

class EXPOSURE_RENDER_DLL Vec4f
{
public:
	HOD Vec4f(void)
	{
		this->x = 0.0f;
		this->y = 0.0f;
		this->z = 0.0f;
		this->w = 0.0f;
	}

	HOD Vec4f(const float& x, const float& y, const float& z, const float& w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	HOD Vec4f operator * (float f) const
	{
		return Vec4f(x * f, y * f, z * f, w * f);
	}

	HOD Vec4f& operator *= (float f)
	{
		x *= f; 
		y *= f; 
		z *= f;
		w *= f;

		return *this;
	}

	void PrintSelf(void)
	{
		printf("[%d, %d, %d, %d]\n", x, y, z, w);
	}

	float x, y, z, w;
};

HOD inline Vec3f Normalize(const Vec3f& v)
{
	return v / v.Length();
}

// Vec2f
static inline HOD Vec2f operator * (const Vec2f& v, const float& f) 	{ return Vec2f(f * v.x, f * v.y);					};
static inline HOD Vec2f operator * (const float& f, const Vec2f& v) 	{ return Vec2f(f * v.x, f * v.y);					};
static inline HOD Vec2f operator * (const Vec2f& v1, const Vec2f& v2) 	{ return Vec2f(v1.x * v2.x, v1.y * v2.y);			};
static inline HOD Vec2f operator / (const Vec2f& v1, const Vec2f& v2) 	{ return Vec2f(v1.x / v2.x, v1.y / v2.y);			};
// static inline HOD Vec2f operator - (const Vec2f& v1, const Vec2f& v2) 	{ return Vec2f(v1.x - v2.x, v1.y - v2.y);			};

// Vec3f
static inline HOD Vec3f operator * (const float& f, const Vec3f& v) 		{ return Vec3f(f * v.x, f * v.y, f * v.z);						};
static inline HOD Vec3f operator * (const Vec3f& v1, const Vec3f& v2) 		{ return Vec3f(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);			};
static inline HOD Vec3f operator / (const Vec3f& v1, const Vec3f& v2) 		{ return Vec3f(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);			};


static inline HOD Vec2f operator * (Vec2f& V2f, Vec2i& V2i)	{ return Vec2f(V2f.x * V2i.x, V2f.y * V2i.y);				};

class EXPOSURE_RENDER_DLL CRay
{	
public:
	// ToDo: Add description
	HOD CRay(void)
	{
	};

	// ToDo: Add description
	HOD CRay(Vec3f Origin, Vec3f Dir, float MinT, float MaxT = INF_MAX, int PixelID = 0)
	{
		m_O			= Origin;
		m_D			= Dir;
		m_MinT		= MinT;
		m_MaxT		= MaxT;
		m_PixelID	= PixelID;
	}

	// ToDo: Add description
	HOD ~CRay(void)
	{
	}

	// ToDo: Add description
	HOD CRay& operator=(const CRay& Other)
	{
		m_O			= Other.m_O;
		m_D			= Other.m_D;
		m_MinT		= Other.m_MinT;
		m_MaxT		= Other.m_MaxT;
		m_PixelID	= Other.m_PixelID;

		// By convention, always return *this
		return *this;
	}

	// ToDo: Add description
	HOD Vec3f operator()(float t) const
	{
		return m_O + Normalize(m_D) * t;
	}

	void PrintSelf(void)
	{
		printf("Origin ");
		m_O.PrintSelf();

		printf("Direction ");
		m_D.PrintSelf();

		printf("Min T: %4.2f\n", m_MinT);
		printf("Max T: %4.2f\n", m_MaxT);
		printf("Pixel ID: %d\n", m_PixelID);
	}

	Vec3f 	m_O;			/*!< Ray origin */
	Vec3f 	m_D;			/*!< Ray direction */
	float	m_MinT;			/*!< Minimum parametric range */
	float	m_MaxT;			/*!< Maximum parametric range */
	int		m_PixelID;		/*!< Pixel ID associated with the ray */
};

class EXPOSURE_RENDER_DLL CSize2D
{
public:
	Vec2f	m_Size;
	Vec2f	m_InvSize;

	HOD CSize2D(void) :
		m_Size(1.0f, 1.0f),
		m_InvSize(1.0f / m_Size.x, 1.0f / m_Size.y)
	{
	};

	HOD CSize2D(const float& X, const float& Y) :
		m_Size(X, Y),
		m_InvSize(1.0f / m_Size.x, 1.0f / m_Size.y)
	{
	};

	HOD CSize2D(const Vec2f& V) :
		m_Size(V),
		m_InvSize(1.0f / m_Size.x, 1.0f / m_Size.y)
	{
	};

	// ToDo: Add description
	HOD CSize2D& operator=(const CSize2D& Other)
	{
		m_Size		= Other.m_Size;
		m_InvSize	= Other.m_InvSize;

		return *this;
	}

	HOD void Update(void)
	{
		m_InvSize = Vec2f(1.0f / m_Size.x, 1.0f / m_Size.y);
	}
};

class EXPOSURE_RENDER_DLL CSize3D
{
public:
	Vec3f	m_Size;
	Vec3f	m_InvSize;

	HOD CSize3D(void) :
		m_Size(1.0f, 1.0f, 1.0f),
		m_InvSize(1.0f / m_Size.x, 1.0f / m_Size.y, 1.0f / m_Size.z)
	{
	};

	HOD CSize3D(const float& X, const float& Y, const float& Z) :
		m_Size(X, Y, Z),
		m_InvSize(1.0f / m_Size.x, 1.0f / m_Size.y, 1.0f / m_Size.z)
	{
	};

	HOD CSize3D(const Vec3f& V) :
		m_Size(V),
		m_InvSize(1.0f / m_Size.x, 1.0f / m_Size.y, 1.0f / m_Size.z)
	{
	};

	// ToDo: Add description
	HOD CSize3D& operator=(const CSize3D& Other)
	{
		m_Size		= Other.m_Size;
		m_InvSize	= Other.m_InvSize;

		return *this;
	}

	HOD void Update(void)
	{
		m_InvSize = Vec3f(1.0f / m_Size.x, 1.0f / m_Size.y, 1.0f / m_Size.z);
	}
};

class EXPOSURE_RENDER_DLL CRange
{
public:
	HOD CRange(void) :
		m_Min(0.0f),
		m_Max(100.0f),
		m_Range(m_Max - m_Min),
		m_InvRange(1.0f / m_Range)
	{
	}

	HOD CRange(const float& MinRange, const float& MaxRange) :
		m_Min(MinRange),
		m_Max(MaxRange),
		m_Range(m_Max - m_Min),
		m_InvRange(1.0f / m_Range)
	{
	}

	HOD CRange& operator = (const CRange& Other)
	{
		m_Min		= Other.m_Min; 
		m_Max		= Other.m_Max; 
		m_Range		= Other.m_Range;
		m_InvRange	= Other.m_InvRange;

		return *this;
	}

	HOD float	GetMin(void) const			{ return m_Min;								}
	HOD void	SetMin(const float& Min)	{ m_Min = Min; m_Range = m_Max - m_Min;		}
	HOD float	GetMax(void) const			{ return m_Max;								}
	HOD void	SetMax(const float& Max)	{ m_Max = Max; m_Range = m_Max - m_Min;		}
	HOD float	GetRange(void) const		{ return m_Range;							}
	HOD float	GetInvRange(void) const		{ return m_InvRange;						}

	void PrintSelf(void)
	{
		printf("[%4.2f - %4.2f]\n", m_Min, m_Max);
	}

private:
	float	m_Min;
	float	m_Max;
	float	m_Range;
	float	m_InvRange;
};

class EXPOSURE_RENDER_DLL CBoundingBox
{
public:
	Vec3f	m_MinP;
	Vec3f	m_MaxP;
		
	CBoundingBox(void) :
		m_MinP(FLT_MAX, FLT_MAX, FLT_MAX),
		m_MaxP(-FLT_MAX, -FLT_MAX, -FLT_MAX)
	{
	};
 
	HOD CBoundingBox(const Vec3f &v1, const Vec3f &v2) :
		m_MinP(v1),
		m_MaxP(v2)
	{
	}
	
	HOD CBoundingBox& operator = (const CBoundingBox& B)
	{
		m_MinP = B.m_MinP; 
		m_MaxP = B.m_MaxP; 

		return *this;
	}

	// Adds a point to this bounding box
	CBoundingBox& operator += (const Vec3f& P)
	{
		if (!Contains(P))
		{
			for (int i = 0; i < 3; i++)
			{
				if (P[i] < m_MinP[i])
					m_MinP[i] = P[i];

				if (P[i] > m_MaxP[i])
					m_MaxP[i] = P[i];
			}
		}

		return *this;
	}

	// Adds a bounding box to this bounding box
	CBoundingBox& operator += (const CBoundingBox& B)
	{
		*this += B.m_MinP;
		*this += B.m_MaxP;

		return *this;
	}

	Vec3f &operator[](int i)
	{
		return (&m_MinP)[i];
	}

	const Vec3f &operator[](int i) const
	{
		return (&m_MinP)[i];
	}

	HOD float LengthX(void) const	{ return fabs(m_MaxP.x - m_MinP.x); };
	HOD float LengthY(void) const	{ return fabs(m_MaxP.y - m_MinP.y); };
	HOD float LengthZ(void) const	{ return fabs(m_MaxP.z - m_MinP.z); };
	
	HOD Vec3f GetCenter(void) const
	{
		return Vec3f(0.5f * (m_MinP.x + m_MaxP.x), 0.5f * (m_MinP.y + m_MaxP.y), 0.5f * (m_MinP.z + m_MaxP.z));
	}

	HOD EContainment Contains(const Vec3f& P) const
	{
		for (int i = 0; i < 3; i++)
		{
			if (P[i] < m_MinP[i] || P[i] > m_MaxP[i])
				return ContainmentNone;
		}

		return ContainmentFull;
	};

	HOD EContainment Contains(const Vec3f* pPoints, long PointCount) const
	{
		long Contain = 0;

		for (int i = 0; i < PointCount; i++)
		{
			if (Contains(pPoints[i]) == ContainmentFull)
				Contain++;
		}

		if (Contain == 0)
			return ContainmentNone;
		else
		{
			if (Contain == PointCount)
				return ContainmentFull;
			else
				return ContainmentPartial;
		}
	}

	HOD EContainment Contains(const CBoundingBox& B) const
	{
		bool ContainsMin = false, ContainsMax = false;

		if (Contains(B.m_MinP) == ContainmentFull)
			ContainsMin = true;

		if (Contains(B.m_MaxP) == ContainmentFull)
			ContainsMax = true;

		if (!ContainsMin && !ContainsMax)
			return ContainmentNone;
		else
		{
			if (ContainsMin && ContainsMax)
				return ContainmentFull;
			else
				return ContainmentPartial;
		}
	}

	HOD EAxis GetDominantAxis(void) const
	{
		return (LengthX() > LengthY() && LengthX() > LengthZ()) ? AxisX : ((LengthY() > LengthZ()) ? AxisY : AxisZ);
	}

	HOD	Vec3f				GetMinP(void) const		{ return m_MinP;				}
	HOD	Vec3f				GetInvMinP(void) const	{ return Vec3f(1.0f) / m_MinP;	}
	HOD	void				SetMinP(Vec3f MinP)		{ m_MinP = MinP;				}
	HOD	Vec3f				GetMaxP(void) const		{ return m_MaxP;				}
	HOD	Vec3f				GetInvMaxP(void) const	{ return Vec3f(1.0f) / m_MaxP;	}
	HOD	void				SetMaxP(Vec3f MaxP)		{ m_MaxP = MaxP;				}

	HO float GetMaxLength(EAxis* pAxis = NULL) const
	{
		if (pAxis)
			*pAxis = GetDominantAxis();

		const Vec3f& MinMax = GetExtent();

		return MinMax[GetDominantAxis()];
	}

	HO float HalfSurfaceArea(void) const
	{
		const Vec3f e(GetExtent());
		return e.x * e.y + e.y * e.z + e.x * e.z;
	}

	HO float GetArea(void) const
	{
		const Vec3f ext(m_MaxP-m_MinP);
		return float(ext.x)*float(ext.y) + float(ext.y)*float(ext.z) + float(ext.x)*float(ext.z);
	}

	HO Vec3f GetExtent(void) const
	{
		return m_MaxP - m_MinP;
	}

	HO float GetEquivalentRadius(void) const
	{
		return 0.5f * GetExtent().Length();
	}

	HOD bool Inside(const Vec3f& pt)
	{
		return (pt.x >= m_MinP.x && pt.x <= m_MaxP.x &&
				pt.y >= m_MinP.y && pt.y <= m_MaxP.y &&
				pt.z >= m_MinP.z && pt.z <= m_MaxP.z);
	}

	// Performs a line box intersection
	HOD bool Intersect(CRay& R, float* pMinT = NULL, float* pMaxT = NULL)
	{
		// Compute intersection of line with all six bounding box planes
		const Vec3f InvR = Vec3f(1.0f / R.m_D.x, 1.0f / R.m_D.y, 1.0f / R.m_D.z);
		const Vec3f BotT = InvR * (m_MinP - R.m_O);
		const Vec3f TopT = InvR * (m_MaxP - R.m_O);

		// re-order intersections to find smallest and largest on each axis
		const Vec3f MinT = Vec3f(min(TopT.x, BotT.x), min(TopT.y, BotT.y), min(TopT.z, BotT.z));
		const Vec3f MaxT = Vec3f(max(TopT.x, BotT.x), max(TopT.y, BotT.y), max(TopT.z, BotT.z));

		// find the largest tmin and the smallest tmax
		const float LargestMinT		= max(max(MinT.x, MinT.y), max(MinT.x, MinT.z));
		const float SmallestMaxT	= min(min(MaxT.x, MaxT.y), min(MaxT.x, MaxT.z));

		if (pMinT)
			*pMinT = LargestMinT;

		if (pMaxT)
			*pMaxT = SmallestMaxT;

		return SmallestMaxT > LargestMinT;
	}

	HOD bool IntersectP(const CRay& ray, float* hitt0 = NULL, float* hitt1 = NULL)
	{
		float t0 = ray.m_MinT, t1 = ray.m_MaxT;
		
		for (int i = 0; i < 3; ++i)
		{
			// Update interval for _i_th bounding box slab
			float invRayDir = 1.f / ray.m_D[i];
			float tNear = (m_MinP[i] - ray.m_O[i]) * invRayDir;
			float tFar  = (m_MaxP[i] - ray.m_O[i]) * invRayDir;

			// Update parametric interval from slab intersection $t$s
			if (tNear > tFar)
				swap(tNear, tFar);

			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar  < t1 ? tFar  : t1;

			if (t0 > t1)
				return false;
		}

		if (hitt0)
			*hitt0 = t0;

		if (hitt1)
			*hitt1 = t1;

		return true;
	}

	void PrintSelf(void)
	{
		printf("Min: ");
		m_MinP.PrintSelf();

		printf("Max: ");
		m_MaxP.PrintSelf();
	}
};

class EXPOSURE_RENDER_DLL CPixel
{
public:
	HOD CPixel(void)
	{
		m_XY	= Vec2i(256);
		m_ID	= 0;
	}

	HOD CPixel(const Vec2f& ImageXY, const Vec2i& Resolution)
	{
		m_XY	= Vec2i(floorf(ImageXY.x), floorf(ImageXY.y));
		m_ID	= (m_XY.y * Resolution.x) + m_XY.x;
	}

	HOD CPixel& operator = (const CPixel& Other)
	{
		m_XY	= Other.m_XY; 
		m_ID	= Other.m_ID;

		return *this;
	}

	Vec2i	m_XY;		/*!< Pixel coordinates */
	int		m_ID;		/*!< Pixel ID */
};

class EXPOSURE_RENDER_DLL CResolution2D
{
public:
	// ToDo: Add description
	HOD CResolution2D(const float& Width, const float& Height)
	{
		m_XY		= Vec2i(Width, Height);

		Update();
	}

	// ToDo: Add description
	HOD CResolution2D(void)
	{
		m_XY		= Vec2i(640, 480);

		Update();
	}

	// ToDo: Add description
	HOD ~CResolution2D(void)
	{
	}

	// ToDo: Add description
	HOD CResolution2D& CResolution2D::operator=(const CResolution2D& Other)
	{
		m_XY				= Other.m_XY;
		m_InvXY				= Other.m_InvXY;
		m_NoElements		= Other.m_NoElements;
		m_AspectRatio		= Other.m_AspectRatio;
		m_DiagonalLength	= Other.m_DiagonalLength;

		return *this;
	}

	HOD int operator[](int i) const
	{
		return m_XY[i];
	}

	HOD int& operator[](int i)
	{
		return m_XY[i];
	}

	HOD bool operator == (const CResolution2D& Other) const
	{
		return GetResX() == Other.GetResX() && GetResY() == Other.GetResY();
	}

	HOD bool operator != (const CResolution2D& Other) const
	{
		return GetResX() != Other.GetResX() || GetResY() != Other.GetResY();
	}

	// ToDo: Add description
	HOD void Update(void)
	{
		m_InvXY				= Vec2f(1.0f / m_XY.x, 1.0f / m_XY.y);
		m_NoElements		= m_XY.x * m_XY.y;
		m_AspectRatio		= (float)m_XY.y / (float)m_XY.x;
		m_DiagonalLength	= sqrtf(powf(m_XY.x, 2.0f) + powf(m_XY.y, 2.0f));
	}

	// ToDo: Add description
	HOD Vec2i ToVector(void) const
	{
		return Vec2i(m_XY.x, m_XY.y);
	}

	HOD void Set(const Vec2i& Resolution)
	{
		m_XY		= Resolution;

		Update();
	}

	HOD int		GetResX(void) const				{ return m_XY.x; }
	HOD void	SetResX(const int& Width)		{ m_XY.x = Width; Update(); }
	HOD int		GetResY(void) const				{ return m_XY.y; }
	HOD void	SetResY(const int& Height)		{ m_XY.y = Height; Update(); }
	HOD Vec2f	GetInv(void) const				{ return m_InvXY; }
	HOD int		GetNoElements(void) const		{ return m_NoElements; }
	HOD float	GetAspectRatio(void) const		{ return m_AspectRatio; }

	void PrintSelf(void)
	{
		printf("[%d x %d]\n", GetResX(), GetResY());
	}
	
private:
	Vec2i	m_XY;					/*!< Resolution width and height */
	Vec2f	m_InvXY;				/*!< Resolution width and height reciprocal */
	int		m_NoElements;			/*!< No. elements */
	float	m_AspectRatio;			/*!< Aspect ratio of image plane */
	float	m_DiagonalLength;		/*!< Diagonal length */
};

class EXPOSURE_RENDER_DLL CResolution3D
{
public:
	// ToDo: Add description
	HOD CResolution3D(void)
	{
		SetResX(0);
		SetResY(0);
		SetResZ(0);
	}

	// ToDo: Add description
	HOD CResolution3D& CResolution3D::operator=(const CResolution3D& Other)
	{
		m_XYZ				= Other.m_XYZ;
		m_InvXYZ			= Other.m_InvXYZ;
		m_NoElements		= Other.m_NoElements;
		m_DiagonalLength	= Other.m_DiagonalLength;
		m_Dirty				= Other.m_Dirty;

		return *this;
	}

	HOD int operator[](int i) const
	{
		return m_XYZ[i];
	}

	HOD int& operator[](int i)
	{
		return m_XYZ[i];
	}

	HOD void Update(void)
	{
		m_InvXYZ.x			= m_XYZ.x == 0.0f ? 1.0f : 1.0f / (float)m_XYZ.x;
		m_InvXYZ.y			= m_XYZ.y == 0.0f ? 1.0f : 1.0f / (float)m_XYZ.y;
		m_InvXYZ.z			= m_XYZ.z == 0.0f ? 1.0f : 1.0f / (float)m_XYZ.z;
		m_NoElements		= m_XYZ.x * m_XYZ.y * m_XYZ.z;
		m_DiagonalLength	= m_XYZ.Length();
	}

	HOD Vec3f ToVector3(void) const
	{
		return Vec3f(m_XYZ.x, m_XYZ.y, m_XYZ.z);
	}

	HOD void SetResXYZ(const Vec3i& Resolution)
	{
		m_Dirty	= m_XYZ.x != Resolution.x || m_XYZ.y != Resolution.y || m_XYZ.z != Resolution.z;
		m_XYZ	= Resolution;

		Update();
	}

	HOD Vec3i	GetResXYZ(void) const				{ return m_XYZ; }
	HOD int		GetResX(void) const					{ return m_XYZ.x; }
	HOD void	SetResX(const int& ResX)			{ m_Dirty = m_XYZ.x != ResX; m_XYZ.x = ResX; Update(); }
	HOD int		GetResY(void) const					{ return m_XYZ.y; }
	HOD void	SetResY(const int& ResY)			{ m_Dirty = m_XYZ.y != ResY; m_XYZ.y = ResY; Update(); }
	HOD int		GetResZ(void) const					{ return m_XYZ.z; }
	HOD void	SetResZ(const int& ResZ)			{ m_Dirty = m_XYZ.z != ResZ; m_XYZ.z = ResZ; Update(); }
	HOD Vec3f	GetInv(void) const					{ return m_InvXYZ; }
	HOD int		GetNoElements(void) const			{ return m_NoElements; }
	HOD int		GetMin(void) const					{ return min(GetResX(), min(GetResY(), GetResZ()));		}
	HOD int		GetMax(void) const					{ return max(GetResX(), max(GetResY(), GetResZ()));		}

	HO void PrintSelf(void)
	{
		printf("[%d x %d x %d], %d elements\n", GetResX(), GetResY(), GetResZ(), GetNoElements());
	}

private:
	Vec3i	m_XYZ;
	Vec3f	m_InvXYZ;
	int		m_NoElements;
	float	m_DiagonalLength;
	bool	m_Dirty;
};

HOD inline float Dot(const Vec3f& a, const Vec3f& b)			
{
	return a.x * b.x + a.y * b.y + a.z * b.z;			
};

HOD inline float AbsDot(const Vec3f& a, const Vec3f& b)			
{
	return fabsf(Dot(a, b));			
};

HOD inline Vec3f Cross(const Vec3f &v1, const Vec3f &v2)		
{
	return Vec3f((v1.y * v2.z) - (v1.z * v2.y), (v1.z * v2.x) - (v1.x * v2.z), (v1.x * v2.y) - (v1.y * v2.x)); 
};

// reflect
HOD inline Vec3f Reflect(Vec3f i, Vec3f n)
{
	return i - 2.0f * n * Dot(n, i);
}

// reflect
HOD inline float Length(const Vec3f& v)
{
	return v.Length();
}

inline HOD float Fmaxf(float a, float b)
{
	return a > b ? a : b;
}

inline HOD float Fminf(float a, float b)
{
	return a < b ? a : b;
}

inline HOD float Clamp(float f, float a, float b)
{
	return Fmaxf(a, Fminf(f, b));
}

// clamp
inline HOD Vec3f Clamp(Vec3f v, float a, float b)
{
	return Vec3f(Clamp(v.x, a, b), Clamp(v.y, a, b), Clamp(v.z, a, b));
}

inline HOD Vec3f Clamp(Vec3f v, Vec3f a, Vec3f b)
{
	return Vec3f(Clamp(v.x, a.x, b.x), Clamp(v.y, a.y, b.y), Clamp(v.z, a.z, b.z));
}

// floor
HOD inline Vec3f Floor(const Vec3f v)
{
	return Vec3f(floor(v.x), floor(v.y), floor(v.z));
}

HOD inline float Distance(Vec3f p1, Vec3f p2)
{
	return (p1 - p2).Length();
}

HOD inline float DistanceSquared(Vec3f p1, Vec3f p2)
{
	return (p1 - p2).LengthSquared();
}

HOD inline Vec3f Reflect(Vec3f& i, Vec3f& n)
{
	return i - 2.0f * n * Dot(n, i);
}

HOD inline void CreateCS(const Vec3f& N, Vec3f& u, Vec3f& v)
{
	if ((N.x == 0) && (N.y == 0))
	{
		if (N.z < 0.0f)
			u = Vec3f(-1.0f, 0.0f, 0.0f);
		else
			u = Vec3f(1.0f, 0.0f, 0.0f);
		
		v = Vec3f(0.0f, 1.0f, 0.0f);
	}
	else
	{
		// Note: The root cannot become zero if
		// N.x == 0 && N.y == 0.
		const float d = 1.0f / sqrtf(N.y*N.y + N.x*N.x);
		
		u = Vec3f(N.y * d, -N.x * d, 0);
		v = Cross(N, u);
	}
}

// Computes the bary-center of a triangle
inline void ComputeTriangleBaryCenter(const Vec3f P[3], Vec3f* pC)
{
	if (!pC)
		return;

	const Vec3f Edge[2] =
	{
		Vec3f(P[1] - P[0]),
		Vec3f(P[2] - P[0])
	};

	*pC = P[0] + (Edge[0]  * 0.33333f) + (Edge[1] * 0.33333f);
}

// Computes the area of a triangle
inline void ComputeTriangleArea(const Vec3f P[3], float* pArea)
{
	if (!pArea)
		return;

	*pArea = 0.5f * Cross(P[1] - P[0], P[2] - P[0]).Length();
}

HOD inline void CoordinateSystem(const Vec3f& v1, Vec3f* v2, Vec3f *v3)
{
	if (fabsf(v1.x) > fabsf(v1.y))
	{
		float invLen = 1.f / sqrtf(v1.x*v1.x + v1.z*v1.z);
		*v2 = Vec3f(-v1.z * invLen, 0.f, v1.x * invLen);
	}
	else {
		float invLen = 1.f / sqrtf(v1.y*v1.y + v1.z*v1.z);
		*v2 = Vec3f(0.f, v1.z * invLen, -v1.y * invLen);
	}
	*v3 = Cross(v1, *v2);
}

inline float RandomFloat(void)
{
	return (float)rand() / RAND_MAX;
}

typedef unsigned char uChar;

template <class T, int Size>
class EXPOSURE_RENDER_DLL Vec
{
public:
	HOD Vec(void)
	{
		for (int i = 0; i < Size; i++)
			m_D[i] = T();
	}

	HOD Vec(const Vec<T, Size>& D)
	{
		for (int i = 0; i < Size; i++)
			m_D[i] = D[i];
	}

	HOD Vec(const T& Other)
	{
		*this = Other;
	}

	HOD T& operator = (const T& Other)
	{
		for (int i = 0; i < Size; i++)
			m_D[i] = Other[i];

		return *this;
	}

	HOD T operator[](const int& i) const
	{
		return m_D[i];
	}

	HOD T& operator[](const int& i)
	{
		return (&m_D)[i];
	}

	HOD Vec<T, Size> operator * (const float& f) const
	{
		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = m_D[i] * f;

		return Result;
	}

	HOD Vec<T, Size>& operator *= (const float& f)
	{
		for (int i = 0; i < Size; i++)
			m_D[i] *= f;

		return *this;
	}

	HOD Vec<T, Size> operator / (const float& f) const
	{
		const float Inv = 1.0f / f;

		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = m_D[i] * Inv;

		return Result;
	}

	HOD Vec<T, Size>& operator /= (float f)
	{
		const float Inv = 1.0f / f;

		for (int i = 0; i < Size; i++)
			m_D[i] *= Inv;

		return *this;
	}

	HOD bool operator < (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (m_D[i] > V[i])
				return false;
		}

		return true;
	}

	HOD bool operator > (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (m_D[i] < V[i])
				return false;
		}

		return true;
	}

	HOD bool operator == (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (m_D[i] != V[i])
				return false;
		}

		return true;
	}

	HOD int Max(void)
	{
		T Max;

		for (int i = 1; i < Size; i++)
		{
			if (m_D[i] > m_D[i - 1])
				Max = m_D[i];
		}
	}

	HOD int Min(void)
	{
		T Min;

		for (int i = 1; i < Size; i++)
		{
			if (m_D[i] < m_D[i - 1])
				Min = m_D[i];
		}
	}

protected:
	T	m_D[Size];
};

template <class T>
class EXPOSURE_RENDER_DLL Vec2 : public Vec<T, 2>
{
public:
	HOD Vec2(void)
	{
		m_D[0] = T();
		m_D[1] = T();
	}

	HOD Vec2(const T& V1, const T& V2)
	{
		m_D[0] = V1;
		m_D[1] = V2;
	}
};

template <class T>
class EXPOSURE_RENDER_DLL Vec3 : public Vec<T, 3>
{
public:
	HOD Vec3(void)
	{
		m_D[0] = T();
		m_D[1] = T();
		m_D[2] = T();
	}

	HOD Vec3(const T& V1, const T& V2, const T& V3)
	{
		m_D[0] = V1;
		m_D[1] = V2;
		m_D[2] = V3;
	}
};

template <class T>
class EXPOSURE_RENDER_DLL Vec4 : public Vec<T, 4>
{
public:
	HOD Vec4(void)
	{
		m_D[0] = T();
		m_D[1] = T();
		m_D[2] = T();
		m_D[3] = T();
	}

	HOD Vec4(const T& V1, const T& V2, const T& V3, const T& V4)
	{
		m_D[0] = V1;
		m_D[1] = V2;
		m_D[2] = V3;
		m_D[3] = V4;
	}
};

template <class T, int Size>
class EXPOSURE_RENDER_DLL ColorRGB : public Vec<T, Size>
{
public:
	HOD ColorRGB(void)
	{
	}

	HOD ColorRGB(const T& R, const T& G, const T& B)
	{
		Set(R, G, B);
	}

	HOD void Set(const T& R, const T& G, const T& B)
	{
		SetR(R);
		SetG(G);
		SetB(B);
	}

	HOD T GetR(void) const
	{
		return m_D[0];
	}

	HOD void SetR(const T& R)
	{
		m_D[0] = R;
	}

	HOD T GetG(void) const
	{
		return m_D[1];
	}

	HOD void SetG(const T& G)
	{
		m_D[1] = G;
	}

	HOD T GetB(void) const
	{
		return m_D[2];
	}

	HOD void SetB(const T& B)
	{
		m_D[2] = B;
	}

	HOD void SetBlack(void)
	{
		Set(T(), T(), T());
	}
};

template <class T>
class EXPOSURE_RENDER_DLL ColorRGBA : public ColorRGB<T, 4>
{
public:
	HOD void Set(const T& R, const T& G, const T& B, const T& A)
	{
		SetR(R);
		SetG(G);
		SetB(B);
		SetA(A);
	}

	HOD T GetA(void) const
	{
		return m_D[3];
	}

	HOD void SetA(const T& A)
	{
		m_D[3] = A;
	}
};

class EXPOSURE_RENDER_DLL ColorRGBuc : public ColorRGB<unsigned char, 3>
{
public:
	HOD ColorRGBuc(const unsigned char& R = 0, const unsigned char& G = 0, const unsigned char& B = 0)
	{
		Set(R, G, B);
	}

	HOD void FromRGBf(const float& R, const float& G, const float& B)
	{
		SetR(clamp2(R, 0.0f, 1.0f) * 255.0f);
		SetG(clamp2(G, 0.0f, 1.0f) * 255.0f);
		SetB(clamp2(B, 0.0f, 1.0f) * 255.0f);
	}

	HOD void FromXYZ(const float& X, const float& Y, const float& Z)
	{
		const float rWeight[3] = { 3.240479f, -1.537150f, -0.498535f };
		const float gWeight[3] = {-0.969256f,  1.875991f,  0.041556f };
		const float bWeight[3] = { 0.055648f, -0.204043f,  1.057311f };

		float R, G, B;

		R =	rWeight[0] * X + rWeight[1] * Y + rWeight[2] * Z; 
		G =	gWeight[0] * X + gWeight[1] * Y + gWeight[2] * Z;
		B =	bWeight[0] * X + bWeight[1] * Y + bWeight[2] * Z;

		clamp2(R, 0.0f, 1.0f);
		clamp2(G, 0.0f, 1.0f);
		clamp2(B, 0.0f, 1.0f);

		SetR((unsigned char)(R * 255.0f));
		SetG((unsigned char)(G * 255.0f));
		SetB((unsigned char)(B * 255.0f));
	}

	HOD void SetBlack(void)
	{
		Set(0, 0, 0);
	}

	HOD void SetWhite(void)
	{
		Set(255, 255, 255);
	}
};

class EXPOSURE_RENDER_DLL ColorRGBAuc : public ColorRGBA<unsigned char>
{
public:
	HOD ColorRGBAuc(const unsigned char& R = 0, const unsigned char& G = 0, const unsigned char& B = 0, const unsigned char& A = 0)
	{
		Set(R, G, B, A);
	}

	HOD ColorRGBAuc(const ColorRGBuc& RGB)
	{
		SetR(RGB.GetR());
		SetG(RGB.GetG());
		SetB(RGB.GetB());
	}

	HOD void FromRGBAf(const float& R, const float& G, const float& B, const float& A)
	{
		SetR(clamp2(R, 0.0f, 1.0f) * 255.0f);
		SetG(clamp2(G, 0.0f, 1.0f) * 255.0f);
		SetB(clamp2(B, 0.0f, 1.0f) * 255.0f);
		SetA(clamp2(A, 0.0f, 1.0f) * 255.0f);
	}

	HOD void FromXYZ(const float& X, const float& Y, const float& Z)
	{
		const float rWeight[3] = { 3.240479f, -1.537150f, -0.498535f };
		const float gWeight[3] = {-0.969256f,  1.875991f,  0.041556f };
		const float bWeight[3] = { 0.055648f, -0.204043f,  1.057311f };

		float R, G, B;

		R =	rWeight[0] * X + rWeight[1] * Y + rWeight[2] * Z;
		G =	gWeight[0] * X + gWeight[1] * Y + gWeight[2] * Z;
		B =	bWeight[0] * X + bWeight[1] * Y + bWeight[2] * Z;

		clamp2(R, 0.0f, 1.0f);
		clamp2(G, 0.0f, 1.0f);
		clamp2(B, 0.0f, 1.0f);

		SetR((unsigned char)(R * 255.0f));
		SetG((unsigned char)(G * 255.0f));
		SetB((unsigned char)(B * 255.0f));
	}

	HOD void SetBlack(void)
	{
		Set(0, 0, 0, 0);
	}

	HOD void SetWhite(void)
	{
		Set(255, 255, 255, 0);
	}
};

class EXPOSURE_RENDER_DLL ColorXYZf : public Vec3<float>
{
public:
	HOD ColorXYZf(float V = 0.0f)
	{
		Set(V, V, V);
	}

	HOD ColorXYZf(const float& X, const float& Y, const float& Z)
	{
		Set(X, Y, Z);
	}

	HOD void Set(const float& X, const float& Y, const float& Z)
	{
		SetX(X);
		SetY(Y);
		SetZ(Z);
	}

	HOD float GetX(void) const
	{
		return m_D[0];
	}

	HOD void SetX(const float& X)
	{
		m_D[0] = X;
	}

	HOD float GetY(void) const
	{
		return m_D[1];
	}

	HOD void SetY(const float& Y)
	{
		m_D[1] = Y;
	}

	HOD float GetZ(void) const
	{
		return m_D[2];
	}

	HOD void SetZ(const float& Z)
	{
		m_D[2] = Z;
	}

	HOD ColorXYZf& operator += (const ColorXYZf& XYZ)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] += XYZ[i];

		return *this;
	}

	HOD ColorXYZf operator + (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] += XYZ[i];

		return Result;
	}

	HOD ColorXYZf operator - (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] -= XYZ[i];

		return Result;
	}

	HOD ColorXYZf operator / (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] /= XYZ[i];

		return Result;
	}

	HOD ColorXYZf operator * (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; i++)
			Result.m_D[i] *= XYZ[i];

		return Result;
	}

	HOD ColorXYZf& operator *= (const ColorXYZf& XYZ)
	{
		for (int i = 0; i < 3; i++)
			m_D[i] *= XYZ[i];

		return *this;
	}

	HOD ColorXYZf operator * (const float& F) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] *= F;

		return Result;
	}

	HOD ColorXYZf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] *= F;

		return *this;
	}

	HOD ColorXYZf operator / (const float& F) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] /= F;

		return Result;
	}

	HOD ColorXYZf& operator /= (float a)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] /= a;

		return *this;
	}

	HOD bool operator == (const ColorXYZf& XYZ) const
	{
		for (int i = 0; i < 3; ++i)
			if (m_D[i] != XYZ[i])
				return false;

		return true;
	}

	HOD bool operator != (const ColorXYZf& XYZ) const
	{
		return !(*this == XYZ);
	}

	HOD float& operator[](int i)
	{
		return m_D[i];
	}

	HOD float operator[](int i) const
	{
		return m_D[i];
	}

	HOD ColorXYZf& ColorXYZf::operator = (const ColorXYZf& Other)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] = Other[i];

		return *this;
	}

	HOD bool IsBlack() const
	{
		for (int i = 0; i < 3; ++i)
			if (m_D[i] != 0.0f)
				return false;

		return true;
	}

	HOD ColorXYZf Clamp(const float& L = 0.0f, const float& H = 1.0f) const
	{
		ColorXYZf Result;

		for (int i = 0; i < 3; ++i)
			Result[i] = clamp2(m_D[i], L, H);

		return Result;
	}

	HOD float Y() const
	{
		float v = 0.0f;

		for (int i = 0; i < 3; i++)
			v += YWeight[i] * m_D[i];

		return v;
	}

	HOD void FromRGB(const float& R, const float& G, const float& B)
	{
		const float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
		const float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
		const float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

		m_D[0] = CoeffX[0] * R + CoeffX[1] * G + CoeffX[2] * B;
		m_D[1] = CoeffY[0] * R + CoeffY[1] * G + CoeffY[2] * B;
		m_D[2] = CoeffZ[0] * R + CoeffZ[1] * G + CoeffZ[2] * B;
	}
};

HOD inline ColorXYZf operator * (const float& F, const ColorXYZf& XYZ)
{
	return XYZ * F;
}

HOD inline ColorXYZf Lerp(const float& T, const ColorXYZf& C1, const ColorXYZf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorXYZf(OneMinusT * C1.GetX() + T * C2.GetX(), OneMinusT * C1.GetY() + T * C2.GetY(), OneMinusT * C1.GetZ() + T * C2.GetZ());
}

class EXPOSURE_RENDER_DLL ColorXYZAf : public Vec4<float>
{
public:
	HOD ColorXYZAf(const float& V = 0.0f)
	{
		Set(V, V, V, V);
	}

	HOD ColorXYZAf(const ColorXYZf& XYZ)
	{
		Set(XYZ.GetX(), XYZ.GetY(), XYZ.GetZ());
	}

	HOD ColorXYZAf(const float& X, const float& Y, const float& Z)
	{
		Set(X, Y, Z);
	}

	HOD ColorXYZAf(const float& X, const float& Y, const float& Z, const float& A)
	{
		Set(X, Y, Z, A);
	}
	
	HOD void Set(const float& X, const float& Y, const float& Z)
	{
		SetX(X);
		SetY(Y);
		SetZ(Z);
	}

	HOD void Set(const float& X, const float& Y, const float& Z, const float& A)
	{
		SetX(X);
		SetY(Y);
		SetZ(Z);
		SetA(A);
	}

	HOD float GetX(void) const
	{
		return m_D[0];
	}

	HOD void SetX(const float& X)
	{
		m_D[0] = X;
	}

	HOD float GetY(void) const
	{
		return m_D[1];
	}

	HOD void SetY(const float& Y)
	{
		m_D[1] = Y;
	}

	HOD float GetZ(void) const
	{
		return m_D[2];
	}

	HOD void SetZ(const float& Z)
	{
		m_D[2] = Z;
	}

	HOD float GetA(void) const
	{
		return m_D[3];
	}

	HOD void SetA(const float& A)
	{
		m_D[3] = A;
	}

	HOD ColorXYZAf& operator += (const ColorXYZAf& XYZ)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] += XYZ[i];

		return *this;
	}

	HOD ColorXYZAf operator + (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] += XYZ[i];

		return Result;
	}

	HOD ColorXYZAf operator - (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] -= XYZ[i];

		return Result;
	}

	HOD ColorXYZAf operator / (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] /= XYZ[i];

		return Result;
	}

	HOD ColorXYZAf operator * (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] *= XYZ[i];

		return Result;
	}

	HOD ColorXYZAf& operator *= (const ColorXYZAf& XYZ)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] *= XYZ[i];

		return *this;
	}

	HOD ColorXYZAf operator * (const float& F) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] *= F;

		return Result;
	}

	HOD ColorXYZAf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] *= F;

		return *this;
	}

	HOD ColorXYZAf operator / (const float& F) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] /= F;

		return Result;
	}

	HOD ColorXYZAf& operator/=(float a)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] /= a;

		return *this;
	}

	HOD bool operator == (const ColorXYZAf& XYZ) const
	{
		for (int i = 0; i < 3; ++i)
			if (m_D[i] != XYZ[i])
				return false;

		return true;
	}

	HOD bool operator != (const ColorXYZAf& XYZ) const
	{
		return !(*this == XYZ);
	}

	HOD float& operator[](int i)
	{
		return m_D[i];
	}

	HOD float operator[](int i) const
	{
		return m_D[i];
	}

	HOD ColorXYZAf& ColorXYZAf::operator = (const ColorXYZAf& Other)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] = Other[i];

		return *this;
	}

	HOD bool IsBlack() const
	{
		for (int i = 0; i < 3; ++i)
			if (m_D[i] != 0.0f)
				return false;

		return true;
	}

	HOD ColorXYZAf Clamp(const float& L = 0.0f, const float& H = 1.0f) const
	{
		ColorXYZAf Result;

		for (int i = 0; i < 3; ++i)
			Result[i] = clamp2(m_D[i], L, H);

		return Result;
	}

	HOD float Y() const
	{
		float v = 0.0f;

		for (int i = 0; i < 3; i++)
			v += YWeight[i] * m_D[i];

		return v;
	}

	HOD void FromRGB(const float& R, const float& G, const float& B)
	{
		const float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
		const float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
		const float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

		m_D[0] = CoeffX[0] * R + CoeffX[1] * G + CoeffX[2] * B;
		m_D[1] = CoeffY[0] * R + CoeffY[1] * G + CoeffY[2] * B;
		m_D[2] = CoeffZ[0] * R + CoeffZ[1] * G + CoeffZ[2] * B;
	}
};

HOD inline ColorXYZAf operator * (const float& F, const ColorXYZAf& XYZA)
{
	return XYZA * F;
}

HOD inline ColorXYZAf Lerp(const float& T, const ColorXYZAf& C1, const ColorXYZAf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorXYZAf(OneMinusT * C1.GetX() + T * C2.GetX(), OneMinusT * C1.GetY() + T * C2.GetY(), OneMinusT * C1.GetZ() + T * C2.GetZ(), OneMinusT * C1.GetA() + T * C2.GetA());
}

class EXPOSURE_RENDER_DLL ColorRGBf : public Vec3<float>
{
public:
	HOD ColorRGBf(void)
	{
	}

	HOD ColorRGBf(const float& R, const float& G, const float& B)
	{
		Set(R, G, B);
	}

	HOD ColorRGBf(const float& RGB)
	{
		Set(RGB, RGB, RGB);
	}

	HOD void Set(const float& R, const float& G, const float& B)
	{
		SetR(R);
		SetG(G);
		SetB(B);
	}

	HOD float GetR(void) const
	{
		return m_D[0];
	}

	HOD void SetR(const float& R)
	{
		m_D[0] = R;
	}

	HOD float GetG(void) const
	{
		return m_D[1];
	}

	HOD void SetG(const float& G)
	{
		m_D[1] = G;
	}

	HOD float GetB(void) const
	{
		return m_D[2];
	}

	HOD void SetB(const float& B)
	{
		m_D[2] = B;
	}

	HOD void SetBlack(void)
	{
		Set(0.0f, 0.0f, 0.0f);
	}

	HOD void SetWhite(void)
	{
		Set(1.0f, 1.0f, 1.0f);
	}

	HOD ColorRGBf& operator = (const ColorRGBf& Other)			
	{
		for (int i = 0; i < 3; i++)
			m_D[i] = Other[i];

		return *this;
	}

	HOD ColorRGBf& operator += (ColorRGBf& Other)		
	{
		for (int i = 0; i < 3; i++)
			m_D[i] += Other[i];

		return *this;
	}

	HOD ColorRGBf operator * (const float& F) const
	{
		return ColorRGBf(m_D[0] * F, m_D[1] * F, m_D[2] * F);
	}

	HOD ColorRGBf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; i++)
			m_D[i] *= F;

		return *this;
	}

	HOD ColorRGBf operator / (const float& F) const
	{
		const float Inv = 1.0f / F;
		return ColorRGBf(m_D[0] * Inv, m_D[1] * Inv, m_D[2] * Inv);
	}

	HOD ColorRGBf& operator /= (const float& F)
	{
		const float Inv = 1.0f / F;
		
		for (int i = 0; i < 3; i++)
			m_D[i] *= Inv;

		return *this;
	}

	HOD float operator[](int i) const
	{
		return m_D[i];
	}

	HOD float operator[](int i)
	{
		return m_D[i];
	}

	HOD bool Black(void)
	{
		for (int i = 0; i < 3; i++)
			if (m_D[i] != 0.0f)
				return false;

		return true;
	}

	HOD ColorRGBf Pow(const float& E)
	{
		return ColorRGBf(powf(m_D[0], E), powf(m_D[1], E), powf(m_D[1], E));
	}

	HOD void FromXYZ(const float& X, const float& Y, const float& Z)
	{
		const float rWeight[3] = { 3.240479f, -1.537150f, -0.498535f };
		const float gWeight[3] = {-0.969256f,  1.875991f,  0.041556f };
		const float bWeight[3] = { 0.055648f, -0.204043f,  1.057311f };

		Set(rWeight[0] * X + rWeight[1] * Y + rWeight[2] * Z, gWeight[0] * X + gWeight[1] * Y + gWeight[2] * Z, bWeight[0] * X + bWeight[1] * Y + bWeight[2] * Z);
	}

	HOD void ToneMap(const float& InvExposure)
	{
		m_D[0] = Clamp(1.0f - expf(-(m_D[0] * InvExposure)), 0.0f, 1.0f);
		m_D[1] = Clamp(1.0f - expf(-(m_D[1] * InvExposure)), 0.0f, 1.0f);
		m_D[2] = Clamp(1.0f - expf(-(m_D[2] * InvExposure)), 0.0f, 1.0f);
	}
};

HOD inline ColorRGBf Lerp(const float& T, const ColorRGBf& C1, const ColorRGBf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorRGBf(OneMinusT * C1.GetR() + T * C2.GetR(), OneMinusT * C1.GetG() + T * C2.GetG(), OneMinusT * C1.GetB() + T * C2.GetB());
}

inline HOD Vec3f MinVec3f(Vec3f a, Vec3f b)
{
	return Vec3f(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

inline HOD Vec3f MaxVec3f(Vec3f a, Vec3f b)
{
	return Vec3f(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

class EXPOSURE_RENDER_DLL CTransferFunction
{
public:
	float			m_P[MAX_NO_TF_POINTS];		/*!< Node positions */
	ColorRGBf		m_C[MAX_NO_TF_POINTS];		/*!< Node colors in HDR RGB */
	int				m_NoNodes;					/*!< No. nodes */

	// ToDo: Add description
	HO CTransferFunction(void)
	{
		for (int i = 0; i < MAX_NO_TF_POINTS; i++)
		{
			m_P[i]	= 0.0f;
//			m_C[i]	= SPEC_BLACK;
		}

		m_NoNodes = 0;
	}

	// ToDo: Add description
	HO ~CTransferFunction(void)
	{
	}

	// ToDo: Add description
	HOD CTransferFunction& operator=(const CTransferFunction& Other)
	{
		for (int i = 0; i < MAX_NO_TF_POINTS; i++)
		{
			m_P[i]	= Other.m_P[i];
			m_C[i]	= Other.m_C[i];
		}

		m_NoNodes = Other.m_NoNodes;

		return *this;
	}

	// ToDo: Add description
	HOD ColorRGBf F(const float& P)
	{
		for (int i = 0; i < m_NoNodes - 1; i++)
		{
			if (P >= m_P[i] && P < m_P[i + 1])
			{
				const float T = (float)(P - m_P[i]) / (m_P[i + 1] - m_P[i]);
				return Lerp(T, m_C[i], m_C[i + 1]);
			}
		}

		return ColorRGBf(0.0f);
	}
};

// ToDo: Add description
class EXPOSURE_RENDER_DLL CTransferFunctions
{
public:
	CTransferFunction	m_Opacity;
	CTransferFunction	m_Diffuse;
	CTransferFunction	m_Specular;
	CTransferFunction	m_Emission;
	CTransferFunction	m_Roughness;

	// ToDo: Add description
	HO CTransferFunctions(void)
	{
	}

	// ToDo: Add description
	HO ~CTransferFunctions(void)
	{
	}

	// ToDo: Add description
	HOD CTransferFunctions& operator=(const CTransferFunctions& Other)
	{
		m_Opacity		= Other.m_Opacity;
		m_Diffuse		= Other.m_Diffuse;
		m_Specular		= Other.m_Specular;
		m_Emission		= Other.m_Emission;
		m_Roughness		= Other.m_Roughness;

		return *this;
	}
};

#define CLR_RAD_BLACK										ColorXYZf(0.0f)
#define CLR_RAD_WHITE										ColorXYZf(1.0f)
#define CLR_RAD_RED											ColorXYZf(1.0f, 0.0f, 0.0)
#define CLR_RAD_GREEN										ColorXYZf(0.0f, 1.0f, 0.0)
#define CLR_RAD_BLUE										ColorXYZf(1.0f)
#define SPEC_BLACK											ColorXYZf(0.0f)
#define SPEC_GRAY_10										ColorXYZf(1.0f)
#define SPEC_GRAY_20										ColorXYZf(1.0f)
#define SPEC_GRAY_30										ColorXYZf(1.0f)
#define SPEC_GRAY_40										ColorXYZf(1.0f)
#define SPEC_GRAY_50										ColorXYZf(0.5f)
#define SPEC_GRAY_60										ColorXYZf(1.0f)
#define SPEC_GRAY_70										ColorXYZf(1.0f)
#define SPEC_GRAY_80										ColorXYZf(1.0f)
#define SPEC_GRAY_90										ColorXYZf(1.0f)
#define SPEC_WHITE											ColorXYZf(1.0f)
#define SPEC_CYAN											ColorXYZf(1.0f)
#define SPEC_RED											ColorXYZf(1.0f, 0.0f, 0.0f)