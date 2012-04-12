/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or witDEVut modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software witDEVut specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT DEVLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT DEVLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) DEVWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Vector.h"

namespace ExposureRender
{

class ColorRGBf;
class ColorXYZf;
class ColorXYZAf;
class ColorRGBuc;
class ColorRGBAuc;

class EXPOSURE_RENDER_DLL ColorRGBf : public Vec3f
{
public:
	HOST_DEVICE ColorRGBf() :
		Vec3f()
	{
	}
	
	HOST_DEVICE ColorRGBf(const Vec3f& Other) :
		Vec3f(Other)
	{
	}

	HOST_DEVICE ColorRGBf(const ColorRGBf& Other)
	{
		*this = Other;
	}

	HOST_DEVICE ColorRGBf(const float& V)
	{
	}

	HOST_DEVICE ColorRGBf(const float& R, const float& G, const float& B)
	{
		this->D[0] = R;
		this->D[1] = G;
		this->D[2] = B;
	}

	HOST_DEVICE ColorRGBf(const ColorXYZf& XYZ);
};

class EXPOSURE_RENDER_DLL ColorXYZf : public Vec3f
{
public:
	HOST_DEVICE ColorXYZf() :
		Vec3f()
	{
	}
	
	HOST_DEVICE ColorXYZf(const Vec3f& Other) :
		Vec3f(Other)
	{
	}

	HOST_DEVICE ColorXYZf(const ColorXYZf& Other)
	{
		*this = Other;
	}

	HOST_DEVICE ColorXYZf(const float& V)
	{
	}

	HOST_DEVICE ColorXYZf(const float& R, const float& G, const float& B)
	{
		this->D[0] = R;
		this->D[1] = G;
		this->D[2] = B;
	}

	HOST_DEVICE ColorXYZf(const ColorRGBf& RGB);
	HOST_DEVICE ColorXYZf(const ColorRGBuc& RGB);
	HOST_DEVICE ColorXYZf(const ColorRGBAuc& RGB);
};

class EXPOSURE_RENDER_DLL ColorXYZAf : public Vec<float, 4>
{
public:
	HOST_DEVICE ColorXYZAf() :
		Vec<float, 4>()
	{
	}
	
	HOST_DEVICE ColorXYZAf(const Vec<float, 4>& V) :
		Vec<float, 4>()
	{
	}

	HOST_DEVICE ColorXYZAf(const ColorXYZAf& Other)
	{
		*this = Other;
	}

	HOST_DEVICE ColorXYZAf(const float& V)
	{
	}
	
	HOST_DEVICE ColorXYZAf(const float& R, const float& G, const float& B, const float& A)
	{
		this->D[0] = R;
		this->D[1] = G;
		this->D[2] = B;
		this->D[3] = A;
	}

	HOST_DEVICE ColorXYZAf(const ColorRGBf& RGB);
};

class EXPOSURE_RENDER_DLL ColorRGBuc : public Vec<unsigned char, 3>
{
public:
	HOST_DEVICE ColorRGBuc() :
		Vec<unsigned char, 3>()
	{
	}
	
	HOST_DEVICE ColorRGBuc(const Vec<unsigned char, 3>& V) :
		Vec<unsigned char, 3>(V)
	{
	}

	HOST_DEVICE ColorRGBuc(const ColorRGBuc& Other)
	{
		*this = Other;
	}

	HOST_DEVICE ColorRGBuc(const float& V)
	{
	}

	HOST_DEVICE ColorRGBuc(const unsigned char& R, const unsigned char& G, const unsigned char& B)
	{
		this->D[0] = R;
		this->D[1] = G;
		this->D[2] = B;
	}

	HOST_DEVICE ColorRGBuc(const ColorXYZf& XYZ);
};

class EXPOSURE_RENDER_DLL ColorRGBAuc : public Vec<unsigned char, 4>
{
public:
	HOST_DEVICE ColorRGBAuc() :
		Vec<unsigned char, 4>()
	{
	}
	
	HOST_DEVICE ColorRGBAuc(const Vec<unsigned char, 4>& V) :
		Vec<unsigned char, 4>(V)
	{
	}

	HOST_DEVICE ColorRGBAuc(const ColorRGBAuc& Other)
	{
		*this = Other;
	}

	HOST_DEVICE ColorRGBAuc(const float& V)
	{
	}

	HOST_DEVICE ColorRGBAuc(const unsigned char& R, const unsigned char& G, const unsigned char& B, const unsigned char& A)
	{
		this->D[0] = R;
		this->D[1] = G;
		this->D[2] = B;
		this->D[3] = A;
	}

	HOST_DEVICE ColorRGBAuc(const ColorXYZf& XYZ);
};

HOST_DEVICE ColorRGBf::ColorRGBf(const ColorXYZf& XYZ)
{
	this->D[0] =  3.240479f * XYZ[0] - 1.537150f * XYZ[1] - 0.498535f * XYZ[2];
	this->D[1] = -0.969256f * XYZ[0] + 1.875991f * XYZ[1] + 0.041556f * XYZ[2];
	this->D[2] =  0.055648f * XYZ[0] - 0.204043f * XYZ[1] + 1.057311f * XYZ[2];
};

HOST_DEVICE ColorXYZf::ColorXYZf(const ColorRGBf& RGB)
{
	this->D[0] = 0.412453f * RGB[0] + 0.357580f * RGB[1] + 0.180423f * RGB[2];
	this->D[1] = 0.212671f * RGB[0] + 0.715160f * RGB[1] + 0.072169f * RGB[2];
	this->D[2] = 0.019334f * RGB[0] + 0.119193f * RGB[1] + 0.950227f * RGB[2];
};

HOST_DEVICE ColorXYZf::ColorXYZf(const ColorRGBuc& RGB)
{
	float RGBf[3] = 
	{
		ONE_OVER_255 * (float)RGB[0],
		ONE_OVER_255 * (float)RGB[1],
		ONE_OVER_255 * (float)RGB[2]
	};

	this->D[0] = 0.412453f * RGBf[0] + 0.357580f * RGBf[1] + 0.180423f * RGBf[2];
	this->D[1] = 0.212671f * RGBf[0] + 0.715160f * RGBf[1] + 0.072169f * RGBf[2];
	this->D[2] = 0.019334f * RGBf[0] + 0.119193f * RGBf[1] + 0.950227f * RGBf[2];
};

HOST_DEVICE ColorXYZf::ColorXYZf(const ColorRGBAuc& RGBA)
{
	float RGBAf[3] = 
	{
		ONE_OVER_255 * (float)RGBA[0],
		ONE_OVER_255 * (float)RGBA[1],
		ONE_OVER_255 * (float)RGBA[2]
	};

	this->D[0] = 0.412453f * RGBAf[0] + 0.357580f * RGBAf[1] + 0.180423f * RGBAf[2];
	this->D[1] = 0.212671f * RGBAf[0] + 0.715160f * RGBAf[1] + 0.072169f * RGBAf[2];
	this->D[2] = 0.019334f * RGBAf[0] + 0.119193f * RGBAf[1] + 0.950227f * RGBAf[2];
};

HOST_DEVICE ColorXYZAf::ColorXYZAf(const ColorRGBf& RGB)
{
	this->D[0] = 0.412453f * RGB[0] + 0.357580f * RGB[1] + 0.180423f * RGB[2];
	this->D[1] = 0.212671f * RGB[0] + 0.715160f * RGB[1] + 0.072169f * RGB[2];
	this->D[2] = 0.019334f * RGB[0] + 0.119193f * RGB[1] + 0.950227f * RGB[2];
};

HOST_DEVICE ColorRGBuc::ColorRGBuc(const ColorXYZf& XYZ)
{
	int RGB[3] = 
	{
		255 * (3.240479f * XYZ[0] - 1.537150f * XYZ[1] - 0.498535f * XYZ[2]),
		255 * (-0.969256f * XYZ[0] + 1.875991f * XYZ[1] + 0.041556f * XYZ[2]),
		255 * (0.055648f * XYZ[0] - 0.204043f * XYZ[1] + 1.057311f * XYZ[2])
	};

	this->D[0] = (unsigned char)ExposureRender::Clamp(RGB[0], 0, 255);
	this->D[1] = (unsigned char)ExposureRender::Clamp(RGB[1], 0, 255);
	this->D[2] = (unsigned char)ExposureRender::Clamp(RGB[2], 0, 255);
};

HOST_DEVICE ColorRGBAuc::ColorRGBAuc(const ColorXYZf& XYZ)
{
	const int RGB[3] = 
	{
		255 * (3.240479f * XYZ[0] - 1.537150f * XYZ[1] - 0.498535f * XYZ[2]),
		255 * (-0.969256f * XYZ[0] + 1.875991f * XYZ[1] + 0.041556f * XYZ[2]),
		255 * (0.055648f * XYZ[0] - 0.204043f * XYZ[1] + 1.057311f * XYZ[2])
	};

	this->D[0] = (unsigned char)ExposureRender::Clamp(RGB[0], 0, 255);
	this->D[1] = (unsigned char)ExposureRender::Clamp(RGB[1], 0, 255);
	this->D[2] = (unsigned char)ExposureRender::Clamp(RGB[2], 0, 255);
};

static inline HOST_DEVICE ColorXYZAf operator - (const ColorXYZAf& A, const ColorXYZAf& B)						{ return A - B;														};
static inline HOST_DEVICE ColorXYZAf operator * (const ColorXYZAf& V, const float& F)							{ return ColorXYZAf(V[0] * F, V[1] * F, V[2] * F, V[3] * F);					};
static inline HOST_DEVICE ColorXYZAf operator * (const float& F, const ColorXYZAf& V)							{ return ColorXYZAf(V[0] * F, V[1] * F, V[2] * F, V[3] * F);					};
static inline HOST_DEVICE ColorXYZAf Lerp(const float& LerpC, const ColorXYZAf& A, const ColorXYZAf& B)			{ return LerpC * (B - A);														};

static inline HOST_DEVICE ColorRGBuc operator * (const ColorRGBuc& V, const unsigned char& C)					{ return ColorRGBuc(V[0] * C, V[1] * C, V[2] * C);								};
static inline HOST_DEVICE ColorRGBuc operator * (const unsigned char& C, const ColorRGBuc& V)					{ return ColorRGBuc(V[0] * C, V[1] * C, V[2] * C);								};
//static inline HOST_DEVICE ColorRGBuc Lerp(const float& LerpC, const ColorRGBuc& A, const ColorRGBuc& B)			{ return LerpC * (B - A);														};

static inline HOST_DEVICE ColorRGBAuc operator * (const ColorRGBAuc& V, const unsigned char& C)					{ return ColorRGBAuc(V[0] * C, V[1] * C, V[2] * C, V[3] * C);					};
static inline HOST_DEVICE ColorRGBAuc operator * (const unsigned char& C, const ColorRGBAuc& V)					{ return ColorRGBAuc(V[0] * C, V[1] * C, V[2] * C, V[3] * C);					};
//static inline HOST_DEVICE ColorRGBAuc Lerp(const float& LerpC, const ColorRGBAuc& A, const ColorRGBAuc& B)		{ return LerpC * (B - A);														};

#define SPEC_BLACK											ColorXYZf(0.0f)

}
