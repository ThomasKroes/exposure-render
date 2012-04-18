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


#define IS_BLACK(size)						\
HOST_DEVICE bool IsBlack()					\
{											\
	for (int i = 0; i < size; i++)			\
		if (this->D[i] != 0)				\
			return false;					\
											\
	return true;							\
}

class ColorRGBf;
class ColorXYZf;
class ColorXYZAf;
class ColorRGBuc;
class ColorRGBAuc;

class EXPOSURE_RENDER_DLL ColorRGBf
{
public:
	CONSTRUCTORS(ColorRGBf, float, 3)
	VEC3_CONSTRUCTOR(ColorRGBf, float)
	ALL_OPERATORS(ColorRGBf, float, 3)
	MIN_MAX(ColorRGBf, float, 3)
	CLAMP(ColorRGBf, float, 3)
	IS_BLACK(3)

	static inline HOST_DEVICE ColorRGBf FromXYZf(const ColorXYZf& XYZ);
	static inline HOST_DEVICE ColorRGBf FromXYZAf(const ColorXYZAf& XYZA);
	static inline HOST_DEVICE ColorRGBf Black() { return ColorRGBf(); }

	DATA(float, 3)
};

class EXPOSURE_RENDER_DLL ColorXYZf
{
public:
	CONSTRUCTORS(ColorXYZf, float, 3)
	VEC3_CONSTRUCTOR(ColorXYZf, float)
	ALL_OPERATORS(ColorXYZf, float, 3)
	MIN_MAX(ColorXYZf, float, 3)
	CLAMP(ColorXYZf, float, 3)
	IS_BLACK(3)

	static inline HOST_DEVICE ColorXYZf FromRGBf(const ColorRGBf& RGB);
	static inline HOST_DEVICE ColorXYZf FromRGBuc(const ColorRGBuc& RGB);
	static inline HOST_DEVICE ColorXYZf FromRGBAuc(const ColorRGBAuc& RGB);
	static inline HOST_DEVICE ColorXYZf Black() { return ColorXYZf(); }

	DATA(float, 3)
};

class EXPOSURE_RENDER_DLL ColorXYZAf
{
public:
	CONSTRUCTORS(ColorXYZAf, float, 4)
	VEC4_CONSTRUCTOR(ColorXYZAf, float)
	ALL_OPERATORS(ColorXYZAf, float, 4)
	MIN_MAX(ColorXYZAf, float, 4)
	CLAMP(ColorXYZAf, float, 4)
	IS_BLACK(4)

	static inline HOST_DEVICE ColorXYZAf FromRGBf(const ColorRGBf& RGB);
	static inline HOST_DEVICE ColorXYZAf Black() { return ColorXYZAf(); }

	DATA(float, 4)
};

class EXPOSURE_RENDER_DLL ColorRGBuc
{
public:
	CONSTRUCTORS(ColorRGBuc, unsigned char, 3)
	VEC3_CONSTRUCTOR(ColorRGBuc, unsigned char)
	ALL_OPERATORS(ColorRGBuc, unsigned char, 3)
//	MIN_MAX(ColorRGBuc, unsigned char, 3)
//	CLAMP(ColorRGBuc, unsigned char, 3)
	IS_BLACK(3)

	static inline HOST_DEVICE ColorRGBuc FromXYZf(const ColorXYZf& XYZ);
	static inline HOST_DEVICE ColorRGBuc Black() { return ColorRGBuc(); }

	DATA(unsigned char, 3)
};

class EXPOSURE_RENDER_DLL ColorRGBAuc
{
public:
	CONSTRUCTORS(ColorRGBAuc, unsigned char, 4)
	VEC4_CONSTRUCTOR(ColorRGBAuc, unsigned char)
	ALL_OPERATORS(ColorRGBAuc, unsigned char, 4)
//	MIN_MAX(ColorRGBAuc, unsigned char, 4)
//	CLAMP(ColorRGBAuc, unsigned char, 4)
	IS_BLACK(4)

	static inline HOST_DEVICE ColorRGBAuc FromXYZf(const ColorXYZf& XYZ);
	static inline HOST_DEVICE ColorRGBAuc Black() { return ColorRGBAuc(); }

	DATA(unsigned char, 4)
};

HOST_DEVICE ColorRGBf ColorRGBf::FromXYZf(const ColorXYZf& XYZ)
{
	ColorRGBf Result;

	Result[0] =  3.240479f * XYZ[0] - 1.537150f * XYZ[1] - 0.498535f * XYZ[2];
	Result[1] = -0.969256f * XYZ[0] + 1.875991f * XYZ[1] + 0.041556f * XYZ[2];
	Result[2] =  0.055648f * XYZ[0] - 0.204043f * XYZ[1] + 1.057311f * XYZ[2];

	return Result;
};

HOST_DEVICE ColorRGBf ColorRGBf::FromXYZAf(const ColorXYZAf& XYZA)
{
	ColorRGBf Result;

	Result[0] =  3.240479f * XYZA[0] - 1.537150f * XYZA[1] - 0.498535f * XYZA[2];
	Result[1] = -0.969256f * XYZA[0] + 1.875991f * XYZA[1] + 0.041556f * XYZA[2];
	Result[2] =  0.055648f * XYZA[0] - 0.204043f * XYZA[1] + 1.057311f * XYZA[2];
	Result[3] = XYZA[3];

	return Result;
};

HOST_DEVICE ColorXYZf ColorXYZf::FromRGBf(const ColorRGBf& RGB)
{
	ColorXYZf Result;

	Result[0] = 0.412453f * RGB[0] + 0.357580f * RGB[1] + 0.180423f * RGB[2];
	Result[1] = 0.212671f * RGB[0] + 0.715160f * RGB[1] + 0.072169f * RGB[2];
	Result[2] = 0.019334f * RGB[0] + 0.119193f * RGB[1] + 0.950227f * RGB[2];

	return Result;
};

HOST_DEVICE ColorXYZf ColorXYZf::FromRGBuc(const ColorRGBuc& RGB)
{
	ColorXYZf Result;

	float RGBf[3] = 
	{
		ONE_OVER_255 * (float)RGB[0],
		ONE_OVER_255 * (float)RGB[1],
		ONE_OVER_255 * (float)RGB[2]
	};

	Result[0] = 0.412453f * RGBf[0] + 0.357580f * RGBf[1] + 0.180423f * RGBf[2];
	Result[1] = 0.212671f * RGBf[0] + 0.715160f * RGBf[1] + 0.072169f * RGBf[2];
	Result[2] = 0.019334f * RGBf[0] + 0.119193f * RGBf[1] + 0.950227f * RGBf[2];

	return Result;
};

HOST_DEVICE ColorXYZf ColorXYZf::FromRGBAuc(const ColorRGBAuc& RGBA)
{
	ColorXYZf Result;

	float RGBAf[3] = 
	{
		ONE_OVER_255 * (float)RGBA[0],
		ONE_OVER_255 * (float)RGBA[1],
		ONE_OVER_255 * (float)RGBA[2]
	};

	Result[0] = 0.412453f * RGBAf[0] + 0.357580f * RGBAf[1] + 0.180423f * RGBAf[2];
	Result[1] = 0.212671f * RGBAf[0] + 0.715160f * RGBAf[1] + 0.072169f * RGBAf[2];
	Result[2] = 0.019334f * RGBAf[0] + 0.119193f * RGBAf[1] + 0.950227f * RGBAf[2];

	return Result;
};

HOST_DEVICE ColorXYZAf ColorXYZAf::FromRGBf(const ColorRGBf& RGB)
{
	ColorXYZAf Result;

	Result[0] = 0.412453f * RGB[0] + 0.357580f * RGB[1] + 0.180423f * RGB[2];
	Result[1] = 0.212671f * RGB[0] + 0.715160f * RGB[1] + 0.072169f * RGB[2];
	Result[2] = 0.019334f * RGB[0] + 0.119193f * RGB[1] + 0.950227f * RGB[2];

	return Result;
};

HOST_DEVICE ColorRGBuc ColorRGBuc::FromXYZf(const ColorXYZf& XYZ)
{
	ColorRGBuc Result;

	int RGB[3] = 
	{
		(int)(255.0f * (3.240479f * XYZ[0] - 1.537150f * XYZ[1] - 0.498535f * XYZ[2])),
		(int)(255.0f * (-0.969256f * XYZ[0] + 1.875991f * XYZ[1] + 0.041556f * XYZ[2])),
		(int)(255.0f * (0.055648f * XYZ[0] - 0.204043f * XYZ[1] + 1.057311f * XYZ[2]))
	};

	Result[0] = (unsigned char)ExposureRender::Clamp(RGB[0], 0, 255);
	Result[1] = (unsigned char)ExposureRender::Clamp(RGB[1], 0, 255);
	Result[2] = (unsigned char)ExposureRender::Clamp(RGB[2], 0, 255);

	return Result;
};

HOST_DEVICE ColorRGBAuc ColorRGBAuc::FromXYZf(const ColorXYZf& XYZ)
{
	ColorRGBAuc Result;

	const int RGB[3] = 
	{
		(int)(255.0f * (3.240479f * XYZ[0] - 1.537150f * XYZ[1] - 0.498535f * XYZ[2])),
		(int)(255.0f * (-0.969256f * XYZ[0] + 1.875991f * XYZ[1] + 0.041556f * XYZ[2])),
		(int)(255.0f * (0.055648f * XYZ[0] - 0.204043f * XYZ[1] + 1.057311f * XYZ[2]))
	};

	Result[0] = (unsigned char)ExposureRender::Clamp(RGB[0], 0, 255);
	Result[1] = (unsigned char)ExposureRender::Clamp(RGB[1], 0, 255);
	Result[2] = (unsigned char)ExposureRender::Clamp(RGB[2], 0, 255);

	return Result;
};

static inline HOST_DEVICE ColorRGBf operator * (ColorRGBf& RGB, const float& F)									{ return RGB *= F; 								};
static inline HOST_DEVICE ColorRGBf operator * (const float& F, ColorRGBf& RGB)									{ return RGB *= F; 								};
static inline HOST_DEVICE ColorRGBf Lerp(const float& LerpC, const ColorRGBf& A, const ColorRGBf& B)			{ return LerpC * (B - A);						};

static inline HOST_DEVICE ColorXYZf operator * (ColorXYZf& XYZ, const float& F)									{ return XYZ *= F; 								};
static inline HOST_DEVICE ColorXYZf operator * (const float& F, ColorXYZf& XYZ)									{ return XYZ *= F; 								};
static inline HOST_DEVICE ColorXYZf Lerp(const float& LerpC, const ColorXYZf& A, const ColorXYZf& B)			{ return LerpC * (B - A);						};

static inline HOST_DEVICE ColorXYZAf operator * (ColorXYZAf& XYZA, const float& F)								{ return XYZA *= F;								};
static inline HOST_DEVICE ColorXYZAf operator * (const float& F, ColorXYZAf& XYZA)								{ return XYZA *= F;								};
static inline HOST_DEVICE ColorXYZAf Lerp(const float& LerpC, const ColorXYZAf& A, const ColorXYZAf& B)			{ return LerpC * (B - A);						};

static inline HOST_DEVICE ColorRGBuc operator * (ColorRGBuc& RGB, const unsigned char& UC)						{ return RGB *= UC;								};
static inline HOST_DEVICE ColorRGBuc operator * (const unsigned char& UC, ColorRGBuc& RGB)						{ return RGB *= UC;								};

static inline HOST_DEVICE ColorRGBAuc operator * (ColorRGBAuc& RGBA, const unsigned char& UC)					{ return RGBA *= UC;							};
static inline HOST_DEVICE ColorRGBAuc operator * (const unsigned char& UC, ColorRGBAuc& RGBA)					{ return RGBA *= UC;							};

}
