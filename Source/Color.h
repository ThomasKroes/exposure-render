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

HOST_DEVICE inline float LuminanceFromRGB(const float& R, const float& G, const float& B)
{
	return Clamp(0.3f * R + 0.59f * G + 0.11f * B, 0.0f, 255.0f);
}

HOST_DEVICE inline void XYZToRGB(const float xyz[3], float rgb[3])
{
	rgb[0] =  3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2];
	rgb[1] = -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2];
	rgb[2] =  0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2];
}

HOST_DEVICE inline void RGBToXYZ(const float rgb[3], float xyz[3])
{
	xyz[0] = 0.412453f*rgb[0] + 0.357580f*rgb[1] + 0.180423f*rgb[2];
	xyz[1] = 0.212671f*rgb[0] + 0.715160f*rgb[1] + 0.072169f*rgb[2];
	xyz[2] = 0.019334f*rgb[0] + 0.119193f*rgb[1] + 0.950227f*rgb[2];
}

class EXPOSURE_RENDER_DLL ColorXYZf : public Vec<float, 3>
{
public:
	ColorXYZf() :
		Vec<float, 3>()
	{
	}
	
	HOST_DEVICE ColorXYZf(const float& XYZ)
	{
		this->D[0] = XYZ;
		this->D[1] = XYZ;
		this->D[2] = XYZ;
	}

	HOST_DEVICE ColorXYZf(const float& X, const float& Y, const float& Z)
	{
		this->D[0] = X;
		this->D[1] = Y;
		this->D[2] = Z;
	}
};

class EXPOSURE_RENDER_DLL ColorXYZAf : public Vec<float, 4>
{
public:
	ColorXYZAf() :
		Vec<float, 4>()
	{
	}
	
	HOST_DEVICE ColorXYZAf(const float& XYZA)
	{
		this->D[0] = XYZA;
		this->D[1] = XYZA;
		this->D[2] = XYZA;
		this->D[3] = XYZA;
	}

	HOST_DEVICE ColorXYZAf(const float& X, const float& Y, const float& Z, const float& A)
	{
		this->D[0] = X;
		this->D[1] = Y;
		this->D[2] = Z;
		this->D[3] = A;
	}
};

class EXPOSURE_RENDER_DLL ColorRGBuc : public Vec<unsigned char, 3>
{
public:
	ColorRGBuc() :
		Vec<unsigned char, 3>()
	{
	}
	
	HOST_DEVICE ColorRGBuc(const unsigned char& RGB)
	{
		this->D[0] = RGB;
		this->D[1] = RGB;
		this->D[2] = RGB;
	}

	HOST_DEVICE ColorRGBuc(const unsigned char& R, const unsigned char& G, const unsigned char& B)
	{
		this->D[0] = R;
		this->D[1] = G;
		this->D[2] = B;
	}
};

class EXPOSURE_RENDER_DLL ColorRGBAuc : public Vec<unsigned char, 4>
{
public:
	ColorRGBAuc() :
		Vec<unsigned char, 4>()
	{
	}
	
	HOST_DEVICE ColorRGBAuc(const unsigned char& RGBA)
	{
		this->D[0] = RGBA;
		this->D[1] = RGBA;
		this->D[2] = RGBA;
		this->D[3] = RGBA;
	}

	HOST_DEVICE ColorRGBAuc(const unsigned char& R, const unsigned char& G, const unsigned char& B, const unsigned char& A)
	{
		this->D[0] = R;
		this->D[1] = G;
		this->D[2] = B;
		this->D[3] = A;
	}
};

template <class T> inline HOST_DEVICE ColorRGBAuc operator / (const ColorRGBAuc& V, const T& F)				{ return ColorRGBAuc((float)V[0] / F, (float)V[1] / F, (float)V[2] / F, V[3]);						};

HOST_DEVICE inline ColorRGBAuc operator * (const float& F, const ColorRGBAuc& XYZ)
{
	return ColorRGBAuc((unsigned char)(XYZ[0] * F), (unsigned char)(XYZ[1] * F), (unsigned char)(XYZ[2] * F));
}

HOST_DEVICE inline ColorRGBAuc operator + (const ColorRGBAuc& A, const ColorRGBAuc& B)
{
	return ColorRGBAuc(A[0] + B[0], A[1] + B[1], A[2] + B[2], A[3] + B[3]);
}

HOST_DEVICE inline ColorXYZf operator * (const float& F, const ColorXYZf& XYZ)
{
	return XYZ * F;
}

HOST_DEVICE inline ColorXYZf Lerp(const float& T, const ColorXYZf& C1, const ColorXYZf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorXYZf(OneMinusT * C1[0] + T * C2[0], OneMinusT * C1[1] + T * C2[1], OneMinusT * C1[2] + T * C2[2]);
}

HOST_DEVICE inline ColorXYZAf operator * (const float& F, const ColorXYZAf& XYZA)
{
	return XYZA * F;
}

HOST_DEVICE inline ColorXYZAf Lerp(const float& T, const ColorXYZAf& C1, const ColorXYZAf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorXYZAf(OneMinusT * C1[0] + T * C2[0], OneMinusT * C1[1] + T * C2[1], OneMinusT * C1[2] + T * C2[2], OneMinusT * C1[3] + T * C2[3]);
}

HOST_DEVICE ColorRGBAuc Lerp(const ColorRGBAuc& A, const ColorRGBAuc& B, const float& T)
{
	ColorRGBAuc Result;

	for (int i = 0; i < 3; i++)
		Result[i] = (unsigned char)((1.0f - T) * (float)A[i] + T * (float)B[i]);

	return Result;
}

#define SPEC_BLACK											ColorXYZf(0.0f)

}
