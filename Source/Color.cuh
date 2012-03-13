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

#include "Vector.cuh"

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

template <class T, int Size>
class ColorRGB : public Vec<T, Size>
{
public:
	HOST_DEVICE ColorRGB(void)
	{
		this->SetBlack();
	}

	HOST_DEVICE ColorRGB(T RGB)
	{
		this->Set(RGB, RGB, RGB);
	}

	HOST_DEVICE ColorRGB(T R, T G, T B)
	{
		this->Set(R, G, B);
	}

	HOST_DEVICE void Set(T R, T G, T B)
	{
		this->SetR(R);
		this->SetG(G);
		this->SetB(B);
	}

	HOST_DEVICE T GetR(void) const
	{
		return this->D[0];
	}

	HOST_DEVICE void SetR(const T& R)
	{
		this->D[0] = R;
	}

	HOST_DEVICE T GetG(void) const
	{
		return this->D[1];
	}

	HOST_DEVICE void SetG(const T& G)
	{
		this->D[1] = G;
	}

	HOST_DEVICE T GetB(void) const
	{
		return this->D[2];
	}

	HOST_DEVICE void SetB(const T& B)
	{
		this->D[2] = B;
	}

	HOST_DEVICE void SetBlack(void)
	{
		this->Set(T(), T(), T());
	}
};

template <class T>
class ColorRGBA : public ColorRGB<T, 4>
{
public:
	HOST_DEVICE ColorRGBA(void)
	{
		this->SetBlack();
	}

	HOST_DEVICE ColorRGBA(T RGBA)
	{
		this->Set(RGBA, RGBA, RGBA, RGBA);
	}

	HOST_DEVICE void Set(T R, T G, T B, T A)
	{
		this->SetR(R);
		this->SetG(G);
		this->SetB(B);
		this->SetA(A);
	}

	HOST_DEVICE T GetA(void) const
	{
		return this->D[3];
	}

	HOST_DEVICE void SetA(const T& A)
	{
		this->D[3] = A;
	}
};

class ColorRGBuc : public ColorRGB<unsigned char, 3>
{
public:
	HOST_DEVICE ColorRGBuc(unsigned char RGB)
	{
		this->Set(RGB, RGB, RGB);
	}

	HOST_DEVICE ColorRGBuc(const unsigned char& R = 0, const unsigned char& G = 0, const unsigned char& B = 0)
	{
		this->Set(R, G, B);
	}

	HOST_DEVICE void FromRGBf(const float& R, const float& G, const float& B)
	{
		this->SetR((unsigned char)(clamp2(R, 0.0f, 1.0f) * 255.0f));
		this->SetG((unsigned char)(clamp2(G, 0.0f, 1.0f) * 255.0f));
		this->SetB((unsigned char)(clamp2(B, 0.0f, 1.0f) * 255.0f));
	}

	HOST_DEVICE void FromXYZ(const float& X, const float& Y, const float& Z)
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

		this->SetR((unsigned char)(R * 255.0f));
		this->SetG((unsigned char)(G * 255.0f));
		this->SetB((unsigned char)(B * 255.0f));
	}

	HOST_DEVICE void SetBlack(void)
	{
		this->Set(0, 0, 0);
	}

	HOST_DEVICE void SetWhite(void)
	{
		this->Set(255, 255, 255);
	}
};

class ColorRGBAuc : public ColorRGBA<unsigned char>
{
public:
	HOST_DEVICE ColorRGBAuc(unsigned char RGBA)
	{
		this->Set(RGBA, RGBA, RGBA, RGBA);
	}

	HOST_DEVICE ColorRGBAuc(const unsigned char& R = 0, const unsigned char& G = 0, const unsigned char& B = 0, const unsigned char& A = 0)
	{
		this->Set(R, G, B, A);
	}

	HOST_DEVICE ColorRGBAuc(const ColorRGBuc& RGB)
	{
		this->SetR(RGB.GetR());
		this->SetG(RGB.GetG());
		this->SetB(RGB.GetB());
	}

	HOST_DEVICE void FromRGBAf(const float& R, const float& G, const float& B, const float& A)
	{
		this->SetR((unsigned char)(clamp2(R, 0.0f, 1.0f) * 255.0f));
		this->SetG((unsigned char)(clamp2(G, 0.0f, 1.0f) * 255.0f));
		this->SetB((unsigned char)(clamp2(B, 0.0f, 1.0f) * 255.0f));
		this->SetA((unsigned char)(clamp2(A, 0.0f, 1.0f) * 255.0f));
	}

	HOST_DEVICE void FromXYZ(const float& X, const float& Y, const float& Z)
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

		this->SetR((unsigned char)(R * 255.0f));
		this->SetG((unsigned char)(G * 255.0f));
		this->SetB((unsigned char)(B * 255.0f));
	}

	HOST_DEVICE void SetBlack(void)
	{
		this->Set(0, 0, 0, 0);
	}

	HOST_DEVICE void SetWhite(void)
	{
		this->Set(255, 255, 255, 0);
	}
};

template <class T> inline HOST_DEVICE ColorRGBAuc operator / (const ColorRGBAuc& V, const T& F)				{ return ColorRGBAuc((float)V[0] / F, (float)V[1] / F, (float)V[2] / F, V[3]);						};

HOST_DEVICE inline ColorRGBAuc operator * (const float& F, const ColorRGBAuc& XYZ)
{
	return ColorRGBAuc(XYZ[0] * F, XYZ[1] * F, XYZ[2] * F);
}

HOST_DEVICE inline ColorRGBAuc operator + (const ColorRGBAuc& A, const ColorRGBAuc& B)
{
	return ColorRGBAuc(A[0] + B[0], A[1] + B[1], A[2] + B[2], A[3] + B[3]);
}

class ColorXYZf : public Vec3f
{
public:
	HOST_DEVICE ColorXYZf(float V = 0.0f)
	{
		this->Set(V, V, V);
	}

	HOST_DEVICE ColorXYZf(float X, float Y, float Z)
	{
		this->Set(X, Y, Z);
	}

	HOST_DEVICE void Set(float X, float Y, float Z)
	{
		this->SetX(X);
		this->SetY(Y);
		this->SetZ(Z);
	}

	HOST_DEVICE ColorXYZf(float V[3])
	{
		this->Set(V[0], V[1], V[2]);
	}

	HOST_DEVICE float GetX(void) const
	{
		return this->D[0];
	}

	HOST_DEVICE void SetX(float X)
	{
		this->D[0] = X;
	}

	HOST_DEVICE float GetY(void) const
	{
		return this->D[1];
	}

	HOST_DEVICE void SetY(float Y)
	{
		this->D[1] = Y;
	}

	HOST_DEVICE float GetZ(void) const
	{
		return this->D[2];
	}

	HOST_DEVICE void SetZ(float Z)
	{
		this->D[2] = Z;
	}

	HOST_DEVICE ColorXYZf operator + (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.D[i] += XYZ[i];

		return Result;
	}
	
	HOST_DEVICE ColorXYZf operator - (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.D[i] -= XYZ[i];

		return Result;
	}

	HOST_DEVICE ColorXYZf operator / (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.D[i] /= XYZ[i];

		return Result;
	}

	HOST_DEVICE ColorXYZf operator * (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; i++)
			Result.D[i] *= XYZ[i];

		return Result;
	}

	HOST_DEVICE ColorXYZf& operator *= (const ColorXYZf& XYZ)
	{
		for (int i = 0; i < 3; i++)
			this->D[i] *= XYZ[i];

		return *this;
	}

	HOST_DEVICE ColorXYZf operator * (const float& F) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.D[i] *= F;

		return Result;
	}

	HOST_DEVICE ColorXYZf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; ++i)
			this->D[i] *= F;

		return *this;
	}

	HOST_DEVICE ColorXYZf operator / (const float& F) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.D[i] /= F;

		return Result;
	}

	HOST_DEVICE ColorXYZf& operator /= (float a)
	{
		for (int i = 0; i < 3; ++i)
			this->D[i] /= a;

		return *this;
	}

	HOST_DEVICE ColorXYZf& ColorXYZf::operator = (const ColorXYZf& Other)
	{
		for (int i = 0; i < 3; ++i)
			D[i] = Other[i];

		return *this;
	}

	HOST_DEVICE bool IsBlack() const
	{
		for (int i = 0; i < 3; ++i)
			if (D[i] != 0.0f)
				return false;

		return true;
	}

	HOST_DEVICE float Y() const
	{
		const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };

		float v = 0.0f;

		for (int i = 0; i < 3; i++)
			v += YWeight[i] * D[i];

		return v;
	}

	HOST_DEVICE void FromRGB(const float& R, const float& G, const float& B)
	{
		const float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
		const float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
		const float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

		D[0] = CoeffX[0] * R + CoeffX[1] * G + CoeffX[2] * B;
		D[1] = CoeffY[0] * R + CoeffY[1] * G + CoeffY[2] * B;
		D[2] = CoeffZ[0] * R + CoeffZ[1] * G + CoeffZ[2] * B;
	}
};

HOST_DEVICE inline ColorXYZf operator * (const float& F, const ColorXYZf& XYZ)
{
	return XYZ * F;
}

HOST_DEVICE inline ColorXYZf Lerp(const float& T, const ColorXYZf& C1, const ColorXYZf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorXYZf(OneMinusT * C1.GetX() + T * C2.GetX(), OneMinusT * C1.GetY() + T * C2.GetY(), OneMinusT * C1.GetZ() + T * C2.GetZ());
}

class ColorXYZAf : public Vec4f
{
public:
	HOST_DEVICE ColorXYZAf(float V = 0.0f)
	{
		Set(V, V, V, V);
	}

	HOST_DEVICE ColorXYZAf(ColorXYZf XYZ)
	{
		Set(XYZ.GetX(), XYZ.GetY(), XYZ.GetZ());
	}

	HOST_DEVICE ColorXYZAf(float X, float Y, float Z)
	{
		Set(X, Y, Z);
	}

	HOST_DEVICE ColorXYZAf(float X, float Y, float Z, float A)
	{
		Set(X, Y, Z, A);
	}
	
	HOST_DEVICE void Set(float X, float Y, float Z)
	{
		SetX(X);
		SetY(Y);
		SetZ(Z);
	}

	HOST_DEVICE void Set(float X, float Y, float Z, float A)
	{
		SetX(X);
		SetY(Y);
		SetZ(Z);
		SetA(A);
	}

	HOST_DEVICE float GetX(void) const
	{
		return D[0];
	}

	HOST_DEVICE void SetX(float X)
	{
		D[0] = X;
	}

	HOST_DEVICE float GetY(void) const
	{
		return D[1];
	}

	HOST_DEVICE void SetY(float Y)
	{
		D[1] = Y;
	}

	HOST_DEVICE float GetZ(void) const
	{
		return D[2];
	}

	HOST_DEVICE void SetZ(float Z)
	{
		D[2] = Z;
	}

	HOST_DEVICE float GetA(void) const
	{
		return D[3];
	}

	HOST_DEVICE void SetA(float A)
	{
		D[3] = A;
	}

	HOST_DEVICE ColorXYZAf& operator += (const ColorXYZAf& XYZ)
	{
		for (int i = 0; i < 4; ++i)
			D[i] += XYZ[i];

		return *this;
	}

	HOST_DEVICE ColorXYZAf operator + (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 4; ++i)
			Result.D[i] += XYZ[i];

		return Result;
	}

	HOST_DEVICE ColorXYZAf operator - (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 4; ++i)
			Result.D[i] -= XYZ[i];

		return Result;
	}

	
	HOST_DEVICE ColorXYZAf operator / (const ColorXYZAf& XYZA) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 4; ++i)
			Result.D[i] /= XYZA[i];

		return Result;
	}
	/**/

	HOST_DEVICE ColorXYZAf operator * (const ColorXYZAf& XYZA) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 4; ++i)
			Result.D[i] *= XYZA[i];

		return Result;
	}
/*
	HOST_DEVICE ColorXYZAf& operator *= (const ColorXYZAf& XYZ)
	{
		for (int i = 0; i < 3; ++i)
			D[i] *= XYZ[i];

		return *this;
	}

	HOST_DEVICE ColorXYZAf operator * (const float& F) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.D[i] *= F;

		return Result;
	}

	HOST_DEVICE ColorXYZAf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; ++i)
			D[i] *= F;

		return *this;
	}

	HOST_DEVICE ColorXYZAf operator / (const float& F) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.D[i] /= F;

		return Result;
	}

	
	HOST_DEVICE ColorXYZAf& operator /= (float a)
	{
		for (int i = 0; i < 3; ++i)
			D[i] /= a;

		return *this;
	}
	

	HOST_DEVICE ColorXYZAf& ColorXYZAf::operator = (const ColorXYZAf& Other)
	{
		for (int i = 0; i < 4; ++i)
			D[i] = Other[i];

		return *this;
	}
*/
	HOST_DEVICE bool IsBlack() const
	{
		for (int i = 0; i < 3; ++i)
			if (D[i] != 0.0f)
				return false;

		return true;
	}

	HOST_DEVICE float Y() const
	{
		const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };

		float v = 0.0f;

		for (int i = 0; i < 3; i++)
			v += YWeight[i] * D[i];

		return v;
	}

	HOST_DEVICE void FromRGB(const float& R, const float& G, const float& B)
	{
		const float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
		const float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
		const float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

		D[0] = CoeffX[0] * R + CoeffX[1] * G + CoeffX[2] * B;
		D[1] = CoeffY[0] * R + CoeffY[1] * G + CoeffY[2] * B;
		D[2] = CoeffZ[0] * R + CoeffZ[1] * G + CoeffZ[2] * B;
	}
};

HOST_DEVICE inline ColorXYZAf operator * (const float& F, const ColorXYZAf& XYZA)
{
	return XYZA * F;
}

HOST_DEVICE inline ColorXYZAf Lerp(const float& T, const ColorXYZAf& C1, const ColorXYZAf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorXYZAf(OneMinusT * C1.GetX() + T * C2.GetX(), OneMinusT * C1.GetY() + T * C2.GetY(), OneMinusT * C1.GetZ() + T * C2.GetZ(), OneMinusT * C1.GetA() + T * C2.GetA());
}

class ColorRGBf : public Vec3<float>
{
public:
	HOST_DEVICE ColorRGBf(void)
	{
	}

	HOST_DEVICE ColorRGBf(const float& R, const float& G, const float& B)
	{
		Set(R, G, B);
	}

	HOST_DEVICE ColorRGBf(const float& RGB)
	{
		Set(RGB, RGB, RGB);
	}

	HOST_DEVICE void Set(const float& R, const float& G, const float& B)
	{
		SetR(R);
		SetG(G);
		SetB(B);
	}

	HOST_DEVICE float GetR(void) const
	{
		return D[0];
	}

	HOST_DEVICE void SetR(const float& R)
	{
		D[0] = R;
	}

	HOST_DEVICE float GetG(void) const
	{
		return D[1];
	}

	HOST_DEVICE void SetG(const float& G)
	{
		D[1] = G;
	}

	HOST_DEVICE float GetB(void) const
	{
		return D[2];
	}

	HOST_DEVICE void SetB(const float& B)
	{
		D[2] = B;
	}

	HOST_DEVICE void SetBlack(void)
	{
		Set(0.0f, 0.0f, 0.0f);
	}

	HOST_DEVICE void SetWhite(void)
	{
		Set(1.0f, 1.0f, 1.0f);
	}

	HOST_DEVICE ColorRGBf& operator = (const ColorRGBf& Other)			
	{
		for (int i = 0; i < 3; i++)
			D[i] = Other[i];

		return *this;
	}

	HOST_DEVICE ColorRGBf& operator += (ColorRGBf& Other)		
	{
		for (int i = 0; i < 3; i++)
			D[i] += Other[i];

		return *this;
	}

	HOST_DEVICE ColorRGBf operator * (const float& F) const
	{
		return ColorRGBf(D[0] * F, D[1] * F, D[2] * F);
	}

	HOST_DEVICE ColorRGBf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; i++)
			D[i] *= F;

		return *this;
	}

	HOST_DEVICE ColorRGBf operator / (const float& F) const
	{
		const float Inv = 1.0f / F;
		return ColorRGBf(D[0] * Inv, D[1] * Inv, D[2] * Inv);
	}

	HOST_DEVICE ColorRGBf& operator /= (const float& F)
	{
		const float Inv = 1.0f / F;
		
		for (int i = 0; i < 3; i++)
			D[i] *= Inv;

		return *this;
	}

	HOST_DEVICE float operator[](int i) const
	{
		return D[i];
	}

	HOST_DEVICE float operator[](int i)
	{
		return D[i];
	}

	HOST_DEVICE bool Black(void)
	{
		for (int i = 0; i < 3; i++)
			if (D[i] != 0.0f)
				return false;

		return true;
	}

	HOST_DEVICE ColorRGBf Pow(const float& E)
	{
		return ColorRGBf(powf(D[0], E), powf(D[1], E), powf(D[2], E));
	}

	HOST_DEVICE void FromXYZ(const float& X, const float& Y, const float& Z)
	{
		const float rWeight[3] = { 3.240479f, -1.537150f, -0.498535f };
		const float gWeight[3] = {-0.969256f,  1.875991f,  0.041556f };
		const float bWeight[3] = { 0.055648f, -0.204043f,  1.057311f };

		Set(rWeight[0] * X + rWeight[1] * Y + rWeight[2] * Z, gWeight[0] * X + gWeight[1] * Y + gWeight[2] * Z, bWeight[0] * X + bWeight[1] * Y + bWeight[2] * Z);
	}

	HOST_DEVICE void ToneMap(float InvExposure)
	{
		this->D[0] = 1.0f - expf(-(this->D[0] * InvExposure));
		this->D[1] = 1.0f - expf(-(this->D[1] * InvExposure));
		this->D[2] = 1.0f - expf(-(this->D[2] * InvExposure));

		this->Clamp(0.0f, 1.0f);
	}
};

HOST_DEVICE inline ColorRGBf Lerp(const float& T, const ColorRGBf& C1, const ColorRGBf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorRGBf(OneMinusT * C1.GetR() + T * C2.GetR(), OneMinusT * C1.GetG() + T * C2.GetG(), OneMinusT * C1.GetB() + T * C2.GetB());
}

HOST_DEVICE ColorRGBAuc Lerp(const ColorRGBAuc& A, const ColorRGBAuc& B, const float& T)
{
	ColorRGBAuc Result;

	for (int i = 0; i < 3; i++)
		Result[i] = (1.0f - T) * (float)A[i] + T * (float)B[i];

	return Result;
}

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