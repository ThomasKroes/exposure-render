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

HOD inline void XYZToRGB(const float xyz[3], float rgb[3])
{
	rgb[0] =  3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2];
	rgb[1] = -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2];
	rgb[2] =  0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2];
}


HOD inline void RGBToXYZ(const float rgb[3], float xyz[3])
{
	xyz[0] = 0.412453f*rgb[0] + 0.357580f*rgb[1] + 0.180423f*rgb[2];
	xyz[1] = 0.212671f*rgb[0] + 0.715160f*rgb[1] + 0.072169f*rgb[2];
	xyz[2] = 0.019334f*rgb[0] + 0.119193f*rgb[1] + 0.950227f*rgb[2];
}

CD static float YWeight[3] =
{
	0.212671f, 0.715160f, 0.072169f
};

template <class T, int Size>
class ColorRGB : public Vec<T, Size>
{
public:
	HOD ColorRGB(void)
	{
		this->SetBlack();
	}

	HOD ColorRGB(T RGB)
	{
		this->Set(RGB, RGB, RGB);
	}

	HOD ColorRGB(T R, T G, T B)
	{
		this->Set(R, G, B);
	}

	HOD void Set(T R, T G, T B)
	{
		this->SetR(R);
		this->SetG(G);
		this->SetB(B);
	}

	HOD T GetR(void) const
	{
		return this->m_D[0];
	}

	HOD void SetR(const T& R)
	{
		this->m_D[0] = R;
	}

	HOD T GetG(void) const
	{
		return this->m_D[1];
	}

	HOD void SetG(const T& G)
	{
		this->m_D[1] = G;
	}

	HOD T GetB(void) const
	{
		return this->m_D[2];
	}

	HOD void SetB(const T& B)
	{
		this->m_D[2] = B;
	}

	HOD void SetBlack(void)
	{
		this->Set(T(), T(), T());
	}
};

template <class T>
class ColorRGBA : public ColorRGB<T, 4>
{
public:
	HOD ColorRGBA(void)
	{
		this->SetBlack();
	}

	HOD ColorRGBA(T RGBA)
	{
		this->Set(RGBA, RGBA, RGBA, RGBA);
	}

	HOD void Set(T R, T G, T B, T A)
	{
		this->SetR(R);
		this->SetG(G);
		this->SetB(B);
		this->SetA(A);
	}

	HOD T GetA(void) const
	{
		return this->m_D[3];
	}

	HOD void SetA(const T& A)
	{
		this->m_D[3] = A;
	}
};

class ColorRGBuc : public ColorRGB<unsigned char, 3>
{
public:
	HOD ColorRGBuc(unsigned char RGB)
	{
		this->Set(RGB, RGB, RGB);
	}

	HOD ColorRGBuc(const unsigned char& R = 0, const unsigned char& G = 0, const unsigned char& B = 0)
	{
		this->Set(R, G, B);
	}

	HOD void FromRGBf(const float& R, const float& G, const float& B)
	{
		this->SetR(clamp2(R, 0.0f, 1.0f) * 255.0f);
		this->SetG(clamp2(G, 0.0f, 1.0f) * 255.0f);
		this->SetB(clamp2(B, 0.0f, 1.0f) * 255.0f);
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

		this->SetR((unsigned char)(R * 255.0f));
		this->SetG((unsigned char)(G * 255.0f));
		this->SetB((unsigned char)(B * 255.0f));
	}

	HOD void SetBlack(void)
	{
		this->Set(0, 0, 0);
	}

	HOD void SetWhite(void)
	{
		this->Set(255, 255, 255);
	}
};

class ColorRGBAuc : public ColorRGBA<unsigned char>
{
public:
	HOD ColorRGBAuc(unsigned char RGBA)
	{
		this->Set(RGBA, RGBA, RGBA, RGBA);
	}

	HOD ColorRGBAuc(const unsigned char& R = 0, const unsigned char& G = 0, const unsigned char& B = 0, const unsigned char& A = 0)
	{
		this->Set(R, G, B, A);
	}

	HOD ColorRGBAuc(const ColorRGBuc& RGB)
	{
		this->SetR(RGB.GetR());
		this->SetG(RGB.GetG());
		this->SetB(RGB.GetB());
	}

	HOD void FromRGBAf(const float& R, const float& G, const float& B, const float& A)
	{
		this->SetR(clamp2(R, 0.0f, 1.0f) * 255.0f);
		this->SetG(clamp2(G, 0.0f, 1.0f) * 255.0f);
		this->SetB(clamp2(B, 0.0f, 1.0f) * 255.0f);
		this->SetA(clamp2(A, 0.0f, 1.0f) * 255.0f);
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

		this->SetR((unsigned char)(R * 255.0f));
		this->SetG((unsigned char)(G * 255.0f));
		this->SetB((unsigned char)(B * 255.0f));
	}

	HOD void SetBlack(void)
	{
		this->Set(0, 0, 0, 0);
	}

	HOD void SetWhite(void)
	{
		this->Set(255, 255, 255, 0);
	}
};

class ColorXYZf : public Vec3f
{
public:
	HOD ColorXYZf(float V = 0.0f)
	{
		this->Set(V, V, V);
	}

	HOD ColorXYZf(float X, float Y, float Z)
	{
		this->Set(X, Y, Z);
	}

	HOD void Set(float X, float Y, float Z)
	{
		this->SetX(X);
		this->SetY(Y);
		this->SetZ(Z);
	}

	HOD ColorXYZf(float V[3])
	{
		this->Set(V[0], V[1], V[2]);
	}

	HOD float GetX(void) const
	{
		return this->m_D[0];
	}

	HOD void SetX(float X)
	{
		this->m_D[0] = X;
	}

	HOD float GetY(void) const
	{
		return this->m_D[1];
	}

	HOD void SetY(float Y)
	{
		this->m_D[1] = Y;
	}

	HOD float GetZ(void) const
	{
		return this->m_D[2];
	}

	HOD void SetZ(float Z)
	{
		this->m_D[2] = Z;
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
			this->m_D[i] *= XYZ[i];

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
			this->m_D[i] *= F;

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
			this->m_D[i] /= a;

		return *this;
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

class ColorXYZAf : public Vec4f
{
public:
	HOD ColorXYZAf(float V = 0.0f)
	{
		Set(V, V, V, V);
	}

	HOD ColorXYZAf(ColorXYZf XYZ)
	{
		Set(XYZ.GetX(), XYZ.GetY(), XYZ.GetZ());
	}

	HOD ColorXYZAf(float X, float Y, float Z)
	{
		Set(X, Y, Z);
	}

	HOD ColorXYZAf(float X, float Y, float Z, float A)
	{
		Set(X, Y, Z, A);
	}
	
	HOD void Set(float X, float Y, float Z)
	{
		SetX(X);
		SetY(Y);
		SetZ(Z);
	}

	HOD void Set(float X, float Y, float Z, float A)
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

	HOD void SetX(float X)
	{
		m_D[0] = X;
	}

	HOD float GetY(void) const
	{
		return m_D[1];
	}

	HOD void SetY(float Y)
	{
		m_D[1] = Y;
	}

	HOD float GetZ(void) const
	{
		return m_D[2];
	}

	HOD void SetZ(float Z)
	{
		m_D[2] = Z;
	}

	HOD float GetA(void) const
	{
		return m_D[3];
	}

	HOD void SetA(float A)
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

class ColorRGBf : public Vec3<float>
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

	HOD void ToneMap(float InvExposure)
	{
		this->m_D[0] = 1.0f - expf(-(this->m_D[0] * InvExposure));
		this->m_D[1] = 1.0f - expf(-(this->m_D[1] * InvExposure));
		this->m_D[2] = 1.0f - expf(-(this->m_D[2] * InvExposure));

		this->Clamp(0.0f, 1.0f);
	}
};

HOD inline ColorRGBf Lerp(const float& T, const ColorRGBf& C1, const ColorRGBf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorRGBf(OneMinusT * C1.GetR() + T * C2.GetR(), OneMinusT * C1.GetG() + T * C2.GetG(), OneMinusT * C1.GetB() + T * C2.GetB());
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