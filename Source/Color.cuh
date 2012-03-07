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

HD inline void XYZToRGB(const float xyz[3], float rgb[3])
{
	rgb[0] =  3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2];
	rgb[1] = -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2];
	rgb[2] =  0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2];
}


HD inline void RGBToXYZ(const float rgb[3], float xyz[3])
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
	HD ColorRGB(void)
	{
		this->SetBlack();
	}

	HD ColorRGB(T RGB)
	{
		this->Set(RGB, RGB, RGB);
	}

	HD ColorRGB(T R, T G, T B)
	{
		this->Set(R, G, B);
	}

	HD void Set(T R, T G, T B)
	{
		this->SetR(R);
		this->SetG(G);
		this->SetB(B);
	}

	HD T GetR(void) const
	{
		return this->m_D[0];
	}

	HD void SetR(const T& R)
	{
		this->m_D[0] = R;
	}

	HD T GetG(void) const
	{
		return this->m_D[1];
	}

	HD void SetG(const T& G)
	{
		this->m_D[1] = G;
	}

	HD T GetB(void) const
	{
		return this->m_D[2];
	}

	HD void SetB(const T& B)
	{
		this->m_D[2] = B;
	}

	HD void SetBlack(void)
	{
		this->Set(T(), T(), T());
	}
};

template <class T>
class ColorRGBA : public ColorRGB<T, 4>
{
public:
	HD ColorRGBA(void)
	{
		this->SetBlack();
	}

	HD ColorRGBA(T RGBA)
	{
		this->Set(RGBA, RGBA, RGBA, RGBA);
	}

	HD void Set(T R, T G, T B, T A)
	{
		this->SetR(R);
		this->SetG(G);
		this->SetB(B);
		this->SetA(A);
	}

	HD T GetA(void) const
	{
		return this->m_D[3];
	}

	HD void SetA(const T& A)
	{
		this->m_D[3] = A;
	}
};

class ColorRGBuc : public ColorRGB<unsigned char, 3>
{
public:
	HD ColorRGBuc(unsigned char RGB)
	{
		this->Set(RGB, RGB, RGB);
	}

	HD ColorRGBuc(const unsigned char& R = 0, const unsigned char& G = 0, const unsigned char& B = 0)
	{
		this->Set(R, G, B);
	}

	HD void FromRGBf(const float& R, const float& G, const float& B)
	{
		this->SetR(clamp2(R, 0.0f, 1.0f) * 255.0f);
		this->SetG(clamp2(G, 0.0f, 1.0f) * 255.0f);
		this->SetB(clamp2(B, 0.0f, 1.0f) * 255.0f);
	}

	HD void FromXYZ(const float& X, const float& Y, const float& Z)
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

	HD void SetBlack(void)
	{
		this->Set(0, 0, 0);
	}

	HD void SetWhite(void)
	{
		this->Set(255, 255, 255);
	}
};

class ColorRGBAuc : public ColorRGBA<unsigned char>
{
public:
	HD ColorRGBAuc(unsigned char RGBA)
	{
		this->Set(RGBA, RGBA, RGBA, RGBA);
	}

	HD ColorRGBAuc(const unsigned char& R = 0, const unsigned char& G = 0, const unsigned char& B = 0, const unsigned char& A = 0)
	{
		this->Set(R, G, B, A);
	}

	HD ColorRGBAuc(const ColorRGBuc& RGB)
	{
		this->SetR(RGB.GetR());
		this->SetG(RGB.GetG());
		this->SetB(RGB.GetB());
	}

	HD void FromRGBAf(const float& R, const float& G, const float& B, const float& A)
	{
		this->SetR(clamp2(R, 0.0f, 1.0f) * 255.0f);
		this->SetG(clamp2(G, 0.0f, 1.0f) * 255.0f);
		this->SetB(clamp2(B, 0.0f, 1.0f) * 255.0f);
		this->SetA(clamp2(A, 0.0f, 1.0f) * 255.0f);
	}

	HD void FromXYZ(const float& X, const float& Y, const float& Z)
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

	HD void SetBlack(void)
	{
		this->Set(0, 0, 0, 0);
	}

	HD void SetWhite(void)
	{
		this->Set(255, 255, 255, 0);
	}
};

class ColorXYZf : public Vec3f
{
public:
	HD ColorXYZf(float V = 0.0f)
	{
		this->Set(V, V, V);
	}

	HD ColorXYZf(float X, float Y, float Z)
	{
		this->Set(X, Y, Z);
	}

	HD void Set(float X, float Y, float Z)
	{
		this->SetX(X);
		this->SetY(Y);
		this->SetZ(Z);
	}

	HD ColorXYZf(float V[3])
	{
		this->Set(V[0], V[1], V[2]);
	}

	HD float GetX(void) const
	{
		return this->m_D[0];
	}

	HD void SetX(float X)
	{
		this->m_D[0] = X;
	}

	HD float GetY(void) const
	{
		return this->m_D[1];
	}

	HD void SetY(float Y)
	{
		this->m_D[1] = Y;
	}

	HD float GetZ(void) const
	{
		return this->m_D[2];
	}

	HD void SetZ(float Z)
	{
		this->m_D[2] = Z;
	}

	HD ColorXYZf operator + (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] += XYZ[i];

		return Result;
	}
	
	HD ColorXYZf operator - (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] -= XYZ[i];

		return Result;
	}

	HD ColorXYZf operator / (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] /= XYZ[i];

		return Result;
	}

	HD ColorXYZf operator * (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; i++)
			Result.m_D[i] *= XYZ[i];

		return Result;
	}

	HD ColorXYZf& operator *= (const ColorXYZf& XYZ)
	{
		for (int i = 0; i < 3; i++)
			this->m_D[i] *= XYZ[i];

		return *this;
	}

	HD ColorXYZf operator * (const float& F) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] *= F;

		return Result;
	}

	HD ColorXYZf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; ++i)
			this->m_D[i] *= F;

		return *this;
	}

	HD ColorXYZf operator / (const float& F) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] /= F;

		return Result;
	}

	HD ColorXYZf& operator /= (float a)
	{
		for (int i = 0; i < 3; ++i)
			this->m_D[i] /= a;

		return *this;
	}

	HD ColorXYZf& ColorXYZf::operator = (const ColorXYZf& Other)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] = Other[i];

		return *this;
	}

	HD bool IsBlack() const
	{
		for (int i = 0; i < 3; ++i)
			if (m_D[i] != 0.0f)
				return false;

		return true;
	}

	HD float Y() const
	{
		float v = 0.0f;

		for (int i = 0; i < 3; i++)
			v += YWeight[i] * m_D[i];

		return v;
	}

	HD void FromRGB(const float& R, const float& G, const float& B)
	{
		const float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
		const float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
		const float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

		m_D[0] = CoeffX[0] * R + CoeffX[1] * G + CoeffX[2] * B;
		m_D[1] = CoeffY[0] * R + CoeffY[1] * G + CoeffY[2] * B;
		m_D[2] = CoeffZ[0] * R + CoeffZ[1] * G + CoeffZ[2] * B;
	}
};

HD inline ColorXYZf operator * (const float& F, const ColorXYZf& XYZ)
{
	return XYZ * F;
}

HD inline ColorXYZf Lerp(const float& T, const ColorXYZf& C1, const ColorXYZf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorXYZf(OneMinusT * C1.GetX() + T * C2.GetX(), OneMinusT * C1.GetY() + T * C2.GetY(), OneMinusT * C1.GetZ() + T * C2.GetZ());
}

class ColorXYZAf : public Vec4f
{
public:
	HD ColorXYZAf(float V = 0.0f)
	{
		Set(V, V, V, V);
	}

	HD ColorXYZAf(ColorXYZf XYZ)
	{
		Set(XYZ.GetX(), XYZ.GetY(), XYZ.GetZ());
	}

	HD ColorXYZAf(float X, float Y, float Z)
	{
		Set(X, Y, Z);
	}

	HD ColorXYZAf(float X, float Y, float Z, float A)
	{
		Set(X, Y, Z, A);
	}
	
	HD void Set(float X, float Y, float Z)
	{
		SetX(X);
		SetY(Y);
		SetZ(Z);
	}

	HD void Set(float X, float Y, float Z, float A)
	{
		SetX(X);
		SetY(Y);
		SetZ(Z);
		SetA(A);
	}

	HD float GetX(void) const
	{
		return m_D[0];
	}

	HD void SetX(float X)
	{
		m_D[0] = X;
	}

	HD float GetY(void) const
	{
		return m_D[1];
	}

	HD void SetY(float Y)
	{
		m_D[1] = Y;
	}

	HD float GetZ(void) const
	{
		return m_D[2];
	}

	HD void SetZ(float Z)
	{
		m_D[2] = Z;
	}

	HD float GetA(void) const
	{
		return m_D[3];
	}

	HD void SetA(float A)
	{
		m_D[3] = A;
	}

	HD ColorXYZAf& operator += (const ColorXYZAf& XYZ)
	{
		for (int i = 0; i < 4; ++i)
			m_D[i] += XYZ[i];

		return *this;
	}

	HD ColorXYZAf operator + (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 4; ++i)
			Result.m_D[i] += XYZ[i];

		return Result;
	}

	HD ColorXYZAf operator - (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 4; ++i)
			Result.m_D[i] -= XYZ[i];

		return Result;
	}

	
	HD ColorXYZAf operator / (const ColorXYZAf& XYZA) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 4; ++i)
			Result.m_D[i] /= XYZA[i];

		return Result;
	}
	/**/

	HD ColorXYZAf operator * (const ColorXYZAf& XYZA) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 4; ++i)
			Result.m_D[i] *= XYZA[i];

		return Result;
	}
/*
	HD ColorXYZAf& operator *= (const ColorXYZAf& XYZ)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] *= XYZ[i];

		return *this;
	}

	HD ColorXYZAf operator * (const float& F) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] *= F;

		return Result;
	}

	HD ColorXYZAf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] *= F;

		return *this;
	}

	HD ColorXYZAf operator / (const float& F) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] /= F;

		return Result;
	}

	
	HD ColorXYZAf& operator /= (float a)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] /= a;

		return *this;
	}
	

	HD ColorXYZAf& ColorXYZAf::operator = (const ColorXYZAf& Other)
	{
		for (int i = 0; i < 4; ++i)
			m_D[i] = Other[i];

		return *this;
	}
*/
	HD bool IsBlack() const
	{
		for (int i = 0; i < 3; ++i)
			if (m_D[i] != 0.0f)
				return false;

		return true;
	}

	HD float Y() const
	{
		float v = 0.0f;

		for (int i = 0; i < 3; i++)
			v += YWeight[i] * m_D[i];

		return v;
	}

	HD void FromRGB(const float& R, const float& G, const float& B)
	{
		const float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
		const float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
		const float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

		m_D[0] = CoeffX[0] * R + CoeffX[1] * G + CoeffX[2] * B;
		m_D[1] = CoeffY[0] * R + CoeffY[1] * G + CoeffY[2] * B;
		m_D[2] = CoeffZ[0] * R + CoeffZ[1] * G + CoeffZ[2] * B;
	}
};

HD inline ColorXYZAf operator * (const float& F, const ColorXYZAf& XYZA)
{
	return XYZA * F;
}

HD inline ColorXYZAf Lerp(const float& T, const ColorXYZAf& C1, const ColorXYZAf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorXYZAf(OneMinusT * C1.GetX() + T * C2.GetX(), OneMinusT * C1.GetY() + T * C2.GetY(), OneMinusT * C1.GetZ() + T * C2.GetZ(), OneMinusT * C1.GetA() + T * C2.GetA());
}

class ColorRGBf : public Vec3<float>
{
public:
	HD ColorRGBf(void)
	{
	}

	HD ColorRGBf(const float& R, const float& G, const float& B)
	{
		Set(R, G, B);
	}

	HD ColorRGBf(const float& RGB)
	{
		Set(RGB, RGB, RGB);
	}

	HD void Set(const float& R, const float& G, const float& B)
	{
		SetR(R);
		SetG(G);
		SetB(B);
	}

	HD float GetR(void) const
	{
		return m_D[0];
	}

	HD void SetR(const float& R)
	{
		m_D[0] = R;
	}

	HD float GetG(void) const
	{
		return m_D[1];
	}

	HD void SetG(const float& G)
	{
		m_D[1] = G;
	}

	HD float GetB(void) const
	{
		return m_D[2];
	}

	HD void SetB(const float& B)
	{
		m_D[2] = B;
	}

	HD void SetBlack(void)
	{
		Set(0.0f, 0.0f, 0.0f);
	}

	HD void SetWhite(void)
	{
		Set(1.0f, 1.0f, 1.0f);
	}

	HD ColorRGBf& operator = (const ColorRGBf& Other)			
	{
		for (int i = 0; i < 3; i++)
			m_D[i] = Other[i];

		return *this;
	}

	HD ColorRGBf& operator += (ColorRGBf& Other)		
	{
		for (int i = 0; i < 3; i++)
			m_D[i] += Other[i];

		return *this;
	}

	HD ColorRGBf operator * (const float& F) const
	{
		return ColorRGBf(m_D[0] * F, m_D[1] * F, m_D[2] * F);
	}

	HD ColorRGBf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; i++)
			m_D[i] *= F;

		return *this;
	}

	HD ColorRGBf operator / (const float& F) const
	{
		const float Inv = 1.0f / F;
		return ColorRGBf(m_D[0] * Inv, m_D[1] * Inv, m_D[2] * Inv);
	}

	HD ColorRGBf& operator /= (const float& F)
	{
		const float Inv = 1.0f / F;
		
		for (int i = 0; i < 3; i++)
			m_D[i] *= Inv;

		return *this;
	}

	HD float operator[](int i) const
	{
		return m_D[i];
	}

	HD float operator[](int i)
	{
		return m_D[i];
	}

	HD bool Black(void)
	{
		for (int i = 0; i < 3; i++)
			if (m_D[i] != 0.0f)
				return false;

		return true;
	}

	HD ColorRGBf Pow(const float& E)
	{
		return ColorRGBf(powf(m_D[0], E), powf(m_D[1], E), powf(m_D[2], E));
	}

	HD void FromXYZ(const float& X, const float& Y, const float& Z)
	{
		const float rWeight[3] = { 3.240479f, -1.537150f, -0.498535f };
		const float gWeight[3] = {-0.969256f,  1.875991f,  0.041556f };
		const float bWeight[3] = { 0.055648f, -0.204043f,  1.057311f };

		Set(rWeight[0] * X + rWeight[1] * Y + rWeight[2] * Z, gWeight[0] * X + gWeight[1] * Y + gWeight[2] * Z, bWeight[0] * X + bWeight[1] * Y + bWeight[2] * Z);
	}

	HD void ToneMap(float InvExposure)
	{
		this->m_D[0] = 1.0f - expf(-(this->m_D[0] * InvExposure));
		this->m_D[1] = 1.0f - expf(-(this->m_D[1] * InvExposure));
		this->m_D[2] = 1.0f - expf(-(this->m_D[2] * InvExposure));

		this->Clamp(0.0f, 1.0f);
	}
};

HD inline ColorRGBf Lerp(const float& T, const ColorRGBf& C1, const ColorRGBf& C2)
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