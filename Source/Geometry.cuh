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

#include "Defines.h"

#include <algorithm>
#include <math.h>

#ifdef _EXPORTING
	#define EXPOSURE_RENDER_DLL    __declspec(dllexport)
#else
	#define EXPOSURE_RENDER_DLL    __declspec(dllimport)
#endif

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

DEV inline float clamp2(float v, float a, float b)
{
	return max(a, min(v, b));
}

DEV  inline void swap(int& a, int& b)
{
	int t = a; a = b; b = t;
}

DEV  inline void swap(float& a, float& b)
{
	float t = a; a = b; b = t;
}

DEV inline void Swap(float* pF1, float* pF2)
{
	const float TempFloat = *pF1;

	*pF1 = *pF2;
	*pF2 = TempFloat;
}

DEV inline void Swap(float& F1, float& F2)
{
	const float TempFloat = F1;

	F1 = F2;
	F2 = TempFloat;
}

DEV inline void Swap(int* pI1, int* pI2)
{
	const int TempInt = *pI1;

	*pI1 = *pI2;
	*pI2 = TempInt;
}

DEV inline void Swap(int& I1, int& I2)
{
	const int TempInt = I1;

	I1 = I2;
	I2 = TempInt;

}
class CColorXyz;

DEV inline void XYZToRGB(const float xyz[3], float rgb[3])
{
	rgb[0] =  3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2];
	rgb[1] = -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2];
	rgb[2] =  0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2];
}


DEV inline void RGBToXYZ(const float rgb[3], float xyz[3])
{
	xyz[0] = 0.412453f*rgb[0] + 0.357580f*rgb[1] + 0.180423f*rgb[2];
	xyz[1] = 0.212671f*rgb[0] + 0.715160f*rgb[1] + 0.072169f*rgb[2];
	xyz[2] = 0.019334f*rgb[0] + 0.119193f*rgb[1] + 0.950227f*rgb[2];
}

CD static float YWeight[3] =
{
	0.212671f, 0.715160f, 0.072169f
};

class EXPOSURE_RENDER_DLL Vec2f
{
public:
	DEV Vec2f(void)
	{
		this->x = 0.0f;
		this->y = 0.0f;
	}

	DEV Vec2f(const float& x, const float& y)
	{
		this->x = x;
		this->y = y;
	}

	DEV Vec2f(const float& xy)
	{
		this->x = xy;
		this->y = xy;
	}

	DEV Vec2f(const Vec2f& v)
	{
		this->x = v.x;
		this->y = v.y;
	}

	DEV float operator[](int i) const
	{
		return (&x)[i];
	}

	DEV float& operator[](int i)
	{
		return (&x)[i];
	}

	DEV Vec2f& operator = (const Vec2f& v)
	{
		x = v.x; 
		y = v.y; 

		return *this;
	}

	DEV Vec2f operator + (const Vec2f& v) const
	{
		return Vec2f(x + v.x, y + v.y);
	}

	DEV Vec2f& operator += (const Vec2f& v)
	{
		x += v.x; y += v.y;
		return *this;
	}

	DEV Vec2f operator - (const Vec2f& v) const
	{
		return Vec2f(x - v.x, y - v.y);
	}

	DEV Vec2f& operator -= (const Vec2f& v)
	{
		x -= v.x; y -= v.y;
		return *this;
	}

	DEV Vec2f operator * (float f) const
	{
		return Vec2f(x * f, y * f);
	}

	DEV Vec2f& operator *= (float f)
	{
		x *= f; 
		y *= f; 

		return *this;
	}

	DEV bool operator < (const Vec2f& V) const
	{
		return V.x < x && V.y < y;
	}

	DEV bool operator > (const Vec2f& V) const
	{
		return V.x > x && V.y > y;
	}

	DEV bool operator == (const Vec2f& V) const
	{
		return V.x == x && V.y == y;
	}

	DEV float LengthSquared(void) const
	{
		return x * x + y * y;
	}

	DEV float Length(void) const
	{
		return sqrtf(LengthSquared());
	}

	float x, y;
};

class EXPOSURE_RENDER_DLL  Vec2i
{
public:
	DEV Vec2i(void)
	{
		this->x = 0;
		this->y = 0;
	}

	DEV Vec2i(const int& x, const int& y)
	{
		this->x = x;
		this->y = y;
	}

	DEV Vec2i(int& x, int& y)
	{
		this->x = x;
		this->y = y;
	}

	DEV Vec2i(const Vec2f& V)
	{
		this->x = (int)V.x;
		this->y = (int)V.y;
	}

	DEV Vec2i(const int& xy)
	{
		this->x = xy;
		this->y = xy;
	}

	DEV int operator[](int i) const
	{
		return (&x)[i];
	}

	DEV int& operator[](int i)
	{
		return (&x)[i];
	}

	DEV Vec2i& operator = (const Vec2i& v)
	{
		x = v.x; 
		y = v.y; 

		return *this;
	}

	DEV bool operator < (const Vec2i& V) const
	{
		return V.x < x && V.y < y;
	}

	DEV bool operator > (const Vec2i& V) const
	{
		return V.x > x && V.y > y;
	}

	DEV bool operator == (const Vec2i& V) const
	{
		return V.x == x && V.y == y;
	}

	int x, y;
};

class EXPOSURE_RENDER_DLL Vec3i
{
public:
	DEV Vec3i(void)
	{
		this->x = 0;
		this->y = 0;
		this->z = 0;
	}

	DEV Vec3i(const int& x, const int& y, const int& z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	DEV Vec3i(const int& xyz)
	{
		this->x = xyz;
		this->y = xyz;
		this->z = xyz;
	}

	DEV int operator[](int i) const
	{
		return (&x)[i];
	}

	DEV int& operator[](int i)
	{
		return (&x)[i];
	}

	DEV Vec3i& operator = (const Vec3i &v)
	{
		x = v.x; 
		y = v.y; 
		z = v.z;

		return *this;
	}

	DEV bool operator < (const Vec3i& V) const
	{
		return V.x < x && V.y < y && V.z < z;
	}

	DEV bool operator > (const Vec3i& V) const
	{
		return V.x > x && V.y > y && V.z > z;
	}

	DEV bool operator == (const Vec3i& V) const
	{
		return V.x == x && V.y == y && V.z == z;
	}

	DEV float LengthSquared(void) const
	{
		return x * x + y * y;
	}

	DEV float Length(void) const
	{
		return sqrtf(LengthSquared());
	}

	DEV int Max(void)
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

	DEV int Min(void)
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

	int x, y, z;
};

class EXPOSURE_RENDER_DLL Vec3f
{
public:
	DEV Vec3f(void)
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}

	DEV Vec3f(const Vec3f& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
	}

	DEV Vec3f(const float& x, const float& y, const float& z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	DEV Vec3f(const float& xyz)
	{
		this->x = xyz;
		this->y = xyz;
		this->z = xyz;
	}

	DEV float operator[](int i) const
	{
		return (&x)[i];
	}

	DEV float& operator[](int i)
	{
		return (&x)[i];
	}

	DEV Vec3f& operator = (const Vec3f &v)
	{
		x = v.x; 
		y = v.y; 
		z = v.z;

		return *this;
	}

	DEV Vec3f operator + (const Vec3f& v) const
	{
		Vec3f Result;

		return Vec3f(x + v.x, y + v.y, z + v.z);
	}

	DEV Vec3f& operator += (const Vec3f& v)
	{
		x += v.x; y += v.y; z += v.z;
		return *this;
	}

	DEV Vec3f operator - (const Vec3f& v) const
	{
		return Vec3f(x - v.x, y - v.y, z - v.z);
	}

	DEV Vec3f& operator -= (const Vec3f& v)
	{
		x -= v.x; y -= v.y; z -= v.z;
		return *this;
	}

	DEV Vec3f operator * (float f) const
	{
		return Vec3f(x * f, y * f, z * f);
	}

	DEV Vec3f& operator *= (float f)
	{
		x *= f; 
		y *= f; 
		z *= f;

		return *this;
	}

	DEV Vec3f operator / (float f) const
	{
		float inv = 1.f / f;
		return Vec3f(x * inv, y * inv, z * inv);
	}

	DEV Vec3f& operator /= (float f)
	{
		float inv = 1.f / f;
		x *= inv; y *= inv; z *= inv;
		return *this;
	}

	DEV Vec3f operator / (Vec3i V)
	{
		return Vec3f(x / (float)V.x, y / (float)V.y, z / (float)V.z);
	}

	DEV Vec3f& operator /= (Vec3i V)
	{
		x /= (float)V.x;
		y /= (float)V.y;
		z /= (float)V.z;

		return *this;
	}

	DEV Vec3f operator-() const
	{
		return Vec3f(-x, -y, -z);
	}

	DEV bool operator < (const Vec3f& V) const
	{
		return V.x < x && V.y < y && V.z < z;
	}

	DEV bool operator > (const Vec3f& V) const
	{
		return V.x > x && V.y > y && V.z > z;
	}

	DEV bool operator == (const Vec3f& V) const
	{
		return V.x == x && V.y == y && V.z == z;
	}

	DEV float Max(void) const
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

	DEV float Min(void) const
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

	DEV void Normalize(void)
	{
		const float L = Length();
		x /= L;
		y /= L;
		z /= L;
	}

	DEV float Dot(const Vec3f& rhs) const
	{
		return (x * rhs.x + y * rhs.y + z * rhs.z);
	}

	DEV Vec3f Cross(const Vec3f& rhs) const
	{
		return Vec3f(	(y * rhs.z) - (z * rhs.y),
							(z * rhs.x) - (x * rhs.z),
							(x * rhs.y) - (y * rhs.x));
	}

	DEV void ScaleBy(const float& factor)
	{
		x *= factor;
		y *= factor;
		z *= factor;
	}

	DEV void RotateAxis(const Vec3f& axis, const float degrees)
	{
		RadianRotateAxis(axis, degrees * DEG_TO_RAD);
	}

	DEV void RadianRotateAxis(const Vec3f& axis, const float radians)
	{
		// Formula goes CW around axis. I prefer to think in terms of CCW
		// rotations, to be consistant with the other rotation metDEVs.
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

	DEV float NormLengthSquared(void)
	{
		float vl = x * x + y * y + z * z;
		
		if (vl != 0.0f)
		{
			const float d = 1.0f / sqrt(vl);
			x *= d; y *= d; z *= d;
		}

		return vl;
	}

	float x, y, z;
};

class EXPOSURE_RENDER_DLL Vec4i
{
public:
	DEV Vec4i(void)
	{
		this->x = 0;
		this->y = 0;
		this->z = 0;
		this->w = 0;
	}

	DEV Vec4i(const int& x, const int& y, const int& z, const int& w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	DEV Vec4i(const int& xyzw)
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
	DEV Vec4f(void)
	{
		this->x = 0.0f;
		this->y = 0.0f;
		this->z = 0.0f;
		this->w = 0.0f;
	}

	DEV Vec4f(const float& x, const float& y, const float& z, const float& w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	DEV Vec4f operator * (float f) const
	{
		return Vec4f(x * f, y * f, z * f, w * f);
	}

	DEV Vec4f& operator *= (float f)
	{
		x *= f; 
		y *= f; 
		z *= f;
		w *= f;

		return *this;
	}

	float x, y, z, w;
};

HOD inline Vec3f Normalize(const Vec3f& v)
{
	return v / v.Length();
}

// Vec2f
inline DEV Vec2f operator * (const Vec2f& v, const float& f) 	{ return Vec2f(f * v.x, f * v.y);					};
inline DEV Vec2f operator * (const float& f, const Vec2f& v) 	{ return Vec2f(f * v.x, f * v.y);					};
inline DEV Vec2f operator * (const Vec2f& v1, const Vec2f& v2) 	{ return Vec2f(v1.x * v2.x, v1.y * v2.y);			};
inline DEV Vec2f operator / (const Vec2f& v1, const Vec2f& v2) 	{ return Vec2f(v1.x / v2.x, v1.y / v2.y);			};

// Vec3f
inline DEV Vec3f operator * (const float& f, const Vec3f& v) 		{ return Vec3f(f * v.x, f * v.y, f * v.z);						};
inline DEV Vec3f operator * (const Vec3f& v1, const Vec3f& v2) 		{ return Vec3f(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);			};
inline DEV Vec3f operator / (const Vec3f& v1, const Vec3f& v2) 		{ return Vec3f(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);			};


inline DEV Vec2f operator * (Vec2f& V2f, Vec2i& V2i)	{ return Vec2f(V2f.x * V2i.x, V2f.y * V2i.y);				};

class EXPOSURE_RENDER_DLL CRay
{	
public:
	// ToDo: Add description
	DEV CRay(void)
	{
	};

	// ToDo: Add description
	DEV CRay(Vec3f Origin, Vec3f Dir, float MinT, float MaxT = INF_MAX, int PixelID = 0)
	{
		m_O			= Origin;
		m_D			= Dir;
		m_MinT		= MinT;
		m_MaxT		= MaxT;
		m_PixelID	= PixelID;
	}

	// ToDo: Add description
	DEV ~CRay(void)
	{
	}

	// ToDo: Add description
	DEV CRay& operator=(const CRay& Other)
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
	DEV Vec3f operator()(float t) const
	{
		return m_O + Normalize(m_D) * t;
	}

	Vec3f 	m_O;			/*!< Ray origin */
	Vec3f 	m_D;			/*!< Ray direction */
	float	m_MinT;			/*!< Minimum parametric range */
	float	m_MaxT;			/*!< Maximum parametric range */
	int		m_PixelID;		/*!< Pixel ID associated with the ray */
};

class EXPOSURE_RENDER_DLL CPixel
{
public:
	DEV CPixel(void)
	{
		m_XY	= Vec2i(256);
		m_ID	= 0;
	}

	DEV CPixel(const Vec2f& ImageXY, const Vec2i& Resolution)
	{
		m_XY	= Vec2i(floorf(ImageXY.x), floorf(ImageXY.y));
		m_ID	= (m_XY.y * Resolution.x) + m_XY.x;
	}

	DEV CPixel& operator = (const CPixel& Other)
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
	CResolution2D(const float& Width, const float& Height)
	{
		m_XY		= Vec2i(Width, Height);

		Update();
	}

	// ToDo: Add description
	DEV CResolution2D(void)
	{
		m_XY		= Vec2i(640, 480);

		Update();
	}

	// ToDo: Add description
	DEV ~CResolution2D(void)
	{
	}

	// ToDo: Add description
	DEV CResolution2D& CResolution2D::operator=(const CResolution2D& Other)
	{
		m_XY				= Other.m_XY;
		m_InvXY				= Other.m_InvXY;
		m_NoElements		= Other.m_NoElements;
		m_AspectRatio		= Other.m_AspectRatio;
		m_DiagonalLength	= Other.m_DiagonalLength;

		return *this;
	}

	DEV int operator[](int i) const
	{
		return m_XY[i];
	}

	DEV int& operator[](int i)
	{
		return m_XY[i];
	}

	DEV bool operator == (const CResolution2D& Other) const
	{
		return GetResX() == Other.GetResX() && GetResY() == Other.GetResY();
	}

	DEV bool operator != (const CResolution2D& Other) const
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
	DEV Vec2i ToVector(void) const
	{
		return Vec2i(m_XY.x, m_XY.y);
	}

	void Set(const Vec2i& Resolution)
	{
		m_XY		= Resolution;

		Update();
	}

	HOD int		GetResX(void) const				{ return m_XY.x; }
	DEV void	SetResX(const int& Width)		{ m_XY.x = Width; Update(); }
	HOD int		GetResY(void) const				{ return m_XY.y; }
	DEV void	SetResY(const int& Height)		{ m_XY.y = Height; Update(); }
	DEV Vec2f	GetInv(void) const				{ return m_InvXY; }
	HOD int		GetNoElements(void) const		{ return m_NoElements; }
	DEV float	GetAspectRatio(void) const		{ return m_AspectRatio; }


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
	DEV CResolution3D(void)
	{
		SetResX(0);
		SetResY(0);
		SetResZ(0);
	}

	// ToDo: Add description
	DEV CResolution3D& CResolution3D::operator=(const CResolution3D& Other)
	{
		m_XYZ				= Other.m_XYZ;
		m_InvXYZ			= Other.m_InvXYZ;
		m_NoElements		= Other.m_NoElements;
		m_DiagonalLength	= Other.m_DiagonalLength;
		m_Dirty				= Other.m_Dirty;

		return *this;
	}

	DEV int operator[](int i) const
	{
		return m_XYZ[i];
	}

	DEV int& operator[](int i)
	{
		return m_XYZ[i];
	}

	DEV void Update(void)
	{
		m_InvXYZ.x			= m_XYZ.x == 0.0f ? 1.0f : 1.0f / (float)m_XYZ.x;
		m_InvXYZ.y			= m_XYZ.y == 0.0f ? 1.0f : 1.0f / (float)m_XYZ.y;
		m_InvXYZ.z			= m_XYZ.z == 0.0f ? 1.0f : 1.0f / (float)m_XYZ.z;
		m_NoElements		= m_XYZ.x * m_XYZ.y * m_XYZ.z;
		m_DiagonalLength	= m_XYZ.Length();
	}

	DEV Vec3f ToVector3(void) const
	{
		return Vec3f(m_XYZ.x, m_XYZ.y, m_XYZ.z);
	}

	DEV void SetResXYZ(const Vec3i& Resolution)
	{
		m_Dirty	= m_XYZ.x != Resolution.x || m_XYZ.y != Resolution.y || m_XYZ.z != Resolution.z;
		m_XYZ	= Resolution;

		Update();
	}

	DEV Vec3i	GetResXYZ(void) const				{ return m_XYZ; }
	DEV int		GetResX(void) const					{ return m_XYZ.x; }
	DEV void	SetResX(const int& ResX)			{ m_Dirty = m_XYZ.x != ResX; m_XYZ.x = ResX; Update(); }
	DEV int		GetResY(void) const					{ return m_XYZ.y; }
	DEV void	SetResY(const int& ResY)			{ m_Dirty = m_XYZ.y != ResY; m_XYZ.y = ResY; Update(); }
	DEV int		GetResZ(void) const					{ return m_XYZ.z; }
	DEV void	SetResZ(const int& ResZ)			{ m_Dirty = m_XYZ.z != ResZ; m_XYZ.z = ResZ; Update(); }
	DEV Vec3f	GetInv(void) const					{ return m_InvXYZ; }
	DEV int		GetNoElements(void) const			{ return m_NoElements; }
	DEV int		GetMin(void) const					{ return min(GetResX(), min(GetResY(), GetResZ()));		}
	DEV int		GetMax(void) const					{ return max(GetResX(), max(GetResY(), GetResZ()));		}

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

DEV inline float AbsDot(const Vec3f& a, const Vec3f& b)			
{
	return fabsf(Dot(a, b));			
};

HOD inline Vec3f Cross(const Vec3f &v1, const Vec3f &v2)		
{
	return Vec3f((v1.y * v2.z) - (v1.z * v2.y), (v1.z * v2.x) - (v1.x * v2.z), (v1.x * v2.y) - (v1.y * v2.x)); 
};

// reflect
DEV inline Vec3f Reflect(Vec3f i, Vec3f n)
{
	return i - 2.0f * n * Dot(n, i);
}

DEV inline float Length(const Vec3f& v)
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
DEV inline Vec3f Floor(const Vec3f v)
{
	return Vec3f(floor(v.x), floor(v.y), floor(v.z));
}

DEV inline float Distance(Vec3f p1, Vec3f p2)
{
	return (p1 - p2).Length();
}

DEV inline float DistanceSquared(Vec3f p1, Vec3f p2)
{
	return (p1 - p2).LengthSquared();
}

DEV inline Vec3f Reflect(Vec3f& i, Vec3f& n)
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

DEV inline void CoordinateSystem(const Vec3f& v1, Vec3f* v2, Vec3f *v3)
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
	DEV Vec()
	{
		for (int i = 0; i < Size; i++)
			m_D[i] = T();
	}

	DEV Vec(const Vec<T, Size>& D)
	{
		for (int i = 0; i < Size; i++)
			m_D[i] = D[i];
	}

	DEV Vec(const T& Other)
	{
		*this = Other;
	}

	DEV T& operator = (const T& Other)
	{
		for (int i = 0; i < Size; i++)
			m_D[i] = Other[i];

		return *this;
	}

	DEV T operator[](const int& i) const
	{
		return m_D[i];
	}

	DEV T& operator[](const int& i)
	{
		return (&m_D)[i];
	}

	DEV Vec<T, Size> operator * (const float& f) const
	{
		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = m_D[i] * f;

		return Result;
	}

	DEV Vec<T, Size>& operator *= (const float& f)
	{
		for (int i = 0; i < Size; i++)
			m_D[i] *= f;

		return *this;
	}

	DEV Vec<T, Size> operator / (const float& f) const
	{
		const float Inv = 1.0f / f;

		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = m_D[i] * Inv;

		return Result;
	}

	DEV Vec<T, Size>& operator /= (float f)
	{
		const float Inv = 1.0f / f;

		for (int i = 0; i < Size; i++)
			m_D[i] *= Inv;

		return *this;
	}

	DEV bool operator < (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (m_D[i] > V[i])
				return false;
		}

		return true;
	}

	DEV bool operator > (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (m_D[i] < V[i])
				return false;
		}

		return true;
	}

	DEV bool operator == (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (m_D[i] != V[i])
				return false;
		}

		return true;
	}

	DEV int Max(void)
	{
		T Max;

		for (int i = 1; i < Size; i++)
		{
			if (m_D[i] > m_D[i - 1])
				Max = m_D[i];
		}

		return Max;
	}

	DEV int Min(void)
	{
		T Min;

		for (int i = 1; i < Size; i++)
		{
			if (m_D[i] < m_D[i - 1])
				Min = m_D[i];
		}

		return Min;
	}

protected:
	T	m_D[Size];
};

template <class T>
class EXPOSURE_RENDER_DLL Vec2 : public Vec<T, 2>
{
public:
	DEV Vec2(void)
	{
		m_D[0] = T();
		m_D[1] = T();
	}

	DEV Vec2(const T& V1, const T& V2)
	{
		m_D[0] = V1;
		m_D[1] = V2;
	}
};

template <class T>
class EXPOSURE_RENDER_DLL Vec3 : public Vec<T, 3>
{
public:
	DEV Vec3()
	{
		m_D[0] = T();
		m_D[1] = T();
		m_D[2] = T();
	}

	DEV Vec3(const T& V1, const T& V2, const T& V3)
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
	DEV Vec4(void)
	{
		m_D[0] = T();
		m_D[1] = T();
		m_D[2] = T();
		m_D[3] = T();
	}

	DEV Vec4(const T& V1, const T& V2, const T& V3, const T& V4)
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
	DEV ColorRGB(void)
	{
	}

	DEV ColorRGB(const T& R, const T& G, const T& B)
	{
		Set(R, G, B);
	}

	DEV void Set(const T& R, const T& G, const T& B)
	{
		SetR(R);
		SetG(G);
		SetB(B);
	}

	DEV T GetR(void) const
	{
		return m_D[0];
	}

	DEV void SetR(const T& R)
	{
		m_D[0] = R;
	}

	DEV T GetG(void) const
	{
		return m_D[1];
	}

	DEV void SetG(const T& G)
	{
		m_D[1] = G;
	}

	DEV T GetB(void) const
	{
		return m_D[2];
	}

	DEV void SetB(const T& B)
	{
		m_D[2] = B;
	}

	DEV void SetBlack(void)
	{
		Set(T(), T(), T());
	}
};

template <class T>
class EXPOSURE_RENDER_DLL ColorRGBA : public ColorRGB<T, 4>
{
public:
	DEV void Set(const T& R, const T& G, const T& B, const T& A)
	{
		SetR(R);
		SetG(G);
		SetB(B);
		SetA(A);
	}

	DEV T GetA(void) const
	{
		return m_D[3];
	}

	DEV void SetA(const T& A)
	{
		m_D[3] = A;
	}
};

class EXPOSURE_RENDER_DLL ColorRGBuc : public ColorRGB<unsigned char, 3>
{
public:
	DEV ColorRGBuc(const unsigned char& R = 0, const unsigned char& G = 0, const unsigned char& B = 0)
	{
		Set(R, G, B);
	}

	DEV void FromRGBf(const float& R, const float& G, const float& B)
	{
		SetR(clamp2(R, 0.0f, 1.0f) * 255.0f);
		SetG(clamp2(G, 0.0f, 1.0f) * 255.0f);
		SetB(clamp2(B, 0.0f, 1.0f) * 255.0f);
	}

	DEV void FromXYZ(const float& X, const float& Y, const float& Z)
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

	DEV void SetBlack(void)
	{
		Set(0, 0, 0);
	}

	DEV void SetWhite(void)
	{
		Set(255, 255, 255);
	}
};

class EXPOSURE_RENDER_DLL ColorRGBAuc : public ColorRGBA<unsigned char>
{
public:
	DEV ColorRGBAuc(const unsigned char& R = 0, const unsigned char& G = 0, const unsigned char& B = 0, const unsigned char& A = 0)
	{
		Set(R, G, B, A);
	}

	DEV ColorRGBAuc(const ColorRGBuc& RGB)
	{
		SetR(RGB.GetR());
		SetG(RGB.GetG());
		SetB(RGB.GetB());
	}

	DEV void FromRGBAf(const float& R, const float& G, const float& B, const float& A)
	{
		SetR(clamp2(R, 0.0f, 1.0f) * 255.0f);
		SetG(clamp2(G, 0.0f, 1.0f) * 255.0f);
		SetB(clamp2(B, 0.0f, 1.0f) * 255.0f);
		SetA(clamp2(A, 0.0f, 1.0f) * 255.0f);
	}

	DEV void FromXYZ(const float& X, const float& Y, const float& Z)
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

	DEV void SetBlack(void)
	{
		Set(0, 0, 0, 0);
	}

	DEV void SetWhite(void)
	{
		Set(255, 255, 255, 0);
	}
};

class EXPOSURE_RENDER_DLL ColorXYZf : public Vec3<float>
{
public:
	DEV ColorXYZf(float V = 0.0f)
	{
		Set(V, V, V);
	}

	DEV ColorXYZf(float X, float Y, float Z)
	{
		Set(X, Y, Z);
	}

	DEV void Set(float X, float Y, float Z)
	{
		SetX(X);
		SetY(Y);
		SetZ(Z);
	}

	DEV float GetX(void) const
	{
		return m_D[0];
	}

	DEV void SetX(float X)
	{
		m_D[0] = X;
	}

	DEV float GetY(void) const
	{
		return m_D[1];
	}

	DEV void SetY(float Y)
	{
		m_D[1] = Y;
	}

	DEV float GetZ(void) const
	{
		return m_D[2];
	}

	DEV void SetZ(float Z)
	{
		m_D[2] = Z;
	}

	DEV ColorXYZf& operator += (const ColorXYZf& XYZ)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] += XYZ[i];

		return *this;
	}

	DEV ColorXYZf operator + (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] += XYZ[i];

		return Result;
	}

	DEV ColorXYZf operator - (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] -= XYZ[i];

		return Result;
	}

	DEV ColorXYZf operator / (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] /= XYZ[i];

		return Result;
	}

	DEV ColorXYZf operator * (const ColorXYZf& XYZ) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; i++)
			Result.m_D[i] *= XYZ[i];

		return Result;
	}

	DEV ColorXYZf& operator *= (const ColorXYZf& XYZ)
	{
		for (int i = 0; i < 3; i++)
			m_D[i] *= XYZ[i];

		return *this;
	}

	DEV ColorXYZf operator * (const float& F) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] *= F;

		return Result;
	}

	DEV ColorXYZf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] *= F;

		return *this;
	}

	DEV ColorXYZf operator / (const float& F) const
	{
		ColorXYZf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] /= F;

		return Result;
	}

	DEV ColorXYZf& operator /= (float a)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] /= a;

		return *this;
	}

	DEV bool operator == (const ColorXYZf& XYZ) const
	{
		for (int i = 0; i < 3; ++i)
			if (m_D[i] != XYZ[i])
				return false;

		return true;
	}

	DEV bool operator != (const ColorXYZf& XYZ) const
	{
		return !(*this == XYZ);
	}

	DEV float& operator[](int i)
	{
		return m_D[i];
	}

	DEV float operator[](int i) const
	{
		return m_D[i];
	}

	DEV ColorXYZf& ColorXYZf::operator = (const ColorXYZf& Other)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] = Other[i];

		return *this;
	}

	DEV bool IsBlack() const
	{
		for (int i = 0; i < 3; ++i)
			if (m_D[i] != 0.0f)
				return false;

		return true;
	}

	DEV ColorXYZf Clamp(const float& L = 0.0f, const float& H = 1.0f) const
	{
		ColorXYZf Result;

		for (int i = 0; i < 3; ++i)
			Result[i] = clamp2(m_D[i], L, H);

		return Result;
	}

	DEV float Y() const
	{
		float v = 0.0f;

		for (int i = 0; i < 3; i++)
			v += YWeight[i] * m_D[i];

		return v;
	}

	DEV void FromRGB(const float& R, const float& G, const float& B)
	{
		const float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
		const float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
		const float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

		m_D[0] = CoeffX[0] * R + CoeffX[1] * G + CoeffX[2] * B;
		m_D[1] = CoeffY[0] * R + CoeffY[1] * G + CoeffY[2] * B;
		m_D[2] = CoeffZ[0] * R + CoeffZ[1] * G + CoeffZ[2] * B;
	}
};

DEV inline ColorXYZf operator * (const float& F, const ColorXYZf& XYZ)
{
	return XYZ * F;
}

DEV inline ColorXYZf Lerp(const float& T, const ColorXYZf& C1, const ColorXYZf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorXYZf(OneMinusT * C1.GetX() + T * C2.GetX(), OneMinusT * C1.GetY() + T * C2.GetY(), OneMinusT * C1.GetZ() + T * C2.GetZ());
}

class EXPOSURE_RENDER_DLL ColorXYZAf : public Vec4<float>
{
public:
	DEV ColorXYZAf(float V = 0.0f)
	{
		Set(V, V, V, V);
	}

	DEV ColorXYZAf(ColorXYZf XYZ)
	{
		Set(XYZ.GetX(), XYZ.GetY(), XYZ.GetZ());
	}

	DEV ColorXYZAf(float X, float Y, float Z)
	{
		Set(X, Y, Z);
	}

	DEV ColorXYZAf(float X, float Y, float Z, float A)
	{
		Set(X, Y, Z, A);
	}
	
	DEV void Set(float X, float Y, float Z)
	{
		SetX(X);
		SetY(Y);
		SetZ(Z);
	}

	DEV void Set(float X, float Y, float Z, float A)
	{
		SetX(X);
		SetY(Y);
		SetZ(Z);
		SetA(A);
	}

	DEV float GetX(void) const
	{
		return m_D[0];
	}

	DEV void SetX(float X)
	{
		m_D[0] = X;
	}

	DEV float GetY(void) const
	{
		return m_D[1];
	}

	DEV void SetY(float Y)
	{
		m_D[1] = Y;
	}

	DEV float GetZ(void) const
	{
		return m_D[2];
	}

	DEV void SetZ(float Z)
	{
		m_D[2] = Z;
	}

	DEV float GetA(void) const
	{
		return m_D[3];
	}

	DEV void SetA(float A)
	{
		m_D[3] = A;
	}

	DEV ColorXYZAf& operator += (const ColorXYZAf& XYZ)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] += XYZ[i];

		return *this;
	}

	DEV ColorXYZAf operator + (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] += XYZ[i];

		return Result;
	}

	DEV ColorXYZAf operator - (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] -= XYZ[i];

		return Result;
	}

	DEV ColorXYZAf operator / (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] /= XYZ[i];

		return Result;
	}

	DEV ColorXYZAf operator * (const ColorXYZAf& XYZ) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] *= XYZ[i];

		return Result;
	}

	DEV ColorXYZAf& operator *= (const ColorXYZAf& XYZ)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] *= XYZ[i];

		return *this;
	}

	DEV ColorXYZAf operator * (const float& F) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] *= F;

		return Result;
	}

	DEV ColorXYZAf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] *= F;

		return *this;
	}

	DEV ColorXYZAf operator / (const float& F) const
	{
		ColorXYZAf Result = *this;

		for (int i = 0; i < 3; ++i)
			Result.m_D[i] /= F;

		return Result;
	}

	DEV ColorXYZAf& operator/=(float a)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] /= a;

		return *this;
	}

	DEV bool operator == (const ColorXYZAf& XYZ) const
	{
		for (int i = 0; i < 3; ++i)
			if (m_D[i] != XYZ[i])
				return false;

		return true;
	}

	DEV bool operator != (const ColorXYZAf& XYZ) const
	{
		return !(*this == XYZ);
	}

	DEV float& operator[](int i)
	{
		return m_D[i];
	}

	DEV float operator[](int i) const
	{
		return m_D[i];
	}

	DEV ColorXYZAf& ColorXYZAf::operator = (const ColorXYZAf& Other)
	{
		for (int i = 0; i < 3; ++i)
			m_D[i] = Other[i];

		return *this;
	}

	DEV bool IsBlack() const
	{
		for (int i = 0; i < 3; ++i)
			if (m_D[i] != 0.0f)
				return false;

		return true;
	}

	DEV ColorXYZAf Clamp(const float& L = 0.0f, const float& H = 1.0f) const
	{
		ColorXYZAf Result;

		for (int i = 0; i < 3; ++i)
			Result[i] = clamp2(m_D[i], L, H);

		return Result;
	}

	DEV float Y() const
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

DEV inline ColorXYZAf operator * (const float& F, const ColorXYZAf& XYZA)
{
	return XYZA * F;
}

DEV inline ColorXYZAf Lerp(const float& T, const ColorXYZAf& C1, const ColorXYZAf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorXYZAf(OneMinusT * C1.GetX() + T * C2.GetX(), OneMinusT * C1.GetY() + T * C2.GetY(), OneMinusT * C1.GetZ() + T * C2.GetZ(), OneMinusT * C1.GetA() + T * C2.GetA());
}

class EXPOSURE_RENDER_DLL ColorRGBf : public Vec3<float>
{
public:
	DEV ColorRGBf(void)
	{
	}

	DEV ColorRGBf(const float& R, const float& G, const float& B)
	{
		Set(R, G, B);
	}

	DEV ColorRGBf(const float& RGB)
	{
		Set(RGB, RGB, RGB);
	}

	DEV void Set(const float& R, const float& G, const float& B)
	{
		SetR(R);
		SetG(G);
		SetB(B);
	}

	DEV float GetR(void) const
	{
		return m_D[0];
	}

	DEV void SetR(const float& R)
	{
		m_D[0] = R;
	}

	DEV float GetG(void) const
	{
		return m_D[1];
	}

	DEV void SetG(const float& G)
	{
		m_D[1] = G;
	}

	DEV float GetB(void) const
	{
		return m_D[2];
	}

	DEV void SetB(const float& B)
	{
		m_D[2] = B;
	}

	DEV void SetBlack(void)
	{
		Set(0.0f, 0.0f, 0.0f);
	}

	DEV void SetWhite(void)
	{
		Set(1.0f, 1.0f, 1.0f);
	}

	DEV ColorRGBf& operator = (const ColorRGBf& Other)			
	{
		for (int i = 0; i < 3; i++)
			m_D[i] = Other[i];

		return *this;
	}

	DEV ColorRGBf& operator += (ColorRGBf& Other)		
	{
		for (int i = 0; i < 3; i++)
			m_D[i] += Other[i];

		return *this;
	}

	DEV ColorRGBf operator * (const float& F) const
	{
		return ColorRGBf(m_D[0] * F, m_D[1] * F, m_D[2] * F);
	}

	DEV ColorRGBf& operator *= (const float& F)
	{
		for (int i = 0; i < 3; i++)
			m_D[i] *= F;

		return *this;
	}

	DEV ColorRGBf operator / (const float& F) const
	{
		const float Inv = 1.0f / F;
		return ColorRGBf(m_D[0] * Inv, m_D[1] * Inv, m_D[2] * Inv);
	}

	DEV ColorRGBf& operator /= (const float& F)
	{
		const float Inv = 1.0f / F;
		
		for (int i = 0; i < 3; i++)
			m_D[i] *= Inv;

		return *this;
	}

	DEV float operator[](int i) const
	{
		return m_D[i];
	}

	DEV float operator[](int i)
	{
		return m_D[i];
	}

	DEV bool Black(void)
	{
		for (int i = 0; i < 3; i++)
			if (m_D[i] != 0.0f)
				return false;

		return true;
	}

	DEV ColorRGBf Pow(const float& E)
	{
		return ColorRGBf(powf(m_D[0], E), powf(m_D[1], E), powf(m_D[1], E));
	}

	DEV void FromXYZ(const float& X, const float& Y, const float& Z)
	{
		const float rWeight[3] = { 3.240479f, -1.537150f, -0.498535f };
		const float gWeight[3] = {-0.969256f,  1.875991f,  0.041556f };
		const float bWeight[3] = { 0.055648f, -0.204043f,  1.057311f };

		Set(rWeight[0] * X + rWeight[1] * Y + rWeight[2] * Z, gWeight[0] * X + gWeight[1] * Y + gWeight[2] * Z, bWeight[0] * X + bWeight[1] * Y + bWeight[2] * Z);
	}

	DEV void ToneMap(const float& InvExposure)
	{
		m_D[0] = Clamp(1.0f - expf(-(m_D[0] * InvExposure)), 0.0f, 1.0f);
		m_D[1] = Clamp(1.0f - expf(-(m_D[1] * InvExposure)), 0.0f, 1.0f);
		m_D[2] = Clamp(1.0f - expf(-(m_D[2] * InvExposure)), 0.0f, 1.0f);
	}
};

DEV inline ColorRGBf Lerp(const float& T, const ColorRGBf& C1, const ColorRGBf& C2)
{
	const float OneMinusT = 1.0f - T;
	return ColorRGBf(OneMinusT * C1.GetR() + T * C2.GetR(), OneMinusT * C1.GetG() + T * C2.GetG(), OneMinusT * C1.GetB() + T * C2.GetB());
}

inline DEV Vec3f MinVec3f(Vec3f a, Vec3f b)
{
	return Vec3f(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

inline DEV Vec3f MaxVec3f(Vec3f a, Vec3f b)
{
	return Vec3f(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

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