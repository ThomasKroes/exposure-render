#pragma once

#include "Spectrum.h"

class CColorRgbHdr;
class CColorRgbLdr;
class Vec2i;
class Vec2f;
class Vec3i;
class Vec3f;
class Vec4i;
class Vec4f;
class CColorXyz;

class EXPOSURE_RENDER_DLL CColorRgbHdr
{
public:
	HOD CColorRgbHdr(void)
	{
		r = 0.0f;
		g = 0.0f;
		b = 0.0f;
	}

	HOD CColorRgbHdr(const float& r, const float& g, const float& b)
	{
		this->r = r;
		this->g = g;
		this->b = b;
	}

	HOD CColorRgbHdr(const float& rgb)
	{
		r = rgb;
		g = rgb;
		b = rgb;
	}

	HOD CColorRgbHdr& operator = (const CColorRgbHdr& p)			
	{
		r = p.r;
		g = p.g;
		b = p.b;

		return *this;
	}

	HOD CColorRgbHdr& operator = (const CColorXyz& S);	

	HOD CColorRgbHdr& operator += (CColorRgbHdr &p)		
	{
		r += p.r;
		g += p.g;
		b += p.b;	

		return *this;
	}

	HOD CColorRgbHdr operator * (float f) const
	{
		return CColorRgbHdr(r * f, g * f, b * f);
	}

	HOD CColorRgbHdr& operator *= (float f)
	{
		for (int i = 0; i < 3; i++)
			(&r)[i] *= f;

		return *this;
	}

	HOD CColorRgbHdr operator / (float f) const
	{
		float inv = 1.0f / f;
		return CColorRgbHdr(r * inv, g * inv, b * inv);
	}

	HOD CColorRgbHdr& operator /= (float f)
	{
		float inv = 1.f / f;
		r *= inv; g *= inv; b *= inv;
		return *this;
	}

	HOD float operator[](int i) const
	{
		return (&r)[i];
	}

	HOD float operator[](int i)
	{
		return (&r)[i];
	}

	HOD bool Black(void)
	{
		return r == 0.0f && g == 0.0f && b == 0.0f;
	}

	HOD CColorRgbHdr Pow(float e)
	{
		return CColorRgbHdr(powf(r, e), powf(g, e), powf(b, e));
	}

	HOD void FromXYZ(float x, float y, float z)
	{
		const float rWeight[3] = { 3.240479f, -1.537150f, -0.498535f };
		const float gWeight[3] = {-0.969256f,  1.875991f,  0.041556f };
		const float bWeight[3] = { 0.055648f, -0.204043f,  1.057311f };

		r =	rWeight[0] * x +
			rWeight[1] * y +
			rWeight[2] * z;

		g =	gWeight[0] * x +
			gWeight[1] * y +
			gWeight[2] * z;

		b =	bWeight[0] * x +
			bWeight[1] * y +
			bWeight[2] * z;
	}

	HOD CColorXyz ToXYZ(void)
	{
		return CColorXyz::FromRGB(r, g, b);
	}

	void PrintSelf(void)
	{
		printf("[%g, %g, %g]\n", r, g, b);
	}

	float	r;
	float	g;
	float	b;
};

class EXPOSURE_RENDER_DLL CColorRgbLdr
{
public:
	HOD CColorRgbLdr(void)
	{
		r = 0;
		g = 0;
		b = 0;
	}

	HOD CColorRgbLdr(unsigned char R, unsigned char G, unsigned char B)
	{
		r = R;
		g = G;
		b = B;
	}

	HOD CColorRgbLdr& operator = (const CColorRgbLdr& Other)			
	{
		r = Other.r;
		g = Other.g;
		b = Other.b;

		return *this;
	}

	HOD CColorRgbLdr operator * (float f) const
	{
		return CColorRgbLdr((unsigned char)(r * f), (unsigned char)(g * f), (unsigned char)(b * f));
	}

	HOD CColorRgbLdr& operator *= (float f)
	{
		for (int i = 0; i < 3; i++)
			(&r)[i] *= f;

		return *this;
	}

	HOD CColorRgbLdr operator / (float f) const
	{
		float inv = 1.0f / f;
		return CColorRgbLdr((unsigned char)(r * inv), (unsigned char)(g * inv), (unsigned char)(b * inv));
	}

	HOD CColorRgbLdr& operator /= (float f)
	{
		float inv = 1.0f / f;
		r = (unsigned char)((float)r * inv); g = (unsigned char)((float)g * inv); b = (unsigned char)((float)b * inv);
		return *this;
	}

	HOD float operator[](int i) const
	{
		return (&r)[i];
	}

	HOD float operator[](int i)
	{
		return (&r)[i];
	}

	HOD CColorRgbLdr& operator += (CColorRgbLdr& p)		
	{
		r += p.r;
		g += p.g;
		b += p.b;	

		return *this;
	}

	HOD CColorRgbLdr Pow(float e)
	{
		return CColorRgbLdr((unsigned char)powf(r, e), (unsigned char)powf(g, e), (unsigned char)powf(b, e));
	}

	HOD void FromXYZ(float x, float y, float z)
	{
		const float rWeight[3] = { 3.240479f, -1.537150f, -0.498535f };
		const float gWeight[3] = {-0.969256f,  1.875991f,  0.041556f };
		const float bWeight[3] = { 0.055648f, -0.204043f,  1.057311f };

		float R, G, B;

		R =	rWeight[0] * x +
			rWeight[1] * y +
			rWeight[2] * z;

		G =	gWeight[0] * x +
			gWeight[1] * y +
			gWeight[2] * z;

		B =	bWeight[0] * x +
			bWeight[1] * y +
			bWeight[2] * z;

		clamp2(R, 0.0f, 1.0f);
		clamp2(G, 0.0f, 1.0f);
		clamp2(B, 0.0f, 1.0f);

		r = (unsigned char)(R * 255.0f);
		g = (unsigned char)(G * 255.0f);
		b = (unsigned char)(B * 255.0f);
	}

	void PrintSelf(void)
	{
		printf("[%d, %d, %d]\n", r, g, b);
	}

public:
	unsigned char	r;
	unsigned char	g;
	unsigned char	b;
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
	HOD Vec3f(void)
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}

	HOD Vec3f(const CColorRgbHdr& p)
	{
		x = p.r;
		y = p.g;
		z = p.b;
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

static HOD CColorRgbHdr operator * (const CColorRgbHdr& v, const float& f) 			{ return CColorRgbHdr(f * v.r, f * v.g, f * v.b); 					};
static HOD CColorRgbHdr operator * (const float& f, const CColorRgbHdr& v) 			{ return CColorRgbHdr(f * v.r, f * v.g, f * v.b); 					};
static HOD CColorRgbHdr operator * (const CColorRgbHdr& p1, const CColorRgbHdr& p2) 	{ return CColorRgbHdr(p1.r * p2.r, p1.g * p2.g, p1.b * p2.b); 		};
static HOD CColorRgbHdr operator + (const CColorRgbHdr& a, const CColorRgbHdr& b)		{ return CColorRgbHdr(a.r + b.r, a.g + b.g, a.b + b.b);				};
static HOD CColorRgbLdr operator + (const CColorRgbLdr& a, const CColorRgbLdr& b)		{ return CColorRgbLdr(a.r + b.r, a.g + b.g, a.b + b.b);				};
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

static inline HOD CColorRgbHdr operator * (const Vec3f& v1, const CColorRgbHdr& p)	{ return CColorRgbHdr(v1.x * p.r, v1.y * p.g, v1.z * p.b);				};

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
	float	m_Min;			/*!< Minimum range */
	float	m_InvMin;		/*!< Minimum range reciprocal */
	float	m_Max;			/*!< Maximum range */
	float	m_InvMax;		/*!< Maximum range reciprocal */
	float	m_Length;		/*!< Length */
	float	m_InvLength;	/*!< Length reciprocal */

	HOD CRange(void) :
		m_Min(0.0f),
		m_InvMin(m_Min != 0.0f ? 1.0f / m_Min : 0.0f),
		m_Max(100.0f),
		m_InvMax(m_Max != 0.0f ? 1.0f / m_Max : 0.0f),
		m_Length(m_Max - m_Min),
		m_InvLength(1.0f / m_Length)
	{
	}

	HOD CRange(const float& MinRange, const float& MaxRange) :
		m_Min(MinRange),
		m_InvMin(m_Min != 0.0f ? 1.0f / m_Min : 0.0f),
		m_Max(MaxRange),
		m_InvMax(m_Max != 0.0f ? 1.0f / m_Max : 0.0f),
		m_Length(m_Max - m_Min),
		m_InvLength(1.0f / m_Length)
	{
	}

	HOD CRange& operator = (const CRange& Other)
	{
		m_Min		= Other.m_Min; 
		m_Max		= Other.m_Max; 
		m_Length	= Other.m_Length;
		m_InvLength	= Other.m_InvLength;

		return *this;
	}

	void PrintSelf(void)
	{
		printf("[%4.2f - %4.2f]\n", m_Min, m_Max);
	}
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

	HOD	Vec3f				GetMinP(void) const		{ return m_MinP;	}
	HOD	void				SetMinP(Vec3f MinP)		{ m_MinP = MinP;	}
	HOD	Vec3f				GetMaxP(void) const		{ return m_MaxP;	}
	HOD	void				SetMaxP(Vec3f MaxP)		{ m_MaxP = MaxP;	}

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

HOD inline CColorRgbHdr& CColorRgbHdr::operator = (const CColorXyz& S)			
{
	r = S.c[0];
	g = S.c[1];
	b = S.c[2];

	return *this;
}

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

HOD inline CColorRgbHdr Lerp(float T, const CColorRgbHdr& C1, const CColorRgbHdr& C2)
{
	const float OneMinusT = 1.0f - T;
	return CColorRgbHdr(OneMinusT * C1.r + T * C2.r, OneMinusT * C1.g + T * C2.g, OneMinusT * C1.b + T * C2.b);
}

HOD inline CColorXyz Lerp(float T, const CColorXyz& C1, const CColorXyz& C2)
{
	const float OneMinusT = 1.0f - T;
	return CColorXyz(OneMinusT * C1.c[0] + T * C2[0], OneMinusT * C1.c[0] + T * C2[0], OneMinusT * C1.c[0] + T * C2[0]);
}

// ToDo: Add description
class EXPOSURE_RENDER_DLL CTransferFunction
{
public:
	float			m_P[MAX_NO_TF_POINTS];		/*!< Node positions */
	CColorRgbHdr	m_C[MAX_NO_TF_POINTS];		/*!< Node colors in HDR RGB */
	int				m_NoNodes;					/*!< No. nodes */

	// ToDo: Add description
	HO CTransferFunction(void)
	{
		for (int i = 0; i < MAX_NO_TF_POINTS; i++)
		{
			m_P[i]	= 0.0f;
			m_C[i]	= SPEC_BLACK;
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
	HOD CColorRgbHdr F(const float& P)
	{
		for (int i = 0; i < m_NoNodes - 1; i++)
		{
			if (P >= m_P[i] && P < m_P[i + 1])
			{
				const float T = (float)(P - m_P[i]) / (m_P[i + 1] - m_P[i]);
				return Lerp(T, m_C[i], m_C[i + 1]);
			}
		}

		return CColorRgbHdr(0.0f);
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

inline float RandomFloat(void)
{
	return (float)rand() / RAND_MAX;
}