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

using namespace std;

HOD inline float clamp2(float v, float a, float b)
{
	return max(a, min(v, b));
}

typedef unsigned char uChar;

template <class T, int Size>
class Vec
{
public:
	HOD Vec()
	{
		for (int i = 0; i < Size; i++)
			this->m_D[i] = T();
	}

	HOD Vec(T V)
	{
		for (int i = 0; i < Size; i++)
			this->m_D[i] = V;
	}

	HOD Vec(const Vec<T, Size>& D)
	{
		for (int i = 0; i < Size; i++)
			this->m_D[i] = D[i];
	}

	HOD Vec(const T& Other)
	{
		*this = Other;
	}

	HOD Vec<T, Size>& operator = (const Vec<T, Size>& Other)
	{
		for (int i = 0; i < Size; i++)
			this->m_D[i] = Other[i];

		return *this;
	}

	HOD T operator[](const int& i) const
	{
		return this->m_D[i];
	}

	HOD T& operator[](const int& i)
	{
		return this->m_D[i];
	}

	HOD Vec<T, Size> operator + (Vec<T, Size> V) const
	{
		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = this->m_D[i] + V[i];

		return Result;
	}

	HOD Vec<T, Size>& operator += (const Vec<T, Size>& V)
	{
		for (int i = 0; i < Size; i++)
			this->m_D[i] += V[i];

		return *this;
	}

	HOD Vec<T, Size> operator -(Vec<T, Size> V) const
	{
		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = this->m_D[i] - V[i];

		return Result;
	}

	HOD Vec<T, Size>& operator -= (const Vec<T, Size>& V)
	{
		for (int i = 0; i < Size; i++)
			this->m_D[i] -= V[i];

		return *this;
	}

	HOD Vec<T, Size> operator * (const float& f) const
	{
		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = this->m_D[i] * f;

		return Result;
	}

	HOD Vec<T, Size>& operator *= (const float& f)
	{
		for (int i = 0; i < Size; i++)
			this->m_D[i] *= f;

		return *this;
	}

	HOD Vec<T, Size>& operator *= (const Vec<T, Size>& V)
	{
		for (int i = 0; i < Size; i++)
			this->m_D[i] *= V[i];

		return *this;
	}

	HOD Vec<T, Size> operator / (const float& f) const
	{
		const float Inv = 1.0f / f;

		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = this->m_D[i] * Inv;

		return Result;
	}

	HOD Vec<T, Size>& operator /= (float f)
	{
		const float Inv = 1.0f / f;

		for (int i = 0; i < Size; i++)
			this->m_D[i] *= Inv;

		return *this;
	}

	HOD Vec<T, Size>& operator /= (Vec<T, Size> V)
	{
		for (int i = 0; i < Size; i++)
			this->m_D[i] / V[i];

		return *this;
	}

	HOD bool operator < (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->m_D[i] > V[i])
				return false;
		}

		return true;
	}

	HOD bool operator <= (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->m_D[i] > V[i])
				return false;
		}

		return true;
	}

	HOD bool operator > (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->m_D[i] < V[i])
				return false;
		}

		return true;
	}

	HOD bool operator >= (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->m_D[i] < V[i])
				return false;
		}

		return true;
	}

	HOD bool operator == (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->m_D[i] != V[i])
				return false;
		}

		return true;
	}

	HOD bool operator != (const T& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->m_D[i] != V[i])
				return true;
		}

		return false;
	}

	HOD int Max(void)
	{
		T Max;

		for (int i = 1; i < Size; i++)
		{
			if (m_D[i] > this->m_D[i - 1])
				Max = this->m_D[i];
		}

		return Max;
	}

	HOD int Min(void)
	{
		T Min;

		for (int i = 1; i < Size; i++)
		{
			if (this->m_D[i] < m_D[i - 1])
				Min = this->m_D[i];
		}

		return Min;
	}

	HOD void Clamp(T Min, T Max)
	{
		for (int i = 0; i < Size; ++i)
			this->m_D[i] = max(Min, min(this->m_D[i], Max));
	}

protected:
	T	m_D[Size];
};

template <class T>
class Vec2 : public Vec<T, 2>
{
public:
	HOD Vec2()
	{
		for (int i = 0; i < 2; i++)
			this->m_D[i] = T();
	}

	HOD Vec2(T V)
	{
		for (int i = 0; i < 2; i++)
			this->m_D[i] = V;
	}

	HOD Vec2(T V1, T V2)
	{
		this->m_D[0] = V1;
		this->m_D[1] = V2;
	}

	HOD float LengthSquared(void) const
	{
		return this->m_D[0] * this->m_D[0] + this->m_D[1] * this->m_D[1];
	}

	HOD float Length(void) const
	{
		return sqrtf(this->LengthSquared());
	}

	HOD void Normalize(void)
	{
		const float L = this->Length();
		this->m_D[0] /= L;
		this->m_D[1] /= L;
	}
};

template <class T>
class Vec3 : public Vec<T, 3>
{
public:
	HOD Vec3()
	{
		for (int i = 0; i < 3; i++)
			this->m_D[i] = T();
	}

	HOD Vec3(T V)
	{
		for (int i = 0; i < 3; i++)
			this->m_D[i] = V;
	}

	HOD Vec3(T V1, T V2, T V3)
	{
		m_D[0] = V1;
		m_D[1] = V2;
		m_D[2] = V3;
	}

	HOD float LengthSquared(void) const
	{
		return this->m_D[0] * this->m_D[0] + this->m_D[1] * this->m_D[1] + this->m_D[2] * this->m_D[2];
	}

	HOD float Length(void) const
	{
		return sqrtf(this->LengthSquared());
	}

	HOD void Normalize(void)
	{
		const float L = this->Length();
		this->m_D[0] /= L;
		this->m_D[1] /= L;
		this->m_D[2] /= L;
	}

	HOD float Dot(Vec3 V) const
	{
		return (this->m_D[0] * V[0] + this->m_D[1] * V[1] + this->m_D[2] * V[2]);
	}

	HOD Vec3 Cross(Vec3 V) const
	{
		return Vec3( (this->m_D[1] * V[2]) - (this->m_D[2] * V[1]), (this->m_D[2] * V[0]) - (this->m_D[0] * V[2]), (this->m_D[0] * V[1]) - (this->m_D[1] * V[0]) );
	}

	HOD void ScaleBy(float F)
	{
		this->m_D[0] *= F;
		this->m_D[1] *= F;
		this->m_D[2] *= F;
	}

	HOD Vec3 operator-() const
	{
		Vec3 Result;

		for (int i = 0; i < 3; i++)
			Result[i] = -(this->m_D[i]);

		return Result;
	}
};

template <class T>
class Vec4 : public Vec<T, 4>
{
public:
	HOD Vec4()
	{
		for (int i = 0; i < 4; i++)
			this->m_D[i] = T();
	}

	HOD Vec4(T V)
	{
		for (int i = 0; i < 4; i++)
			this->m_D[i] = V;
	}

	HOD Vec4(T V1, T V2, T V3, T V4)
	{
		m_D[0] = V1;
		m_D[1] = V2;
		m_D[2] = V3;
		m_D[3] = V4;
	}
};

typedef Vec2<unsigned char>		Vec2uc;
typedef Vec2<short>				Vec2s;
typedef Vec2<int>				Vec2i;
typedef Vec2<float>				Vec2f;
typedef Vec2<double>			Vec2d;

typedef Vec3<unsigned char>		Vec3uc;
typedef Vec3<short>				Vec3s;
typedef Vec3<int>				Vec3i;
typedef Vec3<float>				Vec3f;
typedef Vec3<double>			Vec3d;

typedef Vec4<unsigned char>		Vec4uc;
typedef Vec4<short>				Vec4s;
typedef Vec4<int>				Vec4i;
typedef Vec4<float>				Vec4f;
typedef Vec4<double>			Vec4d;

template <class T> inline HOD Vec2<T> operator + (const Vec2<T>& A, const Vec2<T>& B)		{ return Vec2<T>(A[0] + B[0], A[1] + B[1]);							};
template <class T> inline HOD Vec2<T> operator - (const Vec2<T>& A, const Vec2<T>& B)		{ return Vec2<T>(A[0] - B[0], A[1] - B[1]);							};
template <class T> inline HOD Vec2<T> operator * (const Vec2<T>& V, const T& F)				{ return Vec2<T>(V[0] * F, V[1] * F);								};
template <class T> inline HOD Vec2<T> operator * (T F, Vec2<T> V)							{ return Vec2<T>(V[0] * F, V[1] * F);								};
template <class T> inline HOD Vec2<T> operator * (const Vec2<T>& A, const Vec2<T>& B)		{ return Vec2<T>(A[0] * B[0], A[1] * B[1]);							};
template <class T> inline HOD Vec2<T> operator / (const Vec2<T>& V, const T& F)				{ return Vec2<T>(V[0] / F, V[1] / F);								};
template <class T> inline HOD Vec2<T> operator / (const Vec2<T>& A, const Vec2<T>& B)		{ return Vec2<T>(A[0] / B[0], A[1] / B[1]);							};

template <class T> inline HOD Vec3<T> operator + (const Vec3<T>& V1, const Vec3<T>& V3)		{ return Vec3<T>(V1[0] + V3[0], V1[1] + V3[1], V1[2] + V3[2]);		};
template <class T> inline HOD Vec3<T> operator - (const Vec3<T>& V1, const Vec3<T>& V3)		{ return Vec3<T>(V1[0] - V3[0], V1[1] - V3[1], V1[2] - V3[2]);		};
template <class T> inline HOD Vec3<T> operator * (const Vec3<T>& V, const T& F)				{ return Vec3<T>(V[0] * F, V[1] * F, V[2] * F);						};
template <class T> inline HOD Vec3<T> operator * (T F, Vec3<T> V)							{ return Vec3<T>(V[0] * F, V[1] * F, V[2] * F);						};
template <class T> inline HOD Vec3<T> operator * (const Vec3<T>& A, const Vec3<T>& B)		{ return Vec3<T>(A[0] * B[0], A[1] * B[1], A[2] * B[2]);			};
template <class T> inline HOD Vec3<T> operator / (const Vec3<T>& V, const T& F)				{ return Vec3<T>(V[0] / F, V[1] / F, V[2] / F);						};
template <class T> inline HOD Vec3<T> operator / (const Vec3<T>& A, const Vec3<T>& B)		{ return Vec3<T>(A[0] / B[0], A[1] / B[1], A[2] / B[2]);			};

template <class T> HOD inline Vec2<T> Normalize(Vec2<T> V)									{ Vec2<T> R = V; R.Normalize(); return R; 							};
template <class T> HOD inline Vec3<T> Normalize(Vec3<T> V)									{ Vec3<T> R = V; R.Normalize(); return R; 							};

HOD inline float clamp(float x, float a, float b)
{
    return x < a ? a : (x > b ? b : x);
}

HOD inline float Length(Vec3f V)											{ return V.Length();						};
HOD inline Vec3f Cross(Vec3f A, Vec3f B)									{ return A.Cross(B);						};
HOD inline float Dot(Vec3f A, Vec3f B)										{ return A.Dot(B);							};
HOD inline float AbsDot(Vec3f A, Vec3f B)									{ return fabs(A.Dot(B));					};
HOD inline float ClampedAbsDot(Vec3f A, Vec3f B)							{ return clamp(fabs(A.Dot(B)), 0.0f, 1.0f);	};
HOD inline float ClampedDot(Vec3f A, Vec3f B)								{ return clamp(Dot(A, B), 0.0f, 1.0f);		};
HOD inline float Distance(Vec3f A, Vec3f B)									{ return (A - B).Length();					};
HOD inline float DistanceSquared(Vec3f A, Vec3f B)							{ return (A - B).LengthSquared();			};









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
	return Vec3f(Clamp(v[0], a, b), Clamp(v[1], a, b), Clamp(v[2], a, b));
}

inline HOD Vec3f Clamp(Vec3f v, Vec3f a, Vec3f b)
{
	return Vec3f(Clamp(v[0], a[0], b[0]), Clamp(v[1], a[1], b[1]), Clamp(v[2], a[2], b[2]));
}

// floor
HOD inline Vec3f Floor(const Vec3f v)
{
	return Vec3f(floor(v[0]), floor(v[1]), floor(v[2]));
}

HOD inline Vec3f Reflect(Vec3f& i, Vec3f& n)
{
	return i - 2.0f * n * Dot(n, i);
}

inline HOD Vec3f MinVec3f(Vec3f a, Vec3f b)
{
	return Vec3f(min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]));
}

inline HOD Vec3f MaxVec3f(Vec3f a, Vec3f b)
{
	return Vec3f(max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]));
}
