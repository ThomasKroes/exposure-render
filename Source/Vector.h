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
#include "Enums.h"

namespace ExposureRender
{

template <class T, int Size>
class Vec
{
public:
	HOST_DEVICE Vec()
	{
		for (int i = 0; i < Size; i++)
			this->D[i] = T();
	}

	HOST_DEVICE Vec(T V)
	{
		for (int i = 0; i < Size; i++)
			this->D[i] = V;
	}

	HOST_DEVICE Vec(const Vec<T, Size>& Other)
	{
		for (int i = 0; i < Size; i++)
			this->D[i] = Other[i];
	}

	HOST_DEVICE virtual ~Vec()
	{
	}

	HOST_DEVICE Vec<T, Size>& operator = (const Vec<T, Size>& Other)
	{
		for (int i = 0; i < Size; i++)
			this->D[i] = Other[i];

		return *this;
	}

	HOST_DEVICE T operator[](const int& i) const
	{
		return this->D[i];
	}

	HOST_DEVICE T& operator[](const int& i)
	{
		return this->D[i];
	}

	HOST_DEVICE Vec<T, Size> operator + (const Vec<T, Size>& V) const
	{
		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = this->D[i] + V[i];

		return Result;
	}

	HOST_DEVICE Vec<T, Size>& operator += (const Vec<T, Size>& V)
	{
		for (int i = 0; i < Size; i++)
			this->D[i] += V[i];

		return *this;
	}

	HOST_DEVICE Vec<T, Size> operator - (const Vec<T, Size>& V) const
	{
		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = this->D[i] - V[i];

		return Result;
	}

	HOST_DEVICE Vec<T, Size>& operator -= (const Vec<T, Size>& V)
	{
		for (int i = 0; i < Size; i++)
			this->D[i] -= V[i];

		return *this;
	}

	HOST_DEVICE Vec<T, Size> operator * (const float& F) const
	{
		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = this->D[i] * F;

		return Result;
	}

	HOST_DEVICE Vec<T, Size>& operator *= (const float& F)
	{
		for (int i = 0; i < Size; i++)
			this->D[i] *= F;

		return *this;
	}

	HOST_DEVICE Vec<T, Size>& operator *= (const Vec<T, Size>& V)
	{
		for (int i = 0; i < Size; i++)
			this->D[i] *= V[i];

		return *this;
	}

	HOST_DEVICE Vec<T, Size> operator / (const float& F) const
	{
		const float Inv = F == 0.0f ? 0.0f : 1.0f / F;

		Vec<T, Size> Result;

		for (int i = 0; i < Size; i++)
			Result[i] = this->D[i] * Inv;

		return Result;
	}

	HOST_DEVICE Vec<T, Size>& operator /= (const T& F)
	{
		const float Inv = F == 0.0f ? 0.0f : 1.0f / F;

		for (int i = 0; i < Size; i++)
			this->D[i] *= Inv;

		return *this;
	}

	HOST_DEVICE Vec<T, Size>& operator /= (const Vec<T, Size>& V)
	{
		for (int i = 0; i < Size; i++)
		{
			const float Inv = V[i] == 0.0f ? 0.0f : 1.0f / V[i];
			this->D[i] *= Inv;
		}

		return *this;
	}

	HOST_DEVICE bool operator < (const Vec<T, Size>& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->D[i] >= V[i])
				return false;
		}

		return true;
	}

	HOST_DEVICE bool operator <= (const Vec<T, Size>& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->D[i] > V[i])
				return false;
		}

		return true;
	}

	HOST_DEVICE bool operator > (const Vec<T, Size>& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->D[i] <= V[i])
				return false;
		}

		return true;
	}

	HOST_DEVICE bool operator >= (const Vec<T, Size>& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->D[i] < V[i])
				return false;
		}

		return true;
	}

	HOST_DEVICE bool operator == (const Vec<T, Size>& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->D[i] != V[i])
				return false;
		}

		return true;
	}

	HOST_DEVICE bool operator != (const Vec<T, Size>& V) const
	{
		for (int i = 0; i < Size; i++)
		{
			if (this->D[i] != V[i])
				return true;
		}

		return false;
	}

	HOST_DEVICE T Max(void)
	{
		T Max;

		for (int i = 1; i < Size; i++)
		{
			if (D[i] > this->D[i - 1])
				Max = this->D[i];
		}

		return Max;
	}

	HOST_DEVICE T Min(void)
	{
		T Min;

		for (int i = 1; i < Size; i++)
		{
			if (this->D[i] < D[i - 1])
				Min = this->D[i];
		}

		return Min;
	}

	HOST_DEVICE Vec<T, Size> Min(const Vec<T, Size>& Other)
	{
		Vec<T, Size> Result;
		
		for (int i = 0; i < Size; i++)
			Result[i] = min(this->D[i], Other[i]);

		return Result;
	}

	HOST_DEVICE Vec<T, Size> Max(const Vec<T, Size>& Other)
	{
		Vec<T, Size> Result;
		
		for (int i = 0; i < Size; i++)
			Result[i] = max(this->D[i], Other[i]);

		return Result;
	}

	HOST_DEVICE void Clamp(T Min, T Max)
	{
		for (int i = 0; i < Size; ++i)
			this->D[i] = max(Min, min(this->D[i], Max));
	}

	HOST_DEVICE void Clamp(const Vec<T, Size>& Min, const Vec<T, Size>& Max)
	{
		for (int i = 0; i < Size; ++i)
			this->D[i] = max(Min[i], min(this->D[i], Max[i]));
	}

protected:
	T	D[Size];
};

template <class T>
class EXPOSURE_RENDER_DLL Vec2 : public Vec<T, 2>
{
public:
	HOST_DEVICE Vec2() :
		Vec<T, 2>()
	{
	}

	HOST_DEVICE Vec2(const Vec<T, 2>& V) :
		Vec<T, 2>(V)
	{
	}

	HOST_DEVICE Vec2(const T& V) :
		Vec<T, 2>(V)
	{
	}

	HOST_DEVICE Vec2(T V1, T V2)
	{
		this->D[0] = V1;
		this->D[1] = V2;
	}

	HOST_DEVICE float LengthSquared(void) const
	{
		return this->D[0] * this->D[0] + this->D[1] * this->D[1];
	}

	HOST_DEVICE float Length(void) const
	{
		return sqrtf(this->LengthSquared());
	}

	HOST_DEVICE void Normalize(void)
	{
		const float L = this->Length();
		this->D[0] /= L;
		this->D[1] /= L;
	}
};

template <class T>
class EXPOSURE_RENDER_DLL Vec3 : public Vec<T, 3>
{
public:
	HOST_DEVICE Vec3() :
		Vec<T, 3>()
	{
	}
	
	HOST_DEVICE Vec3(const Vec<T, 3>& V) :
		Vec<T, 3>(V)
	{
	}

	HOST_DEVICE Vec3(const T& V) :
		Vec<T, 3>(V)
	{
	}

	HOST_DEVICE Vec3(T V1, T V2, T V3)
	{
		this->D[0] = V1;
		this->D[1] = V2;
		this->D[2] = V3;
	}

	HOST_DEVICE float LengthSquared(void) const
	{
		return this->D[0] * this->D[0] + this->D[1] * this->D[1] + this->D[2] * this->D[2];
	}

	HOST_DEVICE float Length(void) const
	{
		return sqrtf(this->LengthSquared());
	}

	HOST_DEVICE void Normalize(void)
	{
		const float L = this->Length();
		this->D[0] /= L;
		this->D[1] /= L;
		this->D[2] /= L;
	}

	HOST_DEVICE float Dot(Vec3 V) const
	{
		return (this->D[0] * V[0] + this->D[1] * V[1] + this->D[2] * V[2]);
	}

	HOST_DEVICE Vec3 Cross(Vec3 V) const
	{
		return Vec3( (this->D[1] * V[2]) - (this->D[2] * V[1]), (this->D[2] * V[0]) - (this->D[0] * V[2]), (this->D[0] * V[1]) - (this->D[1] * V[0]) );
	}

	HOST_DEVICE void ScaleBy(float F)
	{
		this->D[0] *= F;
		this->D[1] *= F;
		this->D[2] *= F;
	}
};

typedef Vec2<unsigned char>	Vec2uc;
typedef Vec2<short>			Vec2s;
typedef Vec2<int>			Vec2i;
typedef Vec2<float>			Vec2f;
typedef Vec2<double>		Vec2d;

typedef Vec3<unsigned char>	Vec3uc;
typedef Vec3<short>			Vec3s;
typedef Vec3<int>			Vec3i;
typedef Vec3<float>			Vec3f;
typedef Vec3<double>		Vec3d;


template <class T> inline HOST_DEVICE Vec2<T> operator + (const Vec2<T>& A, const Vec2<T>& B)		{ return Vec2<T>(A[0] + B[0], A[1] + B[1]);							};
template <class T> inline HOST_DEVICE Vec2<T> operator - (const Vec2<T>& A, const Vec2<T>& B)		{ return Vec2<T>(A[0] - B[0], A[1] - B[1]);							};
template <class T> inline HOST_DEVICE Vec2<T> operator * (const Vec2<T>& V, const T& F)				{ return Vec2<T>(V[0] * F, V[1] * F);								};
template <class T> inline HOST_DEVICE Vec2<T> operator * (T F, Vec2<T> V)							{ return Vec2<T>(V[0] * F, V[1] * F);								};
template <class T> inline HOST_DEVICE Vec2<T> operator * (const Vec2<T>& A, const Vec2<T>& B)		{ return Vec2<T>(A[0] * B[0], A[1] * B[1]);							};
template <class T> inline HOST_DEVICE Vec2<T> operator / (const Vec2<T>& V, const T& F)				{ return Vec2<T>(V[0] / F, V[1] / F);								};
template <class T> inline HOST_DEVICE Vec2<T> operator / (const Vec2<T>& A, const Vec2<T>& B)		{ return Vec2<T>(A[0] / B[0], A[1] / B[1]);							};

template <class T> inline HOST_DEVICE Vec3<T> operator + (const Vec3<T>& V1, const Vec3<T>& V3)		{ return Vec3<T>(V1[0] + V3[0], V1[1] + V3[1], V1[2] + V3[2]);		};
template <class T> inline HOST_DEVICE Vec3<T> operator - (const Vec3<T>& V1, const Vec3<T>& V3)		{ return Vec3<T>(V1[0] - V3[0], V1[1] - V3[1], V1[2] - V3[2]);		};
template <class T> inline HOST_DEVICE Vec3<T> operator * (const Vec3<T>& V, const T& F)				{ return Vec3<T>(V[0] * F, V[1] * F, V[2] * F);						};
template <class T> inline HOST_DEVICE Vec3<T> operator * (T F, Vec3<T> V)							{ return Vec3<T>(V[0] * F, V[1] * F, V[2] * F);						};
template <class T> inline HOST_DEVICE Vec3<T> operator * (const Vec3<T>& A, const Vec3<T>& B)		{ return Vec3<T>(A[0] * B[0], A[1] * B[1], A[2] * B[2]);			};
template <class T> inline HOST_DEVICE Vec3<T> operator / (const Vec3<T>& V, const T& F)				{ return Vec3<T>(V[0] / F, V[1] / F, V[2] / F);						};
template <class T> inline HOST_DEVICE Vec3<T> operator / (const Vec3<T>& A, const Vec3<T>& B)		{ return Vec3<T>(A[0] / B[0], A[1] / B[1], A[2] / B[2]);			};

template <class T> HOST_DEVICE inline Vec2<T> Normalize(Vec2<T> V)									{ Vec2<T> R = V; R.Normalize(); return R; 							};
template <class T> HOST_DEVICE inline Vec3<T> Normalize(Vec3<T> V)									{ Vec3<T> R = V; R.Normalize(); return R; 							};

template <class T>
HOST_DEVICE inline float Clamp(const T& Value, const T& Min, const T& Max)
{
    return Value < Min ? Min : (Value > Max ? Max : Value);
}



HOST_DEVICE inline float Length(const Vec3f& V)										{ return V.Length();						};

HOST_DEVICE inline Vec3f Cross(Vec3f A, Vec3f B)									{ return A.Cross(B);						};
HOST_DEVICE inline float Dot(Vec3f A, Vec3f B)										{ return A.Dot(B);							};
HOST_DEVICE inline float AbsDot(Vec3f A, Vec3f B)									{ return fabs(A.Dot(B));					};
HOST_DEVICE inline float ClampedAbsDot(Vec3f A, Vec3f B)							{ return Clamp(fabs(A.Dot(B)), 0.0f, 1.0f);	};
HOST_DEVICE inline float ClampedDot(Vec3f A, Vec3f B)								{ return Clamp(Dot(A, B), 0.0f, 1.0f);		};
HOST_DEVICE inline float Distance(Vec3f A, Vec3f B)									{ return (A - B).Length();					};
HOST_DEVICE inline float DistanceSquared(Vec3f A, Vec3f B)							{ return (A - B).LengthSquared();			};

HOST_DEVICE inline Vec3f Lerp(Vec3f A, Vec3f B, float T)							{ return A + T * (B - A);					};

}
