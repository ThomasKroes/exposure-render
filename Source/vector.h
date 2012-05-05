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

#include "defines.h"
#include "enums.h"

namespace ExposureRender
{

template <class T>
HOST_DEVICE inline T Min(const T& A, const T& B)
{
    return A < B ? A : B;
}

template <class T>
HOST_DEVICE inline T Max(const T& A, const T& B)
{
    return A > B ? A : B;
}

template <class T>
HOST_DEVICE inline T Clamp(const T& Value, const T& Min, const T& Max)
{
	return ExposureRender::Max(Min, ExposureRender::Min(Value, Max));
}

//static inline HOST_DEVICE float Lerp(const float& A, const float& B, const float& LerpC)	{ return A + LerpC * (B - A);												};

#define DATA(type, size)																	\
	type	D[size];																		\


#define DEFAULT_CONSTRUCTOR(classname, size)												\
HOST_DEVICE classname()																		\
{																							\
	for (int i = 0; i < size; i++)															\
		this->D[i] = 0;																		\
}
	
#define COPY_CONSTRUCTOR(classname, size)													\
HOST_DEVICE classname(const classname& Other)												\
{																							\
	for (int i = 0; i < size; i++)															\
		this->D[i] = Other[i];																\
}

#define GENERIC_CONSTRUCTOR(classname, type, size)											\
HOST_DEVICE classname(const type& V)														\
{																							\
	for (int i = 0; i < size; i++)															\
		this->D[i] = V;																		\
}

#define ARRAY_CONSTRUCTOR(classname, type, size)											\
HOST_DEVICE classname(const type V[size])													\
{																							\
	for (int i = 0; i < size; i++)															\
		this->D[i] = V[i];																	\
}

#define VEC2_CONSTRUCTOR(classname, type)													\
HOST_DEVICE classname(const type& V1, const type& V2)										\
{																							\
	this->D[0] = V1;																		\
	this->D[1] = V2;																		\
}

#define VEC3_CONSTRUCTOR(classname, type)													\
HOST_DEVICE classname(const type& V1, const type& V2, const type& V3)						\
{																							\
	this->D[0] = V1;																		\
	this->D[1] = V2;																		\
	this->D[2] = V3;																		\
}

#define VEC4_CONSTRUCTOR(classname, type)													\
HOST_DEVICE classname(const type& V1, const type& V2, const type& V3, const type& V4)		\
{																							\
	this->D[0] = V1;																		\
	this->D[1] = V2;																		\
	this->D[2] = V3;																		\
	this->D[3] = V4;																		\
}

#define CONSTRUCTORS(classname, type, size)													\
DEFAULT_CONSTRUCTOR(classname, size)														\
COPY_CONSTRUCTOR(classname, size)															\
GENERIC_CONSTRUCTOR(classname, type, size)													\
ARRAY_CONSTRUCTOR(classname, type, size)													

#define OPERATOR_SUBSCRIPT(type)															\
HOST_DEVICE type operator[](const int& i) const												\
{																							\
	return this->D[i];																		\
}

#define OPERATOR_SUBSCRIPT_REF(type)														\
HOST_DEVICE type& operator[](const int& i)													\
{																							\
	return this->D[i];																		\
}

#define OPERATOR_ASS(classname, size)														\
HOST_DEVICE classname& operator = (const classname& Other)									\
{																							\
	for (int i = 0; i < size; i++)															\
		this->D[i] = Other[i];																\
																							\
	return *this;																			\
}	

#define OPERATOR_PLUS(classname, size)														\
HOST_DEVICE classname operator + (const classname& V) const									\
{																							\
	classname Result;																		\
																							\
	for (int i = 0; i < size; i++)															\
		Result[i] = this->D[i] + V[i];														\
																							\
	return Result;																			\
}


#define OPERATOR_PLUS_ASS(classname, size)													\
HOST_DEVICE classname& operator += (const classname& V)										\
{																							\
	for (int i = 0; i < size; i++)															\
		this->D[i] += V[i];																	\
																							\
	return *this;																			\
}

#define OPERATOR_MINUS(classname, size)														\
HOST_DEVICE classname operator - (const classname& V) const									\
{																							\
	classname Result;																		\
																							\
	for (int i = 0; i < size; i++)															\
		Result[i] = this->D[i] - V[i];														\
																							\
	return Result;																			\
}


#define OPERATOR_MINUS_ASS(classname, size)													\
HOST_DEVICE classname& operator -= (const classname& V)										\
{																							\
	for (int i = 0; i < size; i++)															\
		this->D[i] -= V[i];																	\
																							\
	return *this;																			\
}

#define OPERATOR_MULTIPLY(classname, type, size)											\
HOST_DEVICE classname operator * (const classname& V) const									\
{																							\
	classname Result;																		\
																							\
	for (int i = 0; i < size; i++)															\
		Result[i] = this->D[i] * V[i];														\
																							\
	return Result;																			\
}

#define OPERATOR_MULTIPLY_ASS(classname, type, size)										\
HOST_DEVICE classname& operator *= (const classname& V)										\
{																							\
	for (int i = 0; i < size; i++)															\
		this->D[i] *= V[i];																	\
																							\
	return *this;																			\
}

#define OPERATOR_DIVIDE(classname, size)													\
HOST_DEVICE classname operator / (const classname& V) const									\
{																							\
	classname Result;																		\
																							\
	for (int i = 0; i < size; i++)															\
		Result[i] = this->D[i] / V[i];														\
																							\
	return Result;																			\
}

#define OPERATOR_DIVIDE_ASS(classname, size)												\
HOST_DEVICE classname& operator /= (const classname& V)										\
{																							\
	for (int i = 0; i < size; i++)															\
		this->D[i] /= V[i];																	\
																							\
	return *this;																			\
}

#define OPERATOR_NEGATE(classname, size)													\
HOST_DEVICE classname operator - () const													\
{																							\
	classname Result;																		\
																							\
	for (int i = 0; i < size; i++)															\
		Result[i] = -this->D[i];															\
																							\
	return Result;																			\
}

#define OPERATOR_LESS(classname, size)														\
HOST_DEVICE bool operator < (const classname& V) const										\
{																							\
	for (int i = 0; i < size; i++)															\
	{																						\
		if (this->D[i] >= V[i])																\
			return false;																	\
	}																						\
																							\
	return true;																			\
}

#define OPERATOR_LESS_EQ(classname, size)													\
HOST_DEVICE bool operator <= (const classname& V) const										\
{																							\
	for (int i = 0; i < size; i++)															\
	{																						\
		if (this->D[i] > V[i])																\
			return false;																	\
	}																						\
																							\
	return true;																			\
}

#define OPERATOR_GREATER(classname, size)													\
HOST_DEVICE bool operator > (const classname& V) const										\
{																							\
	for (int i = 0; i < size; i++)															\
	{																						\
		if (this->D[i] <= V[i])																\
			return false;																	\
	}																						\
																							\
	return true;																			\
}

#define OPERATOR_GREATER_EQ(classname, size)												\
HOST_DEVICE bool operator >= (const classname& V) const										\
{																							\
	for (int i = 0; i < size; i++)															\
	{																						\
		if (this->D[i] < V[i])																\
			return false;																	\
	}																						\
																							\
	return true;																			\
}

#define OPERATOR_EQ(classname, size)														\
HOST_DEVICE bool operator == (const classname& V) const										\
{																							\
	for (int i = 0; i < size; i++)															\
	{																						\
		if (this->D[i] != V[i])																\
			return false;																	\
	}																						\
																							\
	return true;																			\
}

#define OPERATOR_NEQ(classname, size)														\
HOST_DEVICE bool operator != (const classname& V) const										\
{																							\
	for (int i = 0; i < size; i++)															\
	{																						\
		if (this->D[i] != V[i])																\
			return true;																	\
	}																						\
																							\
	return false;																			\
}

#define MIN_ELEMENT(type, size)																\
HOST_DEVICE type Min()																		\
{																							\
	type Min = D[0];																		\
																							\
	for (int i = 1; i < size; i++)															\
	{																						\
		if (D[i] < Min)																		\
			Min = this->D[i];																\
	}																						\
																							\
	return Min;																				\
}

#define MAX_ELEMENT(type, size)																\
HOST_DEVICE type Max()																		\
{																							\
	type Max = D[0];																		\
																							\
	for (int i = 1; i < size; i++)															\
	{																						\
		if (D[i] > Max)																		\
			Max = this->D[i];																\
	}																						\
																							\
	return Max;																				\
}

#define MIN_VECTOR(classname, size)															\
HOST_DEVICE classname Min(const classname& Other) const										\
{																							\
	classname Result;																		\
																							\
	for (int i = 0; i < size; i++)															\
		Result[i] = min(this->D[i], Other[i]);												\
																							\
	return Result;																			\
}

#define MAX_VECTOR(classname, size)															\
HOST_DEVICE classname Max(const classname& Other) const										\
{																							\
	classname Result;																		\
																							\
	for (int i = 0; i < size; i++)															\
		Result[i] = max(this->D[i], Other[i]);												\
																							\
	return Result;																			\
}

#define CLAMP_SINGLE(type, size)															\
HOST_DEVICE void Clamp(const type& Min, const type& Max)									\
{																							\
	for (int i = 0; i < size; ++i)															\
		this->D[i] = max(Min, min(this->D[i], Max));										\
}

#define CLAMP_VECTOR(classname, size)														\
HOST_DEVICE void Clamp(const classname& Min, const classname& Max)							\
{																							\
	for (int i = 0; i < size; ++i)															\
		this->D[i] = max(Min[i], min(this->D[i], Max[i]));									\
}

#define CLAMP(classname, type, size)														\
CLAMP_SINGLE(type, size)																	\
CLAMP_VECTOR(classname, size)																

#define MIN_MAX(classname, type, size)														\
MIN_ELEMENT(type, size)																		\
MAX_ELEMENT(type, size)																		\
MIN_VECTOR(classname, size)																	\
MAX_VECTOR(classname, size)																								

#define SUBSCRIPT_OPERATORS(type)															\
OPERATOR_SUBSCRIPT(type)																	\
OPERATOR_SUBSCRIPT_REF(type)																

#define COMPARISON_OPERATORS(classname, size)												\
OPERATOR_LESS(classname, size)																\
OPERATOR_LESS_EQ(classname, size)															\
OPERATOR_GREATER(classname, size)															\
OPERATOR_GREATER_EQ(classname, size)														\
OPERATOR_EQ(classname, size)																\
OPERATOR_NEQ(classname, size)																

#define ARITHMETIC_OPERATORS(classname, type, size)											\
OPERATOR_PLUS(classname, size)																\
OPERATOR_PLUS_ASS(classname, size)															\
OPERATOR_MINUS(classname, size)																\
OPERATOR_MINUS_ASS(classname, size)															\
OPERATOR_MULTIPLY(classname, type, size)													\
OPERATOR_MULTIPLY_ASS(classname, type, size)												\
OPERATOR_DIVIDE(classname, size)															\
OPERATOR_DIVIDE_ASS(classname, size)														\
OPERATOR_NEGATE(classname, size)															

#define ALL_OPERATORS(classname, type, size)												\
SUBSCRIPT_OPERATORS(type)																	\
ARITHMETIC_OPERATORS(classname, type, size)													\
COMPARISON_OPERATORS(classname, size)														

class EXPOSURE_RENDER_DLL Vec2i
{
public:
	CONSTRUCTORS(Vec2i, int, 2)
	VEC2_CONSTRUCTOR(Vec2i, int)
	ALL_OPERATORS(Vec2i, int, 2)
	MIN_MAX(Vec2i, int, 2)
	CLAMP(Vec2i, int, 2)

	DATA(int, 2)
};

class EXPOSURE_RENDER_DLL Vec2f
{
public:
	CONSTRUCTORS(Vec2f, float, 2)
	VEC2_CONSTRUCTOR(Vec2f, float)
	ALL_OPERATORS(Vec2f, float, 2)
	MIN_MAX(Vec2f, float, 2)
	CLAMP(Vec2f, float, 2)

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

	DATA(float, 2)
};

class EXPOSURE_RENDER_DLL Vec3i
{
public:
	CONSTRUCTORS(Vec3i, int, 3)
	VEC3_CONSTRUCTOR(Vec3i, int)
	ALL_OPERATORS(Vec3i, int, 3)
	MIN_MAX(Vec3i, int, 3)
	CLAMP(Vec3i, int, 3)

	DATA(int, 3)
};

class EXPOSURE_RENDER_DLL Vec3f
{
public:
	CONSTRUCTORS(Vec3f, float, 3)
	VEC3_CONSTRUCTOR(Vec3f, float)
	ALL_OPERATORS(Vec3f, float, 3)
	MIN_MAX(Vec3f, float, 3)
	CLAMP(Vec3f, float, 3)

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

	HOST_DEVICE float Dot(const Vec3f& V) const
	{
		return (this->D[0] * V[0] + this->D[1] * V[1] + this->D[2] * V[2]);
	}

	HOST_DEVICE Vec3f Cross(const Vec3f& V) const
	{
		return Vec3f( (this->D[1] * V[2]) - (this->D[2] * V[1]), (this->D[2] * V[0]) - (this->D[0] * V[2]), (this->D[0] * V[1]) - (this->D[1] * V[0]) );
	}

	HOST_DEVICE void ScaleBy(float F)
	{
		this->D[0] *= F;
		this->D[1] *= F;
		this->D[2] *= F;
	}

	DATA(float, 3)
};

class EXPOSURE_RENDER_DLL Vec4i
{
public:
	CONSTRUCTORS(Vec4i, int, 4)
	VEC4_CONSTRUCTOR(Vec4i, int)
	ALL_OPERATORS(Vec4i, int, 4)
	MIN_MAX(Vec4i, int, 4)
	CLAMP(Vec4i, int, 4)

	DATA(int, 4)
};

class EXPOSURE_RENDER_DLL Vec4f
{
public:
	CONSTRUCTORS(Vec4f, float, 4)
	VEC4_CONSTRUCTOR(Vec4f, float)
	ALL_OPERATORS(Vec4f, float, 4)
	MIN_MAX(Vec4f, float, 4)
	CLAMP(Vec4f, float, 4)

	DATA(float, 4)
};

static inline HOST_DEVICE Vec2i operator * (const Vec2i& V, const int& I)					{ return Vec2i(V[0] * I, V[1] * I);											};
static inline HOST_DEVICE Vec2i operator * (const int& I, const Vec2i& V)					{ return Vec2i(V[0] * I, V[1] * I);											};

static inline HOST_DEVICE Vec2f operator * (const Vec2f& V, const float& F)					{ return Vec2f(V[0] * F, V[1] * F);											};
static inline HOST_DEVICE Vec2f operator * (const float& F, const Vec2f& V)					{ return Vec2f(V[0] * F, V[1] * F);											};
static inline HOST_DEVICE Vec2f operator / (const float& f, const Vec2f& v)					{ return Vec2f(f / v[0], f / v[1]);											};
static inline HOST_DEVICE Vec2f operator / (const float& f, const Vec2i& v)					{ return Vec2f(f / (float)v[0], f / (float)v[1]);							};

static inline HOST_DEVICE Vec2f Normalize(const Vec2f& V)									{ Vec2f R = V; R.Normalize(); return R; 									};
static inline HOST_DEVICE float Length(const Vec2f& V)										{ return V.Length();														};

static inline HOST_DEVICE float Distance(const Vec2f& A, const Vec2f& B)					{ return (A - B).Length();													};
static inline HOST_DEVICE float DistanceSquared(const Vec2f& A, const Vec2f& B)				{ return (A - B).LengthSquared();											};
static inline HOST_DEVICE Vec2f Lerp(const Vec2f& A, const Vec2f& B, const float& LerpC)	{ return A + LerpC * (B - A);												};

static inline HOST_DEVICE Vec3i operator * (const Vec3i& V, const int& I)					{ return Vec3i(V[0] * I, V[1] * I, V[2] * I);								};
static inline HOST_DEVICE Vec3i operator * (const int& I, const Vec3i& V)					{ return Vec3i(V[0] * I, V[1] * I, V[2] * I);								};

static inline HOST_DEVICE Vec3f operator * (const float& F, const Vec3f& V)					{ return Vec3f(V[0] * F, V[1] * F, V[2] * F);								};
static inline HOST_DEVICE Vec3f operator / (const float& f, const Vec3i& v)					{ return Vec3f(f / (float)v[0], f / (float)v[1], f / (float)v[2]);			};
static inline HOST_DEVICE Vec3f operator / (const float& f, const Vec3f& v)					{ return Vec3f(f / v[0], f / v[1], f / v[2]);								};

static inline HOST_DEVICE Vec3f Normalize(const Vec3f& V)									{ Vec3f R = V; R.Normalize(); return R; 									};
static inline HOST_DEVICE float Length(const Vec3f& V)										{ return V.Length();														};
static inline HOST_DEVICE Vec3f Cross(const Vec3f& A, const Vec3f& B)						{ return A.Cross(B);														};
static inline HOST_DEVICE float Dot(const Vec3f& A, const Vec3f& B)							{ return A.Dot(B);															};
static inline HOST_DEVICE float AbsDot(const Vec3f& A, const Vec3f& B)						{ return fabs(A.Dot(B));													};
static inline HOST_DEVICE float ClampedAbsDot(const Vec3f& A, const Vec3f& B)				{ return Clamp(fabs(A.Dot(B)), 0.0f, 1.0f);									};
static inline HOST_DEVICE float ClampedDot(const Vec3f& A, const Vec3f& B)					{ return Clamp(Dot(A, B), 0.0f, 1.0f);										};
static inline HOST_DEVICE float Distance(const Vec3f& A, const Vec3f& B)					{ return (A - B).Length();													};
static inline HOST_DEVICE float DistanceSquared(const Vec3f& A, const Vec3f& B)				{ return (A - B).LengthSquared();											};
static inline HOST_DEVICE Vec3f Lerp(const Vec3f& A, const Vec3f& B, const float& LerpC)	{ return A + LerpC * (B - A);												};

class EXPOSURE_RENDER_DLL Indices
{
public:
	HOST Indices()
	{
		for (int i = 0; i < 256; i++)
			this->D[i] = -1;

		this->Count = 0;
	}

	SUBSCRIPT_OPERATORS(int)
	
	HOST Indices& operator = (const Indices& Other)
	{
		for (int i = 0; i < 256; i++)
			this->D[i] = Other.D[i];

		this->Count = Other.Count;

		return *this;
	}

	int D[256];
	int Count;
};

}
