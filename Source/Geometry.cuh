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

#include "Color.h"
#include "Ray.cuh"

using namespace std;

namespace ExposureRender
{

DEVICE Vec3f TransformVector(const Matrix44& TM, const Vec3f& V)
{
	Vec3f Vt;

	const float x = V[0], y = V[1], z = V[2];

	Vt[0] = TM.NN[0][0] * x + TM.NN[0][1] * y + TM.NN[0][2] * z;
	Vt[1] = TM.NN[1][0] * x + TM.NN[1][1] * y + TM.NN[1][2] * z;
	Vt[2] = TM.NN[2][0] * x + TM.NN[2][1] * y + TM.NN[2][2] * z;

	return Vt;
}

DEVICE Vec3f TransformPoint(const Matrix44& TM, const Vec3f& P)
{
	const float x = P[0], y = P[1], z = P[2];
    
	const float Px = TM.NN[0][0]*x + TM.NN[0][1]*y + TM.NN[0][2]*z + TM.NN[0][3];
    const float Py = TM.NN[1][0]*x + TM.NN[1][1]*y + TM.NN[1][2]*z + TM.NN[1][3];
    const float Pz = TM.NN[2][0]*x + TM.NN[2][1]*y + TM.NN[2][2]*z + TM.NN[2][3];
	
	return Vec3f(Px, Py, Pz);
}

DEVICE Ray TransformRay(const Matrix44& TM, const Ray& R)
{
	Ray Rt;

	Vec3f P		= TransformPoint(TM, R.O);
	Vec3f MinP	= TransformPoint(TM, R(R.MinT));
	Vec3f MaxP	= TransformPoint(TM, R(R.MaxT));

	Rt.O	= P;
	Rt.D	= Normalize(MaxP - Rt.O);
	Rt.MinT	= (MinP - Rt.O).Length();
	Rt.MaxT	= (MaxP - Rt.O).Length();

	return Rt;
}

DEVICE float SphericalTheta(const Vec3f& W)
{
	return acosf(Clamp(W[1], -1.0f, 1.0f));
}

DEVICE float SphericalPhi(const Vec3f& W)
{
	float p = atan2f(W[2], W[0]);
	return (p < 0.0f) ? p + 2.0f * PI_F : p;
}

DEVICE_NI Vec2f SphericalToUV(const Vec3f& W)
{
	const Vec3f V = Normalize(W);
	return Vec2f(INV_TWO_PI_F * SphericalPhi(V), 1.0f - (INV_PI_F * SphericalTheta(V)));
}

HOST_DEVICE float Lerp(float t, float v1, float v2)
{
	return (1.f - t) * v1 + t * v2;
}

HOST_DEVICE void swap(int& a, int& b)
{
	int t = a; a = b; b = t;
}

HOST_DEVICE void swap(float& a, float& b)
{
	float t = a; a = b; b = t;
}

HOST_DEVICE void Swap(float* pF1, float* pF2)
{
	const float TempFloat = *pF1;

	*pF1 = *pF2;
	*pF2 = TempFloat;
}

HOST_DEVICE void Swap(float& F1, float& F2)
{
	const float TempFloat = F1;

	F1 = F2;
	F2 = TempFloat;
}

HOST_DEVICE void Swap(int* pI1, int* pI2)
{
	const int TempInt = *pI1;

	*pI1 = *pI2;
	*pI2 = TempInt;
}

HOST_DEVICE void Swap(int& I1, int& I2)
{
	const int TempInt = I1;

	I1 = I2;
	I2 = TempInt;

}


template <class T, int NoDimensions>
class Resolution : public Vec<T, NoDimensions>
{
public:
	HOST_DEVICE Resolution()
	{
		for (int i = 0; i < NoDimensions; i++)
			this->D[i] = T();
	}

	HOST_DEVICE Resolution(T Res)
	{
		for (int i = 0; i < NoDimensions; i++)
			this->D[i] = Res;
	}

	HOST_DEVICE Resolution(T Resolution[NoDimensions])
	{
		for (int i = 0; i < NoDimensions; i++)
			this->D[i] = Resolution[i];
	}

	HOST_DEVICE Vec<T, NoDimensions> Inv(void) const
	{
		return 1.0f / *this;
	}

	HOST_DEVICE int GetNoElements(void) const
	{
		T NoElements = this->D[0];

		for (int i = 1; i < NoDimensions; i++)
			NoElements *= this->D[i];

		return NoElements;			
	}

	HOST_DEVICE bool operator == (const Resolution& R) const
	{
		for (int i = 0; i < NoDimensions; i++)
		{
			if (this->D[i] != R[i])
				return false;
		}

		return true;
	}

	HOST_DEVICE bool operator != (const Resolution& R) const
	{
		for (int i = 0; i < NoDimensions; i++)
		{
			if (this->D[i] != R[i])
				return true;
		}

		return false;
	}
};

typedef Resolution<int, 2>				Resolution2i;
typedef Resolution<int, 3>				Resolution3i;
typedef Resolution<float, 2>			Resolution2f;
typedef Resolution<float, 3>			Resolution3f;

inline float RandomFloat(void)
{
	return (float)rand() / RAND_MAX;
}

struct ScatterEvent
{
	enum SampleType
	{
		Volume,
		Light,
		Object
	};

	SampleType	Type;
	bool		Valid;
	float		T;
	Vec3f		P;
	Vec3f		N;
	Vec3f		Wo;
	ColorXYZf	Le;
	Vec2f		UV;
	int			ObjectID;
	int			LightID;

	HOST_DEVICE ScatterEvent()
	{
	}

	HOST_DEVICE ScatterEvent(const SampleType& Type)
	{
		this->Type = Type;
		this->SetInvalid();
	}

	HOST_DEVICE void SetValid(float T, Vec3f P, Vec3f N, Vec3f Wo, ColorXYZf Le, Vec2f UV = Vec2f(0.0f))
	{
		this->Valid		= true;
		this->T			= T;
		this->P			= P;
		this->N			= N;
		this->Wo		= Wo;
		this->Le		= Le;
		this->UV		= UV;
	}

	HOST_DEVICE void SetInvalid()
	{
		this->Valid		= false;
		this->T			= 0.0f;
		this->P			= Vec3f(0.0f);
		this->N			= Vec3f(0.0f);
		this->Wo		= Vec3f(0.0f);
		this->Le		= ColorXYZf(0.0f);
		this->UV		= Vec2f(0.0f);
	}

	HOST_DEVICE ScatterEvent& ScatterEvent::operator = (const ScatterEvent& Other)
	{
		this->Type			= Other.Type;
		this->Valid			= Other.Valid;	
		this->T				= Other.T;
		this->P				= Other.P;
		this->N				= Other.N;
		this->Wo			= Other.Wo;
		this->Le			= Other.Le;
		this->UV			= Other.UV;
		this->ObjectID		= Other.ObjectID;
		this->LightID		= Other.LightID;

		return *this;
	}
};

struct Intersection
{
	bool		Valid;
	bool		Front;
	float		NearT;
	float		FarT;
	Vec3f		P;
	Vec3f		N;
	Vec2f		UV;
	
	HOST_DEVICE Intersection()
	{
		this->SetInvalid();
	}

	HOST_DEVICE void SetValid(float NearT, Vec3f P, Vec3f N, Vec2f UV = Vec2f(0.0f))
	{
		this->Valid		= true;
		this->NearT		= NearT;
		this->P			= P;
		this->N			= N;
		this->UV		= UV;
	}

	HOST_DEVICE void SetInvalid()
	{
		this->Valid		= false;
		this->Front		= true;
		this->NearT		= 0.0f;
		this->FarT		= FLT_MAX;
		this->P			= Vec3f(0.0f);
		this->N			= Vec3f(0.0f);
		this->UV		= Vec2f(0.0f);
	}

	HOST_DEVICE Intersection& Intersection::operator = (const Intersection& Other)
	{
		this->Valid			= Other.Valid;	
		this->Front			= Other.Front;
		this->NearT			= Other.NearT;
		this->FarT			= Other.FarT;
		this->P				= Other.P;
		this->N				= Other.N;
		this->UV			= Other.UV;

		return *this;
	}
};

}