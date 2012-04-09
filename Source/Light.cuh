/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "General.cuh"
#include "Texture.cuh"
#include "Shape.cuh"

namespace ExposureRender
{

struct Light : public ErLight
{
	Light()
	{
		printf("Light()\n");
	}

	~Light()
	{
		printf("~Light()\n");
	}

	DEVICE_NI void SampleSurface(LightSample& LS, SurfaceSample& SurfaceSample)
	{
		SampleShape(Shape, LS.SurfaceUVW, SurfaceSample);

		SurfaceSample.P	= TransformPoint(Shape.TM, SurfaceSample.P);
		SurfaceSample.N	= TransformVector(Shape.TM, SurfaceSample.N);
	}

	DEVICE_NI void Sample(LightSample& LS, SurfaceSample& SS, ScatterEvent& SE, Vec3f& Wi, ColorXYZf& Le)
	{
		SampleSurface(LS, SS);

		Wi = Normalize(SS.P - SE.P);

		Le = Multiplier * EvaluateTexture2D(TextureID, SS.UV);

		if (Shape.OneSided && Dot(SE.P - SS.P, SS.N) < 0.0f)
			Le = ColorXYZf(0.0f);

		if (Unit == 1)
			Le /= Shape.Area;
	}

	DEVICE_NI void Intersect(const Ray& R, ScatterEvent& SE)
	{
		const Ray Rt = TransformRay(Shape.InvTM, R);

		Intersection Int;

		IntersectShape(Shape, Rt, Int);

		if (Int.Valid)
		{
			SE.Valid	= true;
			SE.P 		= TransformPoint(Shape.TM, Int.P);
			SE.N 		= TransformVector(Shape.TM, Int.N);
			SE.T 		= Length(SE.P - R.O);
			SE.Wo		= -R.D;
			SE.UV		= Int.UV;
			SE.Le		= Int.Front ? Multiplier * EvaluateTexture2D(TextureID, SE.UV) : ColorXYZf(0.0f);

			if (Unit == 1)
				SE.Le /= Shape.Area;
		}
	}

	DEVICE_NI bool Intersects(const Ray& R)
	{
		return IntersectsShape(Shape, TransformRay(Shape.InvTM, R));
	}

	HOST Light& Light::operator = (const ErLight& Other)
	{
		this->Enabled		= Other.Enabled;
		this->Visible		= Other.Visible;
		this->Shape			= Other.Shape;
		this->TextureID		= Other.TextureID;
		this->Multiplier	= Other.Multiplier;
		this->Unit			= Other.Unit;

		switch (Shape.Type)
		{
			case Enums::Plane:		Shape.Area = PlaneArea(Vec2f(Shape.Size[0], Shape.Size[1]));				break;
			case Enums::Disk:		Shape.Area = DiskArea(Shape.OuterRadius);									break;
			case Enums::Ring:		Shape.Area = RingArea(Shape.OuterRadius, Shape.InnerRadius);				break;
			case Enums::Box:		Shape.Area = BoxArea(Vec3f(Shape.Size[0], Shape.Size[1], Shape.Size[2]));	break;
			case Enums::Sphere:		Shape.Area = SphereArea(Shape.OuterRadius);									break;
			case Enums::Cylinder:	Shape.Area = CylinderArea(Shape.OuterRadius, Shape.Size[2]);				break;
		}

		return *this;
	}
};

struct Lights
{
	Light	List[MAX_NO_LIGHTS];
	int		Count;

	Lights()
	{
		this->Count = 0;
	}
};

__device__ Lights* gpLights = NULL;

DEVICE_NI void IntersectLights(const Ray& R, ScatterEvent& RS, bool RespectVisibility = false)
{
	float T = FLT_MAX; 

	for (int i = 0; i < gpLights->Count; i++)
	{
		Light& Light = gpLights->List[i];
		
		ScatterEvent LocalRS(ScatterEvent::Light);

		LocalRS.LightID = i;

		if (RespectVisibility && !Light.Visible)
			continue;

		Light.Intersect(R, LocalRS);

		if (LocalRS.Valid && LocalRS.T < T)
		{
			RS = LocalRS;
			T = LocalRS.T;
		}
	}
}

DEVICE_NI bool IntersectsLight(const Ray& R)
{
	for (int i = 0; i < gpLights->Count; i++)
	{
		if (gpLights->List[i].Intersects(R))
			return true;
	}

	return false;
}

}
