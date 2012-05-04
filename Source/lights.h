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

#include "light.h"
#include "textures.h"
#include "shapes.h"

namespace ExposureRender
{

HOST_DEVICE void SampleLightSurface(const Light& Light, LightSample& LS, SurfaceSample& SS)
{
	switch (Light.Shape.Type)
	{
		case 0:	SamplePlane(SS, LS.SurfaceUVW, Vec2f(Light.Shape.Size[0], Light.Shape.Size[1]));	break;
		case 1:	SampleDisk(SS, LS.SurfaceUVW, Light.Shape.OuterRadius);								break;
		case 2:	SampleRing(SS, LS.SurfaceUVW, Light.Shape.InnerRadius, Light.Shape.OuterRadius);	break;
		case 3:	SampleBox(SS, LS.SurfaceUVW, Light.Shape.Size);										break;
		case 4:	SampleSphere(SS, LS.SurfaceUVW, Light.Shape.OuterRadius);							break;
//		case 5:	SampleCylinder(SS, LS.SurfaceUVW, Light.Shape.OuterRadius, Light.Shape.Size[2]);	break;
	}

	SS.P = TransformPoint(Light.Shape.TM, SS.P);
	SS.N = TransformVector(Light.Shape.TM, SS.N);
}

HOST_DEVICE void SampleLight(const Light& Light, LightSample& LS, SurfaceSample& SS, ScatterEvent& SE, Vec3f& Wi, ColorXYZf& Le)
{
	SampleLightSurface(Light, LS, SS);

	Wi = Normalize(SS.P - SE.P);

	Le = Light.Multiplier * EvaluateTexture(Light.TextureID, SS.UV);
	
	if (Light.Shape.OneSided && Dot(SE.P - SS.P, SS.N) < 0.0f)
		Le = ColorXYZf::Black();

	if (Light.Unit == 1)
		Le /= Light.Shape.Area;
}

HOST_DEVICE void IntersectLight(const Light& Light, const Ray& R, ScatterEvent& SE)
{
	Ray Rt = TransformRay(Light.Shape.InvTM, R);

	Intersection Int;

	switch (Light.Shape.Type)
	{
		case 0:	IntersectPlane(Rt, Light.Shape.OneSided, Vec2f(Light.Shape.Size[0], Light.Shape.Size[1]), Int);		break;
		case 1:	IntersectDisk(Rt, Light.Shape.OneSided, Light.Shape.OuterRadius, Int);								break;
		case 2:	IntersectRing(Rt, Light.Shape.OneSided, Light.Shape.InnerRadius, Light.Shape.OuterRadius, Int);		break;
		case 3:	IntersectBox(Rt, Light.Shape.Size, Int);															break;
		case 4:	IntersectSphere(Rt, Light.Shape.OuterRadius, Int);													break;
//		case 5:	IntersectCylinder(Rt, Light.Shape.OuterRadius, Light.Shape.Size[1], Int);							break;
	}

	if (Int.Valid)
	{
		SE.Valid	= true;
		SE.P 		= TransformPoint(Light.Shape.TM, Int.P);
		SE.N 		= TransformVector(Light.Shape.TM, Int.N);
		SE.T 		= Length(SE.P - R.O);
		SE.Wo		= -R.D;
		SE.UV		= Int.UV;
		SE.Le		= Int.Front ? Light.Multiplier * EvaluateTexture(Light.TextureID, SE.UV) : ColorXYZf::Black();
		
		if (Light.Unit == 1)
			SE.Le /= Light.Shape.Area;
	}
}

HOST_DEVICE void IntersectLights(const Ray& R, ScatterEvent& RS, bool RespectVisibility = false)
{
	float T = FLT_MAX; 

	for (int i = 0; i < gpTracer->LightIDs.Count; i++)
	{
		const Light& Light = gpLights[gpTracer->LightIDs[i]];
		
		ScatterEvent LocalRS(Enums::Light);

		LocalRS.LightID = i;

		if (RespectVisibility && !Light.Visible)
			continue;

		IntersectLight(Light, R, LocalRS);

		if (LocalRS.Valid && LocalRS.T < T)
		{
			RS = LocalRS;
			T = LocalRS.T;
		}
	}
}

HOST_DEVICE bool IntersectsLight(const Light& Light, const Ray& R)
{
	return IntersectsShape(Light.Shape, TransformRay(Light.Shape.InvTM, R));
}

HOST_DEVICE bool IntersectsLight(const Ray& R)
{
	for (int i = 0; i < gpTracer->LightIDs.Count; i++)
	{
		const Light& Light = gpLights[gpTracer->LightIDs[i]];

		if (IntersectsLight(Light, R))
			return true;
	}

	return false;
}

}
