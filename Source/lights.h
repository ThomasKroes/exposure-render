/*
    Exposure Render: An interactive photo-realistic volume rendering framework
    Copyright (C) 2011 Thomas Kroes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "light.h"
#include "textures.h"
#include "shapes.h"

namespace ExposureRender
{

HOST_DEVICE_NI void SampleLightSurface(const Light& Light, LightSample& LS, SurfaceSample& SS)
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

HOST_DEVICE_NI void SampleLight(const Light& Light, LightSample& LS, SurfaceSample& SS, ScatterEvent& SE, Vec3f& Wi, ColorXYZf& Le)
{
	SampleLightSurface(Light, LS, SS);

	Wi = Normalize(SS.P - SE.P);

	Le = ColorXYZf::Mul(Light.Multiplier, EvaluateTexture(Light.TextureID, SS.UV));
	
	if (Light.Shape.OneSided && Dot(SE.P - SS.P, SS.N) < 0.0f)
		Le = ColorXYZf::Black();

	if (Light.Unit == 1)
		Le /= Light.Shape.Area;
}

HOST_DEVICE_NI void IntersectLight(const Light& Light, const Ray& R, ScatterEvent& SE)
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
		SE.Le		= Int.Front ? ColorXYZf::Mul(Light.Multiplier, EvaluateTexture(Light.TextureID, SE.UV)) : ColorXYZf::Black();
		
		if (Light.Unit == 1)
			SE.Le /= Light.Shape.Area;
	}
}

HOST_DEVICE_NI void IntersectLights(const Ray& R, ScatterEvent& RS, bool RespectVisibility = false)
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

HOST_DEVICE_NI bool IntersectsLight(const Light& Light, const Ray& R)
{
	return IntersectsShape(Light.Shape, TransformRay(Light.Shape.InvTM, R));
}

HOST_DEVICE_NI bool IntersectsLight(const Ray& R)
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
