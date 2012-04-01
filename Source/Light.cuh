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

#include "Shader.cuh"
#include "RayMarching.cuh"
#include "General.cuh"
#include "Texture.cuh"

namespace ExposureRender
{

DEVICE_NI void SampleLightSurface(Light& Light, LightSample& LS, SurfaceSample& SS)
{
	// Sample the light surface
	switch (Light.Shape.Type)
	{
		case 0:	SamplePlane(SS, LS.SurfaceUVW, Vec2f(Light.Shape.Size[0], Light.Shape.Size[1]));		break;
		case 1:	SampleDisk(SS, LS.SurfaceUVW, Light.Shape.OuterRadius);									break;
		case 2:	SampleRing(SS, LS.SurfaceUVW, Light.Shape.InnerRadius, Light.Shape.OuterRadius);		break;
		case 3:	SampleBox(SS, LS.SurfaceUVW, ToVec3f(Light.Shape.Size));								break;
		case 4:	SampleSphere(SS, LS.SurfaceUVW, Light.Shape.OuterRadius);								break;
//		case 5:	SampleCylinder(SS, LS.SurfaceUVW, Light.Shape.OuterRadius, Light.Shape.Size[2]);		break;
	}

	// Transform surface position and normal back to world space
	SS.P	= TransformPoint(Light.Shape.TM, SS.P);
	SS.N	= TransformVector(Light.Shape.TM, SS.N);
}

DEVICE_NI void SampleLight(Light& Light, LightSample& LS, SurfaceSample& SS, ScatterEvent& SE, Vec3f& Wi, ColorXYZf& Le)
{
	// First sample the light surface
	SampleLightSurface(Light, LS, SS);

	// Compute Wi, the normalized vector from the sampled light position to the ray sample position
	Wi = Normalize(SS.P - SE.P);

	// Compute exitant radiance
	Le = Light.Multiplier * EvaluateTexture(Light.TextureID, Vec3f(0.0f));//ColorXYZf(Light.Color[0], Light.Color[1], Light.Color[2]);

	if (Light.Shape.OneSided && Dot(SE.P - SS.P, SS.N) < 0.0f)
		Le = ColorXYZf(0.0f);

	if (Light.Unit == 1)
		Le /= Light.Shape.Area;
}

// Intersects a light with a ray
DEVICE_NI void IntersectLight(Light& Light, const Ray& R, ScatterEvent& SE)
{
	Ray TR = TransformRay(Light.Shape.InvTM, R);

	Intersection Int;

	switch (Light.Shape.Type)
	{
		case 0:	IntersectPlane(TR, Light.Shape.OneSided, Vec2f(Light.Shape.Size[0], Light.Shape.Size[1]), Int);		break;
		case 1:	IntersectDisk(TR, Light.Shape.OneSided, Light.Shape.OuterRadius, Int);								break;
		case 2:	IntersectRing(TR, Light.Shape.OneSided, Light.Shape.InnerRadius, Light.Shape.OuterRadius, Int);		break;
		case 3:	IntersectBox(TR, ToVec3f(Light.Shape.Size), Int);													break;
		case 4:	IntersectSphere(TR, Light.Shape.OuterRadius, Int);													break;
//		case 5:	IntersectCylinder(TR, Light.Shape.OuterRadius, Light.Shape.Size[1], Int);							break;
	}

	if (Int.Valid)
	{
		SE.Valid	= true;
		SE.P 		= TransformPoint(Light.Shape.TM, Int.P);
		SE.N 		= TransformVector(Light.Shape.TM, Int.N);
		SE.T 		= Length(SE.P - R.O);
		SE.Wo		= -R.D;
		SE.Le		= Int.Front ? Light.Multiplier * EvaluateTexture(Light.TextureID, Vec3f(0.0f)) : ColorXYZf(0.0f);
		SE.UV		= Int.UV;

		if (Light.Unit == 1)
			SE.Le /= Light.Shape.Area;
	}
}

// Finds the nearest intersection with any of the scene's lights
DEVICE_NI void IntersectLights(const Ray& R, ScatterEvent& RS, bool RespectVisibility = false)
{
	float T = FLT_MAX; 

	for (int i = 0; i < gLights.NoLights; i++)
	{
		Light& Light = gLights.LightList[i];
		
		ScatterEvent LocalRS(ScatterEvent::Light);

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

// Determine if the ray intersects the light
DEVICE_NI bool IntersectsLight(Light& Light, const Ray& R)
{
	// Transform ray into local shape coordinates
	const Ray TR = TransformRay(Light.Shape.InvTM, R);

	Intersection Int;

	// Intersect shape
	switch (Light.Shape.Type)
	{
		case 0: IntersectPlane(TR, Light.Shape.OneSided, Vec2f(Light.Shape.Size[0], Light.Shape.Size[1]), Int);		break;
		case 1: IntersectDisk(TR, Light.Shape.OneSided, Light.Shape.OuterRadius, Int);								break;
		case 2: IntersectRing(TR, Light.Shape.OneSided, Light.Shape.InnerRadius, Light.Shape.OuterRadius, Int);		break;
		case 3: IntersectBox(TR, ToVec3f(Light.Shape.Size), Int);													break;
		case 4: IntersectSphere(TR, Light.Shape.OuterRadius, Int);													break;
//		case 5: IntersectCylinderP(TR, Light.Shape.OuterRadius, Light.Shape.Size[1], Int);							break;
	}

	return Int.Valid;
}

// Determines if there's an intersection between the ray and any of the scene's lights
DEVICE_NI bool IntersectsLight(const Ray& R)
{
	for (int i = 0; i < gLights.NoLights; i++)
	{
		if (IntersectsLight(gLights.LightList[i], R))
			return true;
	}

	return false;
}

}