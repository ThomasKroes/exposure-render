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

DEV void SampleLightSurface(ErLight& Light, LightSample& LS)
{
	// Sample the light surface
	switch (Light.Shape.Type)
	{
		case 0:	SamplePlane(LS.SS, LS.RndP, Vec2f(Light.Shape.Size[0], Light.Shape.Size[1]));		break;
		case 1:	SampleDisk(LS.SS, LS.RndP, Light.Shape.OuterRadius);								break;
		case 2:	SampleRing(LS.SS, LS.RndP, Light.Shape.InnerRadius, Light.Shape.OuterRadius);		break;
//		case 3:	SampleBox(LS.SS, LS.RndP, ToVec3f(Light.Shape.Size));								break;
		case 4:	SampleSphere(LS.SS, LS.RndP, Light.Shape.OuterRadius);								break;
	}

	// Transform surface position and normal back to world space
	LS.SS.P	= TransformPoint(Light.Shape.TM, LS.SS.P);
	LS.SS.N	= TransformVector(Light.Shape.TM, LS.SS.N);
}

DEV void SampleLight(ErLight& Light, LightSample& LS, ScatterEvent& SE, Vec3f& Wi, ColorXYZf& Le)
{
	// First sample the light surface
	SampleLightSurface(Light, LS);

	// Compute Wi, the normalized vector from the sampled light position to the ray sample position
	Wi = Normalize(LS.SS.P - SE.P);

	// Compute the probability of sampling the light per unit area
//	LightPdf = G(SE.P, SE.N, LS.SS.P, LS.SS.N) * Light.Shape.Area;

	// Compute exitant radiance
	Le = ColorXYZf(Light.Color[0], Light.Color[1], Light.Color[2]);
}

// Intersects a light with a ray
DEV inline void IntersectLight(ErLight& Light, Ray R, ScatterEvent& SE)
{
	Ray TR = TransformRay(R, Light.Shape.InvTM);

	Intersection Int;

	switch (Light.Shape.Type)
	{
		case 0:	Int = IntersectPlane(TR, Light.Shape.OneSided, Vec2f(Light.Shape.Size[0], Light.Shape.Size[1]));	break;
		case 1:	Int = IntersectDisk(TR, Light.Shape.OneSided, Light.Shape.OuterRadius);								break;
		case 2:	Int = IntersectRing(TR, Light.Shape.OneSided, Light.Shape.InnerRadius, Light.Shape.OuterRadius);	break;
		case 3:	Int = IntersectBox(TR, ToVec3f(Light.Shape.Size), NULL);											break;
		case 4:	Int = IntersectSphere(TR, Light.Shape.OuterRadius);													break;
		case 5:	Int = IntersectCylinder(TR, Light.Shape.OuterRadius, Light.Shape.Size[1]);							break;
	}

	if (Int.Valid)
	{
		SE.Valid	= true;
		SE.P 		= TransformPoint(Light.Shape.TM, Int.P);
		SE.N 		= TransformVector(Light.Shape.TM, Int.N);
		SE.T 		= Length(SE.P - R.O);
		SE.Wo		= -R.D;
		SE.Le		= ColorXYZf(Light.Color[0], Light.Color[1], Light.Color[2]);
		SE.UV		= Int.UV;
	}
}

// Finds the nearest intersection with any of the scene's lights
DEV inline void IntersectLights(Ray R, ScatterEvent& RS, bool RespectVisibility = false)
{
	float T = FLT_MAX; 

	for (int i = 0; i < gLights.NoLights; i++)
	{
		ErLight& Light = gLights.LightList[i];
		
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
DEV inline bool IntersectsLight(ErLight& Light, Ray R)
{
	Ray TR = TransformRay(R, Light.Shape.InvTM);

	Intersection Int;

	switch (Light.Shape.Type)
	{
		case 0:	Int = IntersectPlane(TR, Light.Shape.OneSided, Vec2f(Light.Shape.Size[0], Light.Shape.Size[1]));	break;
		case 1:	Int = IntersectDisk(TR, Light.Shape.OneSided, Light.Shape.OuterRadius);								break;
		case 2:	Int = IntersectRing(TR, Light.Shape.OneSided, Light.Shape.InnerRadius, Light.Shape.OuterRadius);	break;
		case 3:	Int = IntersectBox(TR, ToVec3f(Light.Shape.Size), NULL);											break;
		case 4:	Int = IntersectSphere(TR, Light.Shape.OuterRadius);													break;
		case 5:	Int = IntersectCylinder(TR, Light.Shape.OuterRadius, Light.Shape.Size[1]);							break;
	}

	return Int.Valid;
}

// Determines if there's an intersection between the ray and any of the scene's lights
DEV inline bool IntersectsLight(Ray R)
{
	for (int i = 0; i < gLights.NoLights; i++)
	{
		if (IntersectsLight(gLights.LightList[i], R))
			return true;
	}

	return false;
}