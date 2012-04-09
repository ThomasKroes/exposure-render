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

DEVICE_NI void SampleLightSurface(ErLight& Light, LightSample& LS, SurfaceSample& SurfaceSample)
{
	SampleShape(Light.Shape, LS.SurfaceUVW, SurfaceSample);

	// Transform surface position and normal back to world space
	SurfaceSample.P	= TransformPoint(Light.Shape.TM, SurfaceSample.P);
	SurfaceSample.N	= TransformVector(Light.Shape.TM, SurfaceSample.N);
}

DEVICE_NI void SampleLight(ErLight& Light, LightSample& LS, SurfaceSample& SS, ScatterEvent& SE, Vec3f& Wi, ColorXYZf& Le)
{
	// First sample the light surface
	SampleLightSurface(Light, LS, SS);

	// Compute Wi, the normalized vector from the sampled light position to the ray sample position
	Wi = Normalize(SS.P - SE.P);

	// Compute exitant radiance
	Le = Light.Multiplier * EvaluateTexture2D(Light.TextureID, SS.UV);

	if (Light.Shape.OneSided && Dot(SE.P - SS.P, SS.N) < 0.0f)
		Le = ColorXYZf(0.0f);

	if (Light.Unit == 1)
		Le /= Light.Shape.Area;
}

// Intersects a light with a ray
DEVICE_NI void IntersectLight(ErLight& Light, const Ray& R, ScatterEvent& SE)
{
	const Ray Rt = TransformRay(Light.Shape.InvTM, R);

	Intersection Int;

	IntersectShape(Light.Shape, Rt, Int);

	if (Int.Valid)
	{
		SE.Valid	= true;
		SE.P 		= TransformPoint(Light.Shape.TM, Int.P);
		SE.N 		= TransformVector(Light.Shape.TM, Int.N);
		SE.T 		= Length(SE.P - R.O);
		SE.Wo		= -R.D;
		SE.UV		= Int.UV;
		SE.Le		= Int.Front ? Light.Multiplier * EvaluateTexture2D(Light.TextureID, SE.UV) : ColorXYZf(0.0f);

		if (Light.Unit == 1)
			SE.Le /= Light.Shape.Area;
	}
}

// Finds the nearest intersection with any of the scene's lights
DEVICE_NI void IntersectLights(const Ray& R, ScatterEvent& RS, bool RespectVisibility = false)
{
	float T = FLT_MAX; 

	for (int i = 0; i < ((Tracer*)gpTracer)->Lights.Count; i++)
	{
		ErLight& Light = ((Tracer*)gpTracer)->Lights.List[i];
		
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
DEVICE_NI bool IntersectsLight(ErLight& Light, const Ray& R)
{
	return IntersectsShape(Light.Shape, TransformRay(Light.Shape.InvTM, R));
}

// Determines if there's an intersection between the ray and any of the scene's lights
DEVICE_NI bool IntersectsLight(const Ray& R)
{
	for (int i = 0; i < ((Tracer*)gpTracer)->Lights.Count; i++)
	{
		if (IntersectsLight(((Tracer*)gpTracer)->Lights.List[i], R))
			return true;
	}

	return false;
}

}