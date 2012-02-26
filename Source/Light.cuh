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

DEV void SampleLightSurface(LightSurfaceSample& LSS, CRNG& RNG, Vec3f P, int LightID)
{
	ErLight& Light = gLights.LightList[LightID];

	SurfaceSample SS;

	switch (Light.Shape.Type)
	{
		case 0:	SamplePlane(SS, RNG.Get2(), Vec2f(Light.Shape.Size[0], Light.Shape.Size[1]));		break;
		case 1:	SampleDisk(SS, RNG.Get2(), Light.Shape.OuterRadius);								break;
		case 2:	SampleRing(SS, RNG.Get2(), Light.Shape.InnerRadius, Light.Shape.OuterRadius);		break;
		case 3:	SampleBox(SS, RNG.Get3(), ToVec3f(Light.Shape.Size));								break;
		case 4:	SampleSphere(SS, RNG.Get2(), Light.Shape.OuterRadius);								break;
	}
	
	LSS.Le	= ColorXYZf(Light.Color[0], Light.Color[1], Light.Color[2]);
	LSS.P	= TransformPoint(Light.Shape.TM, SS.P);
	LSS.N	= TransformVector(Light.Shape.TM, SS.N);
	LSS.Wi	= Normalize(P - LSS.P);
	LSS.Pdf	= DistanceSquared(P, LSS.P) / SS.Area;
}

DEV void IntersectAreaLight(ErLight& Light, Ray R, RaySample& RS)
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
		RS.Valid	= true;
		RS.P 		= TransformPoint(Light.Shape.TM, Int.P);
		RS.N 		= TransformVector(Light.Shape.TM, Int.N);
		RS.T 		= Length(RS.P - R.O);
		RS.Wo		= -R.D;
		RS.Le		= ColorXYZf(Light.Color[0], Light.Color[1], Light.Color[2]);
		RS.Pdf		= DistanceSquared(R.O, RS.P) / (AbsDot(Normalize(R.O - RS.P), RS.N) * Light.Shape.Area);
		RS.UV		= Int.UV;
	}
}

DEV inline void IntersectAreaLights(Ray R, RaySample& RS, bool RespectVisibility = false)
{
	float T = FLT_MAX;

	for (int i = 0; i < gLights.NoLights; i++)
	{
		ErLight& Light = gLights.LightList[i];
		
		RaySample LocalRS(RaySample::Light);

		LocalRS.LightID = i;

		if (RespectVisibility && !Light.Visible)
			return;

		IntersectAreaLight(Light, R, LocalRS);

		if (LocalRS.Valid && LocalRS.T < T)
		{
			RS = LocalRS;
			T = LocalRS.T;
		}
	}
}