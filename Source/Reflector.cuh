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

// Intersect a reflector with a ray
DEV inline void IntersectReflector(ErReflector& Reflector, Ray R, ScatterEvent& RS)
{
	Ray TR = TransformRay(R, Reflector.Shape.InvTM);

	Intersection Int;

	switch (Reflector.Shape.Type)
	{
		case 0:	Int = IntersectPlane(TR, Reflector.Shape.OneSided, Vec2f(Reflector.Shape.Size[0], Reflector.Shape.Size[1]));		break;
		case 1:	Int = IntersectDisk(TR, Reflector.Shape.OneSided, Reflector.Shape.OuterRadius);										break;
		case 2:	Int = IntersectRing(TR, Reflector.Shape.OneSided, Reflector.Shape.InnerRadius, Reflector.Shape.OuterRadius);		break;
		case 3:	Int = IntersectBox(TR, ToVec3f(Reflector.Shape.Size), NULL);														break;
		case 4:	Int = IntersectSphere(TR, Reflector.Shape.OuterRadius);																break;
//		case 5:	Int = IntersectCylinder(TR, Reflector.Shape.OuterRadius, Reflector.Shape.Size[1]);									break;
	}

	if (Int.Valid)
	{
		RS.Valid	= true;
		RS.P 		= TransformPoint(Reflector.Shape.TM, Int.P);
		RS.N 		= TransformVector(Reflector.Shape.TM, Int.N);
		RS.T 		= Length(RS.P - R.O);
		RS.Wo		= -R.D;
		RS.Le		= ColorXYZf(0.0f);
		RS.UV		= Int.UV;
	}
}

// Finds the nearest intersection with any of the scene's reflectors
DEV inline void IntersectReflectors(Ray R, ScatterEvent& RS)
{
	float T = FLT_MAX;

	for (int i = 0; i < gReflectors.NoReflectors; i++)
	{
		ErReflector& RO = gReflectors.ReflectorList[i];

		ScatterEvent LocalRS(ScatterEvent::Reflector);

		LocalRS.ReflectorID = i;

		IntersectReflector(RO, R, LocalRS);

		if (LocalRS.Valid && LocalRS.T < T)
		{
			RS = LocalRS;
			T = LocalRS.T;
		}
	}
}

// Determine if the ray intersects the reflector
DEV inline bool IntersectsReflector(ErReflector& Reflector, Ray R)
{
	// Transform ray into local shape coordinates
	const Ray TR = TransformRay(R, Reflector.Shape.InvTM);

	// Intersect shape
	switch (Reflector.Shape.Type)
	{
		case 0: return IntersectPlane(TR, false, Vec2f(Reflector.Shape.Size[0], Reflector.Shape.Size[1])).Valid;
		case 1: return IntersectDisk(TR, false, Reflector.Shape.OuterRadius).Valid;
		case 2: return IntersectRing(TR, false, Reflector.Shape.InnerRadius, Reflector.Shape.OuterRadius).Valid;
		case 3: return IntersectBox(TR, ToVec3f(Reflector.Shape.Size)).Valid;
		case 4: return IntersectSphere(TR, Reflector.Shape.OuterRadius).Valid;
//		case 5: return IntersectCylinderP(TR, Light.Shape.OuterRadius, Light.Shape.Size[1]).Valid;
	}

	return false;
}

// Determines if there's an intersection between the ray and any of the scene's reflectors
DEV inline bool IntersectsReflector(Ray R)
{
	for (int i = 0; i < gReflectors.NoReflectors; i++)
	{
		if (IntersectsReflector(gReflectors.ReflectorList[i], R))
			return true;
	}

	return false;
}