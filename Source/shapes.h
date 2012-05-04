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

#include "plane.h"
#include "disk.h"
#include "ring.h"
#include "box.h"
#include "sphere.h"
#include "cylinder.h"

namespace ExposureRender
{

HOST_DEVICE void SampleShape(const Shape& Shape, const Vec3f& SampleUVW, SurfaceSample& SurfaceSample)
{
	switch (Shape.Type)
	{
		case Enums::Plane:		SamplePlane(SurfaceSample, SampleUVW, Vec2f(Shape.Size[0], Shape.Size[1]));					break;
		case Enums::Disk:		SampleDisk(SurfaceSample, SampleUVW, Shape.OuterRadius);									break;
		case Enums::Ring:		SampleRing(SurfaceSample, SampleUVW, Shape.InnerRadius, Shape.OuterRadius);					break;
		case Enums::Box:		SampleBox(SurfaceSample, SampleUVW, Vec3f(Shape.Size[0], Shape.Size[1], Shape.Size[2]));	break;
		case Enums::Sphere:		SampleSphere(SurfaceSample, SampleUVW, Shape.OuterRadius);									break;
//		case Enums::Cylinder:	SampleCylinder(SurfaceSample, SampleUVW, Shape.OuterRadius, Shape.Size[2]);					break;
	}
}

HOST_DEVICE void IntersectShape(const Shape& Shape, const Ray& R, Intersection& Intersection)
{
	switch (Shape.Type)
	{
		case Enums::Plane:		IntersectPlane(R, Shape.OneSided, Vec2f(Shape.Size[0], Shape.Size[1]), Intersection);		break;
		case Enums::Disk:		IntersectDisk(R, Shape.OneSided, Shape.OuterRadius, Intersection);							break;
		case Enums::Ring:		IntersectRing(R, Shape.OneSided, Shape.InnerRadius, Shape.OuterRadius, Intersection);		break;
		case Enums::Box:		IntersectBox(R, Vec3f(Shape.Size[0], Shape.Size[1], Shape.Size[2]), Intersection);			break;
		case Enums::Sphere:		IntersectSphere(R, Shape.OuterRadius, Intersection);										break;
//		case Enums::Cylinder:	IntersectCylinder(R, Shape.OuterRadius, Shape.Size[1], Intersection);						break;
	}
}

HOST_DEVICE bool IntersectsShape(const Shape& Shape, const Ray& R)
{
	Intersection Intersection;

	switch (Shape.Type)
	{
		case Enums::Plane:		 IntersectPlane(R, Shape.OneSided, Vec2f(Shape.Size[0], Shape.Size[1]), Intersection);		break;
		case Enums::Disk:		 IntersectDisk(R, Shape.OneSided, Shape.OuterRadius, Intersection);							break;
		case Enums::Ring:		 IntersectRing(R, Shape.OneSided, Shape.InnerRadius, Shape.OuterRadius, Intersection);		break;
		case Enums::Box:		 IntersectBox(R, Vec3f(Shape.Size[0], Shape.Size[1], Shape.Size[2]), Intersection);			break;
		case Enums::Sphere:		 IntersectSphere(R, Shape.OuterRadius, Intersection);										break;
//		case Enums::Cylinder:	 IntersectCylinderP(R, Shape.OuterRadius, Shape.Size[1], Intersection);						break;
	}

	return Intersection.Valid;
}

}
