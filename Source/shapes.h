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
