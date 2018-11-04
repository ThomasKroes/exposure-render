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

#include "object.h"

namespace ExposureRender
{

HOST_DEVICE_NI void IntersectObject(const Object& Object, const Ray& R, ScatterEvent& RS)
{
	Ray Rt = TransformRay(Object.Shape.InvTM, R);

	Intersection Int;

	IntersectShape(Object.Shape, Rt, Int);

	if (Int.Valid)
	{
		RS.Valid	= true;
		RS.N 		= TransformVector(Object.Shape.TM, Int.N);
		RS.P 		= TransformPoint(Object.Shape.TM, Int.P);
		RS.T 		= Length(RS.P - R.O);
		RS.Wo		= -R.D;
		RS.Le		= ColorXYZf(0.0f);
		RS.UV		= Int.UV;
	}
}

HOST_DEVICE_NI void IntersectObjects(const Ray& R, ScatterEvent& RS)
{
	float T = FLT_MAX;

	for (int i = 0; i < gpTracer->ObjectIDs.Count; i++)
	{
		const Object& Object = gpObjects[i];

		ScatterEvent LocalRS(Enums::Object);

		LocalRS.ObjectID = i;

		IntersectObject(Object, R, LocalRS);

		if (LocalRS.Valid && LocalRS.T < T)
		{
			RS = LocalRS;
			T = LocalRS.T;
		}
	}
}

HOST_DEVICE_NI bool IntersectsObject(const Object& Object, const Ray& R)
{
	return IntersectsShape(Object.Shape, TransformRay(Object.Shape.InvTM, R));
}

HOST_DEVICE_NI bool IntersectsObject(const Ray& R)
{
	for (int i = 0; i < gpTracer->ObjectIDs.Count; i++)
	{
		if (IntersectsObject(gpObjects[i], R))
			return true;
	}
	
	return false;
}

}
