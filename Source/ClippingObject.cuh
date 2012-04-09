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

DEVICE bool InsideClippingObject(ClippingObject& ClippingObject, const Vec3f& P)
{
	bool Inside = false;

	switch (ClippingObject.Shape.Type)
	{
		case 0:		
		{
			Inside = InsidePlane(P);
			break;
		}

		case 1:
		{
			Inside = InsideBox(P, ToVec3f(ClippingObject.Shape.Size));
			break;
		}

		case 2:
		{
			Inside = InsideSphere(P, ClippingObject.Shape.OuterRadius);
			break;
		}

		case 3:
		{
			Inside = InsideCylinder(P, ClippingObject.Shape.OuterRadius, ClippingObject.Shape.Size[1]);
			break;
		}
	}

	return ClippingObject.Invert ? !Inside : Inside;
}

DEVICE bool InsideClippingObjects(const Vec3f& P)
{
	for (int i = 0; i < gpTracer->ClippingObjects.Count; i++)
	{
		const Vec3f P2 = TransformPoint(gpTracer->ClippingObjects.List[i].Shape.InvTM, P);

		if (Inside(gpTracer->ClippingObjects.List[i], P2))
			return true;
	}

	return false;
}

}
