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

#include "Camera.h"
#include "Sample.cuh"
#include "MonteCarlo.cuh"

namespace ExposureRender
{

DEVICE void SampleCamera(const Camera& Camera, Ray& R, const int& U, const int& V, CameraSample& CS)
{
	Vec2f ScreenPoint;

	ScreenPoint[0] = Camera.Screen[0][0] + (Camera.InvScreen[0] * (float)(U + CS.FilmUV[0] * 1.0f / Camera.FilmSize[0]));
	ScreenPoint[1] = Camera.Screen[1][0] + (Camera.InvScreen[1] * (float)(V + CS.FilmUV[1] * 1.0f / Camera.FilmSize[1]));

	R.O		= Camera.Pos;
	R.D		= Normalize(Camera.N + (ScreenPoint[0] * Camera.U) - (ScreenPoint[1] * Camera.V));
	R.MinT	= Camera.ClipNear;
	R.MaxT	= Camera.ClipFar;

	if (Camera.ApertureSize != 0.0f)
	{
		const Vec2f LensUV = Camera.ApertureSize * ConcentricSampleDisk(CS.LensUV);

		const Vec3f LI = Camera.U * LensUV[0] + Camera.V * LensUV[1];

		R.O += LI;
		R.D = Normalize(R.D * Camera.FocalDistance - LI);
	}
}

}

/*
// sample N-gon
// FIXME: this could use concentric sampling
float lensSides = 6.0f;
float lensRotationRadians = 0.0f;
float lensY = CS.LensUV[0] * lensSides;
float side = (int)lensY;
float offs = (float) lensY - side;
float dist = (float) sqrtf(CS.LensUV[1]);
float a0 = (float) (side * PI_F * 2.0f / lensSides + lensRotationRadians);
float a1 = (float) ((side + 1.0f) * PI_F * 2.0f / lensSides + lensRotationRadians);
float eyeX = (float) ((cos(a0) * (1.0f - offs) + cos(a1) * offs) * dist);
float eyeY = (float) ((sin(a0) * (1.0f - offs) + sin(a1) * offs) * dist);
eyeX *= gpTracer->Camera.ApertureSize;
eyeY *= gpTracer->Camera.ApertureSize;

const Vec2f LensUV(eyeX, eyeY);// = gpTracer->Camera.ApertureSize * ConcentricSampleDisk(CS.LensUV);

const Vec3f LI = ToVec3f(gpTracer->Camera.U) * LensUV[0] + ToVec3f(gpTracer->Camera.V) * LensUV[1];

Rc.O += LI;
Rc.D = Normalize(Rc.D * gpTracer->Camera.FocalDistance - LI);
*/
