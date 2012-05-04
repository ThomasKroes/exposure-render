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

#include "macros.cuh"
#include "utilities.h"
#include "transport.h"
#include "camera.h"

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

DEVICE ScatterEvent SampleRay(Ray R, CRNG& RNG)
{
	ScatterEvent SE[3] = { ScatterEvent(ScatterEvent::Volume), ScatterEvent(ScatterEvent::Light), ScatterEvent(ScatterEvent::Object) };

	SampleVolume(R, RNG, SE[0]);
	IntersectLights(R, SE[1], true);
	IntersectObjects(R, SE[2]);

	float T = FLT_MAX;

	ScatterEvent NearestRS(ScatterEvent::Volume);

	for (int i = 0; i < 3; i++)
	{
		if (SE[i].Valid && SE[i].T < T)
		{
			NearestRS = SE[i];
			T = SE[i].T;
		}
	}

	return NearestRS;
}

HOST_DEVICE ColorXYZAf SingleScattering(Tracer* pTracer, const Vec2i& PixelCoord)
{
	CRNG RNG(&gpTracer->FrameBuffer.RandomSeeds1(PixelCoord[0], PixelCoord[1]), &gpTracer->FrameBuffer.RandomSeeds2(PixelCoord[0], PixelCoord[1]));

	ColorXYZf Lv = ColorXYZf::Black();

	ScatterEvent SE;

	MetroSample Sample(RNG); 

	Ray R;

	SampleCamera(gpTracer->Camera, R, PixelCoord[0], PixelCoord[1], Sample.CameraSample);

	ScatterEvent SE;

	SE = SampleRay(R, RNG);

	if (SE.Valid && SE.Type == ScatterEvent::Volume)
		Lv += UniformSampleOneLight(SE, RNG, Sample.LightingSample);

	if (SE.Valid && SE.Type == ScatterEvent::Light)
		Lv += SE.Le;
	
	if (SE.Valid && SE.Type == ScatterEvent::Object)
		Lv += UniformSampleOneLight(SE, RNG, Sample.LightingSample);

	return ColorXYZAf(Lv[0], Lv[1], Lv[2], SE.Valid ? 1.0f : 0.0f);
}

}
