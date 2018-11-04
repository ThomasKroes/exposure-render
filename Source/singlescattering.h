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

#include "macros.cuh"
#include "utilities.h"
#include "transport.h"
#include "camera.h"

namespace ExposureRender
{

HOST_DEVICE_NI void SampleCamera(const Camera& Camera, Ray& R, const int& U, const int& V, CameraSample& CS)
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

HOST_DEVICE_NI ScatterEvent SampleRay(Ray R, CRNG& RNG)
{
	ScatterEvent SE[3] = { ScatterEvent(Enums::Volume), ScatterEvent(Enums::Light), ScatterEvent(Enums::Object) };

	SampleVolume(R, RNG, SE[0]);
	IntersectLights(R, SE[1], true);
	IntersectObjects(R, SE[2]);

	float T = FLT_MAX;

	ScatterEvent NearestRS(Enums::Volume);

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

HOST_DEVICE_NI ColorXYZAf SingleScattering(Tracer* pTracer, const Vec2i& PixelCoord)
{
	CRNG RNG(&gpTracer->FrameBuffer.RandomSeeds1(PixelCoord[0], PixelCoord[1]), &gpTracer->FrameBuffer.RandomSeeds2(PixelCoord[0], PixelCoord[1]));

	ColorXYZf Lv = ColorXYZf::Black();

	MetroSample Sample(RNG); 

	Ray R;

	SampleCamera(gpTracer->Camera, R, PixelCoord[0], PixelCoord[1], Sample.CameraSample);

	ScatterEvent SE;

	SE = SampleRay(R, RNG);

	if (SE.Valid && SE.Type == Enums::Volume)
		Lv += UniformSampleOneLight(SE, RNG, Sample.LightingSample);

	if (SE.Valid && SE.Type == Enums::Light)
		Lv += SE.Le;
	
	if (SE.Valid && SE.Type == Enums::Object)
		Lv += UniformSampleOneLight(SE, RNG, Sample.LightingSample);

	return ColorXYZAf(Lv[0], Lv[1], Lv[2], SE.Valid ? 1.0f : 0.0f);
}

}
