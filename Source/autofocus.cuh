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

#include "montecarlo.h"

namespace ExposureRender
{

KERNEL void KrnlComputeAutoFocusDistance(float* pAutoFocusDistance, int FilmU, int FilmV, unsigned int Seed1, unsigned int Seed2)
{
	CRNG RNG(&Seed1, &Seed2);

	Ray Rc;

	ScatterEvent SE(ScatterEvent::Volume);

	float Sum = 0.0f, SumWeight = 0.0f;

	for (int i = 0; i < 100; i++)
	{
		Vec2f ScreenPoint;

		ScreenPoint[0] = gpTracer->Camera.Screen[0][0] + (gpTracer->Camera.InvScreen[0] * (float)FilmU);
		ScreenPoint[1] = gpTracer->Camera.Screen[1][0] + (gpTracer->Camera.InvScreen[1] * (float)FilmV);

		ScreenPoint += 0.01f * ConcentricSampleDisk(RNG.Get2());

		Rc.O	= gpTracer->Camera.Pos;
		Rc.D	= Normalize(gpTracer->Camera.N + (ScreenPoint[0] * gpTracer->Camera.U) - (ScreenPoint[1] * gpTracer->Camera.V));
		Rc.MinT	= gpTracer->Camera.ClipNear;
		Rc.MaxT	= gpTracer->Camera.ClipFar;

		SampleVolume(Rc, RNG, SE);

		if (SE.Valid)
		{
			Sum += (SE.P - Rc.O).Length();
			SumWeight += 1.0f;
		}
	}

	if (Sum <= 0.0f)
		*pAutoFocusDistance = (SE.P - Rc.O).Length();
	else
		*pAutoFocusDistance = Sum / SumWeight;
}

void ComputeAutoFocusDistance(int FilmU, int FilmV, float& AutoFocusDistance)
{
	float* pAutoFocusDistance = NULL;

	Cuda::Allocate(pAutoFocusDistance);

	LAUNCH_CUDA_KERNEL((KrnlComputeAutoFocusDistance<<<1, 1>>>(pAutoFocusDistance, FilmU, FilmV, rand(), rand())));
	
	Cuda::MemCopyDeviceToHost(pAutoFocusDistance, &AutoFocusDistance);
	Cuda::Free(pAutoFocusDistance);
}

}