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

#include "MonteCarlo.cuh"

KERNEL void KrnlComputeAutoFocusDistance(float* pAutoFocusDistance, int FilmU, int FilmV, unsigned int Seed1, unsigned int Seed2)
{
	CRNG RNG(&Seed1, &Seed2);

	Ray Rc;

	ScatterEvent SE(ScatterEvent::Volume);

	float Sum = 0.0f, SumWeight = 0.0f;

	for (int i = 0; i < 100; i++)
	{
		Vec2f ScreenPoint;

		ScreenPoint[0] = gCamera.Screen[0][0] + (gCamera.InvScreen[0] * (float)FilmU);
		ScreenPoint[1] = gCamera.Screen[1][0] + (gCamera.InvScreen[1] * (float)FilmV);

		ScreenPoint += 0.01f * ConcentricSampleDisk(RNG.Get2());

		Rc.O	= ToVec3f(gCamera.Pos);
		Rc.D	= Normalize(ToVec3f(gCamera.N) + (ScreenPoint[0] * ToVec3f(gCamera.U)) - (ScreenPoint[1] * ToVec3f(gCamera.V)));
		Rc.MinT	= gCamera.ClipNear;
		Rc.MaxT	= gCamera.ClipFar;

		SampleVolume(Rc, RNG, SE);

		if (SE.Valid)
		{
			Sum += (SE.P - Rc.O).Length();
			SumWeight += 1.0f;
		}
	}

	if (Sum <= 0.0f)
		*pAutoFocusDistance = -1.0f;
	else
		*pAutoFocusDistance = Sum / SumWeight;
}

void ComputeAutoFocusDistance(int FilmU, int FilmV, float& AutoFocusDistance)
{
	float* pAutoFocusDistance = NULL;

	CUDA::Allocate(pAutoFocusDistance);

	LAUNCH_CUDA_KERNEL((KrnlComputeAutoFocusDistance<<<1, 1>>>(pAutoFocusDistance, FilmU, FilmV, rand(), rand())));
	
	CUDA::MemCopyDeviceToHost(pAutoFocusDistance, &AutoFocusDistance);
	CUDA::Free(pAutoFocusDistance);
}