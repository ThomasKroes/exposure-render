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

#include "Transport.cuh"

namespace ExposureRender
{

#define KRNL_SINGLE_SCATTERING_BLOCK_W		16
#define KRNL_SINGLE_SCATTERING_BLOCK_H		8
#define KRNL_SINGLE_SCATTERING_BLOCK_SIZE	KRNL_SINGLE_SCATTERING_BLOCK_W * KRNL_SINGLE_SCATTERING_BLOCK_H

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

KERNEL void KrnlSingleScattering(FrameBuffer* pFrameBuffer)
{
	const int X = blockIdx.x * blockDim.x + threadIdx.x;
	const int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= pFrameBuffer->Resolution[0] || Y >= pFrameBuffer->Resolution[1])
		return;
	
	CRNG RNG(pFrameBuffer->CudaRandomSeeds1.GetPtr(X, Y), pFrameBuffer->CudaRandomSeeds2.GetPtr(X, Y));

	ColorXYZf Lv = SPEC_BLACK;

	ScatterEvent SE;

	MetroSample Sample(RNG);

	Ray Rc;

	SampleCamera(Rc, Sample.CameraSample, X, Y);

	SE = SampleRay(Rc, RNG);

	if (SE.Valid && SE.Type == ScatterEvent::Volume)
		Lv += UniformSampleOneLight(SE, RNG, Sample.LightingSample);

	if (SE.Valid && SE.Type == ScatterEvent::Light)
		Lv += SE.Le;

	if (SE.Valid && SE.Type == ScatterEvent::Object)
		Lv += UniformSampleOneLight(SE, RNG, Sample.LightingSample);

	ColorXYZAf L(Lv.GetX(), Lv.GetY(), Lv.GetZ(), SE.Valid >= 0 ? 1.0f : 0.0f);

	pFrameBuffer->CudaFrameEstimate.Set(L, X, Y);
}

void SingleScattering(FrameBuffer* pFrameBuffer, int Width, int Height)
{
	const dim3 BlockDim(KRNL_SINGLE_SCATTERING_BLOCK_W, KRNL_SINGLE_SCATTERING_BLOCK_H);
	const dim3 GridDim((int)ceilf((float)Width / (float)BlockDim.x), (int)ceilf((float)Height / (float)BlockDim.y));

	LAUNCH_CUDA_KERNEL_TIMED((KrnlSingleScattering<<<GridDim, BlockDim>>>(pFrameBuffer)), "Single Scattering"); 
}

}