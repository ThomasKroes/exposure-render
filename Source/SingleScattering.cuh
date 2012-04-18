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
#include "Camera.cuh"

namespace ExposureRender
{

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

KERNEL void KrnlSingleScattering()
{
	KERNEL_2D(gpTracer->FrameBuffer.Resolution[0], gpTracer->FrameBuffer.Resolution[1])

	CRNG RNG(gpTracer->FrameBuffer.CudaRandomSeeds1.GetPtr(IDx, IDy), gpTracer->FrameBuffer.CudaRandomSeeds2.GetPtr(IDx, IDy));

	ColorXYZf Lv = ColorXYZf::Black();

	ScatterEvent SE;

	MetroSample Sample(RNG);

	Ray Rc;

	SampleCamera(gpTracer->Camera, Rc, Sample.CameraSample);

	Intersection Int;

	gpTracer->FrameBuffer.CudaFrameEstimate.Set(ColorXYZAf(0.0f, IDx  > 100 ? 255.0f : 0.0f, 0.0f, 0.0f), IDx, IDy);
	//gpTracer->FrameBuffer.CudaDisplayEstimate(X, Y)[1] = X  > 100 ? 255 : 0;
	return;
	IntersectBox(Rc, gpVolumes->Get(0).MinAABB, gpVolumes->Get(0).MaxAABB, Int);
	


	if (Int.Valid)
		gpTracer->FrameBuffer.CudaFrameEstimate.Set(ColorXYZAf(1.0f), IDx, IDy);
	else
		gpTracer->FrameBuffer.CudaFrameEstimate.Set(ColorXYZAf(0.0f), IDx, IDy);

	/*
	SE = SampleRay(Rc, RNG);

	if (SE.Valid && SE.Type == ScatterEvent::Volume)
		Lv += UniformSampleOneLight(SE, RNG, Sample.LightingSample);

	if (SE.Valid && SE.Type == ScatterEvent::Light)
		Lv += SE.Le;

	if (SE.Valid && SE.Type == ScatterEvent::Object)
		Lv += UniformSampleOneLight(SE, RNG, Sample.LightingSample);

	ColorXYZAf L(Lv.GetX(), Lv.GetY(), Lv.GetZ(), SE.Valid >= 0 ? 1.0f : 0.0f);

	gpTracer->FrameBuffer.CudaFrameEstimate.Set(L, X, Y);
	*/
}

void SingleScattering(int Width, int Height)
{
	LAUNCH_DIMENSIONS(Width, Height, 1, 16, 8, 1)
	LAUNCH_CUDA_KERNEL_TIMED((KrnlSingleScattering<<<GridDim, BlockDim>>>()), "Single Scattering"); 
}

}
