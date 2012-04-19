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
//	IntersectLights(R, SE[1], true);
	//IntersectObjects(R, SE[2]);

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

	Ray R;

	SampleCamera(gpTracer->Camera, R, IDx, IDy, Sample.CameraSample);

	SE = SampleRay(R, RNG);

	if (SE.Valid && SE.Type == ScatterEvent::Volume)
		Lv += UniformSampleOneLight(SE, RNG, Sample.LightingSample);
/*
	if (SE.Valid && SE.Type == ScatterEvent::Light)
		Lv += SE.Le;
	
	
	if (SE.Valid && SE.Type == ScatterEvent::Object)
		Lv += UniformSampleOneLight(SE, RNG, Sample.LightingSample);
	*/

	gpTracer->FrameBuffer.CudaFrameEstimate(IDx, IDy) = ColorXYZAf(Lv[0], Lv[1], Lv[2], SE.Valid ? 1.0f : 0.0f);
}

void SingleScattering(int Width, int Height)
{
	LAUNCH_DIMENSIONS(Width, Height, 1, 16, 8, 1)
	LAUNCH_CUDA_KERNEL_TIMED((KrnlSingleScattering<<<GridDim, BlockDim>>>()), "Single Scattering"); 
}

}
