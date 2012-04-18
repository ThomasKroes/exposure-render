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

#include "Geometry.cuh"

namespace ExposureRender
{

// http://code.google.com/p/bilateralfilter/source/browse/trunk/BilateralFilter.cpp?r=3
// http://code.google.com/p/bilateralfilter/source/browse/trunk/main.cpp


DEVICE ColorRGBuc ToneMap(const ColorXYZAf& XYZA)
{
	ColorRGBf RGBf = ColorRGBf::FromXYZAf(XYZA);

	RGBf[0] = 1.0f - expf(-(RGBf[0] * gpTracer->Camera.InvExposure));
	RGBf[1] = 1.0f - expf(-(RGBf[1] * gpTracer->Camera.InvExposure));
	RGBf[2] = 1.0f - expf(-(RGBf[2] * gpTracer->Camera.InvExposure));

	RGBf.Clamp(0.0f, 1.0f);

	ColorRGBuc RGBuc;

	RGBuc[0] = 255.0f * powf(RGBuc[0], gpTracer->Camera.InvGamma);
	RGBuc[1] = 255.0f * powf(RGBuc[1], gpTracer->Camera.InvGamma);
	RGBuc[2] = 255.0f * powf(RGBuc[2], gpTracer->Camera.InvGamma);

	return RGBuc;
}

KERNEL void KrnlToneMap()
{
	KERNEL_2D(gpTracer->FrameBuffer.Resolution[0], gpTracer->FrameBuffer.Resolution[1])

	const ColorRGBuc RGB = ToneMap(gpTracer->FrameBuffer.CudaRunningEstimateXyza(IDx, IDy));

	gpTracer->FrameBuffer.CudaDisplayEstimate(IDx, IDy)[0] = gpTracer->FrameBuffer.CudaRunningEstimateXyza(IDx, IDy)[0];
	gpTracer->FrameBuffer.CudaDisplayEstimate(IDx, IDy)[1] = gpTracer->FrameBuffer.CudaRunningEstimateXyza(IDx, IDy)[1];
	gpTracer->FrameBuffer.CudaDisplayEstimate(IDx, IDy)[2] = gpTracer->FrameBuffer.CudaRunningEstimateXyza(IDx, IDy)[2];
	gpTracer->FrameBuffer.CudaDisplayEstimate(IDx, IDy)[3] = 255;
}

void ToneMap(int Width, int Height)
{
	LAUNCH_DIMENSIONS(Width, Height, 1, 16, 8, 1)
	LAUNCH_CUDA_KERNEL_TIMED((KrnlToneMap<<<GridDim, BlockDim>>>()), "Tone map");
}

}