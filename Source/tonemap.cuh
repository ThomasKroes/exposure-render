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

#include "geometry.h"

namespace ExposureRender
{

// http://code.google.com/p/bilateralfilter/source/browse/trunk/BilateralFilter.cpp?r=3
// http://code.google.com/p/bilateralfilter/source/browse/trunk/main.cpp


DEVICE ColorRGBuc ToneMap(const ColorXYZAf& XYZA)
{
	ColorRGBf RGBf = ColorRGBf::FromXYZAf(XYZA);

	RGBf[0] = 1.0f - expf(-(RGBf[0] / gpTracer->Camera.Exposure));
	RGBf[1] = 1.0f - expf(-(RGBf[1] / gpTracer->Camera.Exposure));
	RGBf[2] = 1.0f - expf(-(RGBf[2] / gpTracer->Camera.Exposure));

	RGBf.Clamp(0.0f, 1.0f);

	ColorRGBuc RGBuc;

	RGBuc[0] = 255.0f * RGBf[0];//powf(RGBf[0] / gpTracer->Camera.Gamma);
	RGBuc[1] = 255.0f * RGBf[1];//powf(RGBf[1] / gpTracer->Camera.Gamma);
	RGBuc[2] = 255.0f * RGBf[2];//powf(RGBf[2] / gpTracer->Camera.Gamma);

	return RGBuc;
}

KERNEL void KrnlToneMap()
{
	KERNEL_2D(gpTracer->FrameBuffer.Resolution[0], gpTracer->FrameBuffer.Resolution[1])

	const ColorRGBuc RGB = ToneMap(gpTracer->FrameBuffer.RunningEstimateXyza(IDx, IDy));

	gpTracer->FrameBuffer.DisplayEstimate(IDx, IDy)[0] = RGB[0];
	gpTracer->FrameBuffer.DisplayEstimate(IDx, IDy)[1] = RGB[1];
	gpTracer->FrameBuffer.DisplayEstimate(IDx, IDy)[2] = RGB[2];
	gpTracer->FrameBuffer.DisplayEstimate(IDx, IDy)[3] = gpTracer->FrameBuffer.RunningEstimateXyza(IDx, IDy)[3] * 255.0f;
}

void ToneMap(int Width, int Height)
{
	LAUNCH_DIMENSIONS(Width, Height, 1, 16, 8, 1)
	LAUNCH_CUDA_KERNEL_TIMED((KrnlToneMap<<<GridDim, BlockDim>>>()), "Tone map");
}

}