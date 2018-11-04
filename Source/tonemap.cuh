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

#include "geometry.h"

namespace ExposureRender
{

// http://code.google.com/p/bilateralfilter/source/browse/trunk/BilateralFilter.cpp?r=3
// http://code.google.com/p/bilateralfilter/source/browse/trunk/main.cpp


HOST_DEVICE ColorRGBuc ToneMap(const ColorXYZAf& XYZA)
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

void ToneMap(Tracer& Tracer)
{
	LAUNCH_DIMENSIONS(Tracer.FrameBuffer.Resolution[0], Tracer.FrameBuffer.Resolution[1], 1, 16, 8, 1)
	LAUNCH_CUDA_KERNEL_TIMED((KrnlToneMap<<<GridDim, BlockDim>>>()), "Tone map");
}

}
