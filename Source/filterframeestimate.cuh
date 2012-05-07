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

#include "color.h"
#include "filter.h"
#include "tracer.h"

namespace ExposureRender
{

HOST_DEVICE_NI float Gauss2D(const float& Sigma, const int& X, const int& Y)
{
	return expf(-((X * X + Y * Y) / (2 * Sigma * Sigma)));
}

KERNEL void KrnlFilterFrameEstimate(int KernelRadius, float Sigma)
{
	KERNEL_2D(gpTracer->FrameBuffer.Resolution[0], gpTracer->FrameBuffer.Resolution[1])

	int Range[2][2];

	Range[0][0] = max((int)ceilf(IDx - KernelRadius), 0);
	Range[0][1] = min((int)floorf(IDx + KernelRadius), gpTracer->FrameBuffer.Resolution[0] - 1);
	Range[1][0] = max((int)ceilf(IDy - KernelRadius), 0);
	Range[1][1] = min((int)floorf(IDy + KernelRadius), gpTracer->FrameBuffer.Resolution[1] - 1);

	ColorXYZAf Sum		= ColorXYZAf::Black();
	float Weight		= 0.0f;
	float TotalWeight	= 0.0f;

	for (int y = Range[1][0]; y <= Range[1][1]; y++)
	{
		for (int x = Range[0][0]; x <= Range[0][1]; x++)
		{
			Weight		= Gauss2D(Sigma, x - IDx, y - IDy);
			Sum			+= gpTracer->FrameBuffer.FrameEstimate(x, y) * Weight;
			TotalWeight	+= Weight;
		}
	}

	Sum[3] = gpTracer->FrameBuffer.FrameEstimate(IDx, IDy)[3];

	if (TotalWeight > 0.0f)
		gpTracer->FrameBuffer.FrameEstimateTemp(IDx, IDy) = Sum / TotalWeight;
	else
		gpTracer->FrameBuffer.FrameEstimateTemp(IDx, IDy) = ColorXYZAf::Black();
}

void FilterFrameEstimate(Tracer& Tracer)
{
	LAUNCH_DIMENSIONS(Tracer.FrameBuffer.Resolution[0], Tracer.FrameBuffer.Resolution[1], 1, 8, 8, 1)
	LAUNCH_CUDA_KERNEL_TIMED((KrnlFilterFrameEstimate<<<GridDim, BlockDim>>>(1, 1.0f)), "Gaussian filter (Horizontal)");

	Tracer.FrameBuffer.FrameEstimateTemp.Dirty = true;

	Tracer.FrameBuffer.FrameEstimate = Tracer.FrameBuffer.FrameEstimateTemp;
}

}
