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
