/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or witDEVut modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software witDEVut specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT DEVLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT DEVLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) DEVWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Defines.cuh"
#include "Vector.cuh"
#include "General.cuh"
#include "CudaUtilities.cuh"
#include "Framebuffer.cuh"
#include "Filter.cuh"

namespace ExposureRender
{

struct Tracer
{
	ErScalarTransferFunction1D		Opacity1D;
	ErColorTransferFunction1D		Diffuse1D;
	ErColorTransferFunction1D		Specular1D;
	ErScalarTransferFunction1D		Glossiness1D;
	ErColorTransferFunction1D		Emission1D;

	ErCamera						Camera;
	ErRenderSettings				RenderSettings;
	GaussianFilter					FrameEstimateFilter;
	BilateralFilter					PostProcessingFilter;
	FrameBuffer						FrameBuffer;
	int								NoIterations;
	ErKernelTimings					KernelTimings;

	int								VolumeIDs[MAX_NO_VOLUMES];
	int								LightIDs[MAX_NO_LIGHTS];
	int								ObjectIDs[MAX_NO_OBJECTS];
	int								ClippingObjectIDs[MAX_NO_CLIPPING_OBJECTS];

	int								NoVolumes;
	int								NoLights;
	int								NoObjects;
	int								NoClippingObjects;
};


__device__ Tracer*	gpTracer = NULL;

}
