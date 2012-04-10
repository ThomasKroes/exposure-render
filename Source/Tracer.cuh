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

#include "Vector.cuh"
#include "Framebuffer.cuh"
#include "Filter.cuh"
#include "Camera.cuh"
#include "TransferFunction.cuh"
#include "SharedResources.cuh"

namespace ExposureRender
{

struct Tracer
{
	HOST Tracer()
	{
	}

	HOST ~Tracer()
	{
	}
		
	ScalarTransferFunction1D		Opacity1D;
	ColorTransferFunction1D			Diffuse1D;
	ColorTransferFunction1D			Specular1D;
	ScalarTransferFunction1D		Glossiness1D;
	ColorTransferFunction1D			Emission1D;

	Camera							Camera;
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

	HOST Tracer& Tracer::operator = (const Tracer& Other)
	{
		this->Opacity1D				= Other.Opacity1D;
		this->Diffuse1D				= Other.Diffuse1D;
		this->Specular1D			= Other.Specular1D;
		this->Glossiness1D			= Other.Glossiness1D;
		this->Emission1D			= Other.Emission1D;

		this->Camera				= Other.Camera;
		this->RenderSettings		= Other.RenderSettings;
		this->FrameEstimateFilter	= Other.FrameEstimateFilter;
		this->PostProcessingFilter	= Other.PostProcessingFilter;
		this->FrameBuffer			= Other.FrameBuffer;
		this->NoIterations			= Other.NoIterations;
		this->KernelTimings			= Other.KernelTimings;
		
		for (int i = 0; i < MAX_NO_VOLUMES; i++)
			this->VolumeIDs[i] = Other.VolumeIDs[i];

		for (int i = 0; i < MAX_NO_LIGHTS; i++)
			this->LightIDs[i] = Other.LightIDs[i];

		for (int i = 0; i < MAX_NO_OBJECTS; i++)
			this->ObjectIDs[i] = Other.ObjectIDs[i];

		for (int i = 0; i < MAX_NO_CLIPPING_OBJECTS; i++)
			this->ClippingObjectIDs[i] = Other.ClippingObjectIDs[i];

		this->NoVolumes				= Other.NoVolumes;
		this->NoLights				= Other.NoLights;
		this->NoObjects				= Other.NoObjects;
		this->NoClippingObjects		= Other.NoClippingObjects;

		return *this;
	}
};


typedef ResourceList<Tracer, MAX_NO_TRACERS> Tracers;

DEVICE Tracer* gpTracers = NULL;
CD int gActiveTracerID = 0;

SharedResources<Tracer, MAX_NO_TRACERS> gTracers("gpTracers");

}
