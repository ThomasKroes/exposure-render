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

#include "TransferFunction.h"
#include "Camera.h"
#include "RenderSettings.h"

#include "Framebuffer.cuh"
#include "Filter.cuh"

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
	RenderSettings					RenderSettings;
	FrameBuffer						FrameBuffer;
	int								NoIterations;

	int								VolumeID;
	Indices							LightIDs;
	Indices							ObjectIDs;
	Indices							ClippingObjectIDs;
	
	GaussianFilter					FrameEstimateFilter;
	BilateralFilter					PostProcessingFilter;

	HOST Tracer& Tracer::operator = (const Tracer& Other)
	{
		this->Opacity1D				= Other.Opacity1D;
		this->Diffuse1D				= Other.Diffuse1D;
		this->Specular1D			= Other.Specular1D;
		this->Glossiness1D			= Other.Glossiness1D;
		this->Emission1D			= Other.Emission1D;
		this->Camera				= Other.Camera;
		this->RenderSettings		= Other.RenderSettings;
		this->FrameBuffer			= Other.FrameBuffer;
		this->NoIterations			= Other.NoIterations;
		this->VolumeID				= Other.VolumeID;
		this->LightIDs				= Other.LightIDs;
		this->ObjectIDs				= Other.ObjectIDs;
		this->ClippingObjectIDs		= Other.ClippingObjectIDs;
		this->FrameEstimateFilter	= Other.FrameEstimateFilter;
		this->PostProcessingFilter	= Other.PostProcessingFilter;

		return *this;
	}
	
	HOST void BindIDs(Indices SourceIDs, Indices& TargetIDs, map<int, int> HashMap)
	{
		for (int i = 0; i < SourceIDs.Count; i++)
			TargetIDs[i] = HashMap[SourceIDs[i]];

		TargetIDs.Count = SourceIDs.Count;
	}

	HOST void BindLightIDs(Indices LightIDs, map<int, int> HashMap)
	{
		BindIDs(LightIDs, this->LightIDs, HashMap);
	}

	HOST void BindObjectIDs(Indices ObjectIDs, map<int, int> HashMap)
	{
		BindIDs(ObjectIDs, this->ObjectIDs, HashMap);
	}

	HOST void BindClippingObjectIDs(Indices ClippingObjectIDs, map<int, int> HashMap)
	{
		BindIDs(ClippingObjectIDs, this->ClippingObjectIDs, HashMap);
	}

	HOST void BindRenderSettings(const ExposureRender::RenderSettings& RS)
	{
		this->RenderSettings = RenderSettings;

		// FIXME

		this->FrameEstimateFilter.KernelRadius = this->RenderSettings.Filtering.FrameEstimateFilterParams.KernelRadius;

		const int KernelSize = (2 * this->FrameEstimateFilter.KernelRadius) + 1;

		for (int i = 0; i < KernelSize; i++)
			FrameEstimateFilter.KernelD[i] = 1.0f;//Gauss2D(RS.Filtering.FrameEstimateFilterParams.Sigma, RS.Filtering.FrameEstimateFilterParams.KernelRadius - i, 0);

		/*
		BilateralFilter Bilateral;

		const int SigmaMax = (int)max(Filtering.PostProcessingFilter.SigmaD, Filtering.PostProcessingFilter.SigmaR);
		
		Bilateral.KernelRadius = (int)ceilf(2.0f * (float)SigmaMax);  

		const float TwoSigmaRSquared = 2 * Filtering.PostProcessingFilter.SigmaR * Filtering.PostProcessingFilter.SigmaR;

		const int kernelSize = Bilateral.KernelRadius * 2 + 1;
		const int center = (kernelSize - 1) / 2;

		for (int x = -center; x < -center + kernelSize; x++)
			Bilateral.KernelD[x + center] = Gauss2D(Filtering.PostProcessingFilter.SigmaD, x, 0);

		for (int i = 0; i < 256; i++)
			Bilateral.GaussSimilarity[i] = expf(-((float)i / TwoSigmaRSquared));

		gTracers[TracerID].PostProcessingFilter = Bilateral;
		*/
	}
};

}
