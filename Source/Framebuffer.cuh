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

#include "CudaUtilities.h"

#include "Buffer.cuh"
#include "Statistics.cuh"

class FrameBuffer
{
public:
	FrameBuffer(void) :
		Resolution(0, 0),
		CudaRunningEstimateXyza(),
		CudaFrameEstimateXyza(),
		CudaFrameBlurXyza(),
		CudaRunningEstimateRgbaLdr(),
		CudaDisplayEstimateRgbLdr(),
		CudaRandomSeeds1(),
		CudaRandomSeeds2(),
		HostDisplayEstimateRgbaLdr(),
		HostFrameEstimate(),
		HostDepthBuffer(),
		BenchmarkEstimateRgbaLdr(),
		RmsError()
	{
	}

	~FrameBuffer(void)
	{
		Free();
	}
	
	void Resize(const CResolution2D& Resolution)
	{
		if (this->Resolution == Resolution)
			return;

		this->Resolution = Resolution;

		this->CudaRunningEstimateXyza.Resize(this->Resolution);
		this->CudaFrameEstimateXyza.Resize(this->Resolution);
		this->CudaFrameBlurXyza.Resize(this->Resolution);
		this->CudaRunningEstimateRgbaLdr.Resize(this->Resolution);
		this->CudaDisplayEstimateRgbLdr.Resize(this->Resolution);
		this->CudaRandomSeeds1.Resize(this->Resolution);
		this->CudaRandomSeeds2.Resize(this->Resolution);
		this->HostDisplayEstimateRgbaLdr.Resize(this->Resolution);
		this->HostFrameEstimate.Resize(this->Resolution);
		this->HostDepthBuffer.Resize(this->Resolution);
		this->CudaRunningStats.Resize(this->Resolution);
		this->BenchmarkEstimateRgbaLdr.Resize(this->Resolution);
		this->RmsError.Resize(this->Resolution);
	}

	void Reset(void)
	{
		this->CudaRunningEstimateXyza.Reset();
		this->CudaFrameEstimateXyza.Reset();
//		this->CudaFrameBlurXyza.Reset();
		this->CudaRunningEstimateRgbaLdr.Reset();
		this->CudaDisplayEstimateRgbLdr.Reset();
		this->HostDisplayEstimateRgbaLdr.Reset();
//		this->CudaRandomSeeds1.Reset();
//		this->CudaRandomSeeds2.Reset();
//		this->CudaRunningStats.Reset();
//		this->BenchmarkEstimateRgbaLdr.Reset();
//		this->RmsError.Reset();
	}

	void Free(void)
	{
		this->CudaRunningEstimateXyza.Free();
		this->CudaFrameEstimateXyza.Free();
		this->CudaFrameBlurXyza.Free();
		this->CudaRunningEstimateRgbaLdr.Free();
		this->CudaDisplayEstimateRgbLdr.Free();
		this->CudaRandomSeeds1.Free();
		this->CudaRandomSeeds2.Free();
		this->HostDisplayEstimateRgbaLdr.Free();
		this->HostFrameEstimate.Free();
		this->HostDepthBuffer.Free();
		this->CudaRunningStats.Free();
		this->BenchmarkEstimateRgbaLdr.Free();
		this->RmsError.Free();

		this->Resolution.Set(Vec2i(0, 0));
	}

	HOST_DEVICE int GetWidth(void) const
	{
		return this->Resolution.GetResX();
	}

	HOST_DEVICE int GetHeight(void) const
	{
		return this->Resolution.GetResY();
	}

	CResolution2D							Resolution;
	
	// Running estimate
	CCudaBuffer2D<ColorXYZAf, false>		CudaRunningEstimateXyza;
	CCudaBuffer2D<ColorRGBAuc, false>		CudaRunningEstimateRgbaLdr;

	// Frame estimate
	CCudaBuffer2D<ColorXYZAf, false>		CudaFrameEstimateXyza;
	CCudaBuffer2D<ColorXYZAf, false>		CudaFrameBlurXyza;
	
	CCudaBuffer2D<ColorRGBuc, false>		CudaDisplayEstimateRgbLdr;

	// Random seeds
	CCudaRandomBuffer2D						CudaRandomSeeds1;
	CCudaRandomBuffer2D						CudaRandomSeeds2;

	// Host buffers
	CHostBuffer2D<ColorRGBAuc>				HostDisplayEstimateRgbaLdr;
	CHostBuffer2D<ColorRGBAuc>				HostFrameEstimate;
	CHostBuffer2D<ColorRGBAuc>				HostDepthBuffer;

	// Variance
	CCudaBuffer2D<RunningStats, false>		CudaRunningStats;

	CCudaBuffer2D<ColorRGBAuc, false>		BenchmarkEstimateRgbaLdr;
	CCudaBuffer2D<float, false>				RmsError;
};