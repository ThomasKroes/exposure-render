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

#include "Buffer.cuh"
#include "Statistics.cuh"
#include "Sample.cuh"

namespace ExposureRender
{

class FrameBuffer
{
public:
	FrameBuffer(void) :
		Resolution(),
		CudaRunningEstimateXyza(),
		CudaFrameEstimate(),
		CudaFrameEstimateTemp(),
		CudaDisplayEstimate(),
		CudaDisplayEstimateTemp(),
		CudaDisplayEstimateFiltered(),
		CudaRandomSeeds1(),
		CudaRandomSeeds2(),
		CudaRandomSeedsCopy1(),
		CudaRandomSeedsCopy2(),
		CudaRunningStatistics(),
		CudaVariance(),
		BenchmarkEstimateRgbaLdr(),
		RmsError()//,
//		CudaMetroSamples(),
//		CudaNoIterations(),
//		CudaPixelLuminance()
	{
	}

	void Resize(Resolution2i Resolution)
	{
		if (this->Resolution == Resolution)
			return;

		this->Resolution = Resolution;

		this->CudaRunningEstimateXyza.Resize(this->Resolution);
		this->CudaFrameEstimate.Resize(this->Resolution);
		this->CudaFrameEstimateTemp.Resize(this->Resolution);
		this->CudaDisplayEstimate.Resize(this->Resolution);
		this->CudaDisplayEstimateTemp.Resize(this->Resolution);
		this->CudaDisplayEstimateFiltered.Resize(this->Resolution);
		this->CudaRandomSeeds1.Resize(this->Resolution);
		this->CudaRandomSeeds2.Resize(this->Resolution);
		this->CudaRandomSeedsCopy1.Resize(this->Resolution);
		this->CudaRandomSeedsCopy2.Resize(this->Resolution);
		this->CudaRunningStatistics.Resize(this->Resolution);
		this->CudaVariance.Resize(this->Resolution);
		this->BenchmarkEstimateRgbaLdr.Resize(this->Resolution);
		this->RmsError.Resize(this->Resolution);
//		this->CudaMetroSamples.Resize(Resolution2i(MetroSize));
//		this->CudaNoIterations.Resize(this->Resolution);
//		this->CudaPixelLuminance.Resize(this->Resolution);

		CUDA::MemCopyDeviceToDevice(CudaRandomSeeds1.GetPtr(), CudaRandomSeedsCopy1.GetPtr(), CudaRandomSeedsCopy1.GetNoElements());
		CUDA::MemCopyDeviceToDevice(CudaRandomSeeds2.GetPtr(), CudaRandomSeedsCopy2.GetPtr(), CudaRandomSeedsCopy2.GetNoElements());

		this->Reset();
	}

	void Reset(void)
	{
//		this->CudaRunningEstimateXyza.Reset();
//		this->CudaFrameEstimate.Reset();
//		this->CudaFrameEstimateTemp.Reset();
//		this->CudaDisplayEstimate.Reset();
//		this->CudaDisplayEstimateTemp.Reset();
//		this->CudaDisplayEstimateFiltered.Reset();
//		this->CudaRandomSeeds1.Reset();
//		this->CudaRandomSeeds2.Reset();
//		this->CudaRandomSeedsCopy1.Reset();
//		this->CudaRandomSeedsCopy2.Reset();
//		this->CudaRunningStatistics.Reset();
//		this->CudaVariance.Reset();
//		this->BenchmarkEstimateRgbaLdr.Reset();
//		this->RmsError.Reset();
//		this->CudaMetroSamples.Reset();
//		this->CudaNoIterations.Reset();
//		this->CudaPixelLuminance.Reset();

		CUDA::MemCopyDeviceToDevice(CudaRandomSeedsCopy1.GetPtr(), CudaRandomSeeds1.GetPtr(), CudaRandomSeedsCopy1.GetNoElements());
		CUDA::MemCopyDeviceToDevice(CudaRandomSeedsCopy2.GetPtr(), CudaRandomSeeds2.GetPtr(), CudaRandomSeedsCopy2.GetNoElements());
	}

	void Free(void)
	{
		this->CudaRunningEstimateXyza.Free();
		this->CudaFrameEstimate.Free();
		this->CudaFrameEstimateTemp.Free();
		this->CudaDisplayEstimate.Free();
		this->CudaDisplayEstimateTemp.Free();
		this->CudaDisplayEstimateFiltered.Free();
		this->CudaRandomSeeds1.Free();
		this->CudaRandomSeeds2.Free();
		this->CudaRandomSeedsCopy1.Free();
		this->CudaRandomSeedsCopy2.Free();
		this->CudaRunningStatistics.Free();
		this->CudaVariance.Free();
		this->BenchmarkEstimateRgbaLdr.Free();
		this->RmsError.Free();
		this->CudaMetroSamples.Free();
		this->CudaNoIterations.Free();
		this->CudaPixelLuminance.Free();

		this->Resolution = Resolution2i();
	}

	HOST_DEVICE int GetWidth(void) const
	{
		return this->Resolution[0];
	}

	HOST_DEVICE int GetHeight(void) const
	{
		return this->Resolution[1];
	}

	Resolution2i							Resolution;
	
	CCudaBuffer2D<ColorXYZAf, false>		CudaRunningEstimateXyza;
	CCudaBuffer2D<ColorRGBAuc, false>		CudaDisplayEstimate;
	CCudaBuffer2D<ColorRGBAuc, false>		CudaDisplayEstimateTemp;
	CCudaBuffer2D<ColorRGBAuc, false>		CudaDisplayEstimateFiltered;

	CCudaBuffer2D<ColorXYZAf, false>		CudaFrameEstimate;
	CCudaBuffer2D<ColorXYZAf, false>		CudaFrameEstimateTemp;
	
	CCudaRandomBuffer2D						CudaRandomSeeds1;
	CCudaRandomBuffer2D						CudaRandomSeeds2;
	CCudaRandomBuffer2D						CudaRandomSeedsCopy1;
	CCudaRandomBuffer2D						CudaRandomSeedsCopy2;

	// Variance
	CCudaBuffer2D<RunningStats, false>		CudaRunningStatistics;
	CCudaBuffer2D<float, false>				CudaVariance;

	CCudaBuffer2D<ColorRGBAuc, false>		BenchmarkEstimateRgbaLdr;
	CCudaBuffer2D<float, false>				RmsError;

	CCudaBuffer2D<MetroSample, false>		CudaMetroSamples;
	CCudaBuffer2D<int, false>				CudaNoIterations;
	CCudaBuffer2D<float, false>				CudaPixelLuminance;
};

}