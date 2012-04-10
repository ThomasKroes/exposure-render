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

namespace ExposureRender
{

#define MAX_GAUSSIAN_FILTER_KERNEL_SIZE		256
#define MAX_BILATERAL_FILTER_KERNEL_SIZE	256

struct GaussianFilter
{
	int		KernelRadius;
	float	KernelD[MAX_BILATERAL_FILTER_KERNEL_SIZE];

	/*
	GaussianFilter Gaussian;
	
	Gaussian.KernelRadius = Filtering.FrameEstimateFilter.KernelRadius;

	const int KernelSize = (2 * Gaussian.KernelRadius) + 1;

	for (int i = 0; i < KernelSize; i++)
		Gaussian.KernelD[i] = Gauss2D(Filtering.FrameEstimateFilter.Sigma, Gaussian.KernelRadius - i, 0);

	gTracers[TracerID].FrameEstimateFilter = Gaussian;
	*/
};

struct BilateralFilter
{
	int		KernelRadius;
	float	KernelD[MAX_BILATERAL_FILTER_KERNEL_SIZE];
	float	GaussSimilarity[256];

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
};

HOST_DEVICE float Gauss2D(const float& Sigma, const int& X, const int& Y)
{
	return expf(-((X * X + Y * Y) / (2 * Sigma * Sigma)));
}

}