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

#include "buffer.h"

namespace ExposureRender
{

class FrameBuffer
{
public:
	FrameBuffer(void) :
		Resolution(),
		FrameEstimate(Enums::Device),
		FrameEstimateTemp(Enums::Device),
		RunningEstimateXyza(Enums::Device),
		DisplayEstimate(Enums::Device),
		DisplayEstimateTemp(Enums::Device),
		DisplayEstimateFiltered(Enums::Device),
		RandomSeeds1(Enums::Device),
		RandomSeeds2(Enums::Device),
		RandomSeedsCopy1(Enums::Device),
		RandomSeedsCopy2(Enums::Device),
		HostDisplayEstimate(Enums::Host)
	{
	}

	void Resize(const Vec2i& Resolution)
	{
		if (this->Resolution == Resolution)
			return;

		this->Resolution = Resolution;

		this->FrameEstimate.Resize(this->Resolution);
		this->FrameEstimateTemp.Resize(this->Resolution);
		this->RunningEstimateXyza.Resize(this->Resolution);
		this->DisplayEstimate.Resize(this->Resolution);
		this->DisplayEstimateTemp.Resize(this->Resolution);
		this->DisplayEstimateFiltered.Resize(this->Resolution);
		this->RandomSeeds1.Resize(this->Resolution);
		this->RandomSeeds2.Resize(this->Resolution);
		this->RandomSeedsCopy1.Resize(this->Resolution);
		this->RandomSeedsCopy2.Resize(this->Resolution);
		this->HostDisplayEstimate.Resize(this->Resolution);

		RandomSeedsCopy1 = RandomSeeds1;
		RandomSeedsCopy2 = RandomSeeds2;

		this->Reset();
	}

	void Reset(void)
	{
		RandomSeeds1 = RandomSeedsCopy1;
		RandomSeeds2 = RandomSeedsCopy2;
	}

	void Free(void)
	{
		this->FrameEstimate.Free();
		this->FrameEstimateTemp.Free();
		this->RunningEstimateXyza.Free();
		this->DisplayEstimate.Free();
		this->DisplayEstimateTemp.Free();
		this->DisplayEstimateFiltered.Free();
		this->RandomSeeds1.Free();
		this->RandomSeeds2.Free();
		this->RandomSeedsCopy1.Free();
		this->RandomSeedsCopy2.Free();
		this->HostDisplayEstimate.Free();

		this->Resolution = Vec2i(0);
	}

	Vec2i					Resolution;
	Buffer2D<ColorXYZAf>	FrameEstimate;
	Buffer2D<ColorXYZAf>	FrameEstimateTemp;
	Buffer2D<ColorXYZAf>	RunningEstimateXyza;
	Buffer2D<ColorRGBAuc>	DisplayEstimate;
	Buffer2D<ColorRGBAuc>	DisplayEstimateTemp;
	Buffer2D<ColorRGBAuc>	DisplayEstimateFiltered;
	RandomSeedBuffer2D		RandomSeeds1;
	RandomSeedBuffer2D		RandomSeeds2;
	RandomSeedBuffer2D		RandomSeedsCopy1;
	RandomSeedBuffer2D		RandomSeedsCopy2;
	Buffer2D<ColorRGBAuc>	HostDisplayEstimate;
};

}