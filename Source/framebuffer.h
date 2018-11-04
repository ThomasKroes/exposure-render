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

#include "buffer2d.h"

namespace ExposureRender
{

class FrameBuffer
{
public:
	FrameBuffer(void) :
		Resolution(),
		FrameEstimate(Enums::Device, "Frame Estimate XYZA"),
		FrameEstimateTemp(Enums::Device, "Temp Frame Estimate XYZA"),
		RunningEstimateXyza(Enums::Device, "Running Estimate XYZA"),
		DisplayEstimate(Enums::Device, "Display Estimate RGBA"),
		DisplayEstimateTemp(Enums::Device, "Temp Display Estimate RGBA"),
		DisplayEstimateFiltered(Enums::Device, "Filtered Display Estimate RGBA"),
		RandomSeeds1(Enums::Device, "Random Seeds 1"),
		RandomSeeds2(Enums::Device, "Random Seeds 2"),
		RandomSeedsCopy1(Enums::Device, "Random Seeds 1 (Cache)"),
		RandomSeedsCopy2(Enums::Device, "Random Seeds 2 (Cache)"),
		HostDisplayEstimate(Enums::Host, "Display Estimate RGBA")
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
