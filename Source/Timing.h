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

namespace ExposureRender
{

#define	MAX_NO_TIMINGS 64

struct EXPOSURE_RENDER_DLL ErKernelTiming
{
	char	Event[MAX_CHAR_SIZE];
	float	Duration;

	ErKernelTiming()
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, "Undefined");
		this->Duration = 0.0f;
	}

	ErKernelTiming(const char* pEvent, const float& Duration)
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, pEvent);
		this->Duration = Duration;
	}

	ErKernelTiming& operator = (const ErKernelTiming& Other)
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, Other.Event);
		this->Duration = Other.Duration;

		return *this;
	}

	void Init()
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, "");
		this->Duration = 0.0f;
	}
};

struct EXPOSURE_RENDER_DLL ErKernelTimings
{
	int				NoTimings;
	ErKernelTiming	Timings[MAX_NO_TIMINGS];
	
	ErKernelTimings& operator = (const ErKernelTimings& Other)
	{
		for (int i = 0; i < MAX_NO_TIMINGS; i++)
			this->Timings[i] = Other.Timings[i];

		this->NoTimings = Other.NoTimings;

		return *this;
	}

	void Add(const ErKernelTiming& ErKernelTiming)
	{
		this->Timings[this->NoTimings] = ErKernelTiming;
	}

	void Reset()
	{
		this->NoTimings = 0;
	}

	void Init()
	{
		this->NoTimings = 0;

		for (int i = 0; i < MAX_NO_TIMINGS; i++)
			this->Timings[i].Init();
	}
};

}
