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

#include "Dll.h"
#include "Defines.h"

class EXPOSURE_RENDER_DLL CFlags
{
private:
	long	m_Bits;

public:
	CFlags(void) :
		m_Bits(0)
	{
	}

	CFlags(long b) :
		m_Bits(b)
	{
	}

	virtual ~CFlags(void)
	{
	};

	HOD CFlags& CFlags::operator=(const CFlags& Other)
	{
		m_Bits = Other.m_Bits;

		return *this;
	}

	void SetFlag(const long Flag)
	{
		m_Bits |= Flag;
	};

	long Get(void)
	{
		return m_Bits;
	};

	void ClearFlag(const long Flag)
	{
		m_Bits &= ~Flag;
	};

	void ClearAllFlags(void)
	{
		m_Bits = 0;
	};

	void SetConditional(const long Flag, const int YesNo)
	{
		YesNo ? SetFlag(Flag) : ClearFlag(Flag);
	};

	bool HasFlag(const long Flag) const
	{
		return (m_Bits & Flag) != 0;
	};

	int All(const long Flag) const
	{
		return (m_Bits & Flag) == Flag;
	};

	int Not(const long Flag) const
	{
		return (m_Bits & Flag) == 0;
	};

	void ToggleFlag(const long Flag)
	{
		if (HasFlag(Flag))
		{
			ClearFlag(Flag);
		}
		else
		{
			SetFlag(Flag);
		}
	};
};