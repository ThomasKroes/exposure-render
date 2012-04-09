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

DEVICE_NI float EvaluatePLF(const float& Intensity, ErPiecewiseLinearFunction& PLF)
{
	if (Intensity < PLF.NodeRange.Min)
		return PLF.Data[0];

	if (Intensity > PLF.NodeRange.Max)
		return PLF.Data[PLF.Count - 1];

	for (int i = 1; i < PLF.Count; i++)
	{
		float P1 = PLF.Position[i - 1];
		float P2 = PLF.Position[i];
		float DeltaP = P2 - P1;
		float LerpT = (Intensity - P1) / DeltaP;

		if (Intensity >= P1 && Intensity < P2)
			return Lerp(LerpT, PLF.Data[i - 1], PLF.Data[i]);
	}

	return 0.0f;
}

DEVICE_NI float EvaluateScalarTransferFunction1D(const float& Intensity, ErScalarTransferFunction1D& TF)
{
	return EvaluatePLF(Intensity, TF.PLF[0]);
}

DEVICE_NI ColorXYZf EvaluateColorTransferFunction1D(const float& Intensity, ErColorTransferFunction1D& TF)
{
	return ColorXYZf(EvaluatePLF(Intensity, TF.PLF[0]), EvaluatePLF(Intensity, TF.PLF[1]), EvaluatePLF(Intensity, TF.PLF[2]));
}

}