/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Defines.cuh"

namespace ExposureRender
{

DEVICE_NI ColorXYZf EvaluateTexture(const int& ID, const Vec3f& UVW)
{
	Texture& T = Texture();

	for (int i = 0; i < gTextures.NoTextures; i++)
	{
		if (gTextures.TextureList[i].ID == ID)
			T = gTextures.TextureList[i];
	}

	switch (T.Type)
	{
		case 0:
		{
			if (T.Image.pData != NULL)
			{
				const int ImageUV[2] = { (int)floorf(UVW[0] * T.Image.Size[0]), (int)floorf(UVW[1] * T.Image.Size[1]) };
				
				const int PID = ImageUV[1] * T.Image.Size[0] + ImageUV[0];
				
				RGBA Color = T.Image.pData[PID];

				ColorXYZf L;

				return L.FromRGB(Color.Data[0], Color.Data[1], Color.Data[2]);
			}
		}

		case 1:
		{
		}
	}

	return ColorXYZf(0.0f);
}

}