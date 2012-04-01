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
#include "General.cuh"

namespace ExposureRender
{

DEVICE_NI ColorXYZf EvaluateTexture(const int& ID, Vec3f& UVW)
{
	ColorXYZf L;

	int id = 0;

	for (int i = 0; i < gTextures.NoTextures; i++)
	{
		if (gTextures.TextureList[i].ID == ID)
			id = i;
	}

	
	switch (gTextures.TextureList[id].Type)
	{
		case 0:
		{
			L.FromRGB(gTextures.TextureList[id].Procedural.UniformColor[0], gTextures.TextureList[id].Procedural.UniformColor[1], gTextures.TextureList[id].Procedural.UniformColor[2]);
			break;
		}

		case 1:
		{
			
			if (gTextures.TextureList[id].Image.pData != NULL)
			{
				UVW[0] = clamp(UVW[0], 0.0f, 1.0f);
				UVW[1] = clamp(UVW[1], 0.0f, 1.0f);

				const int ImageUV[2] =
				{
					(int)floorf(UVW[0] * (float)gTextures.TextureList[id].Image.Size[0]),
					(int)floorf(UVW[1] * (float)gTextures.TextureList[id].Image.Size[1])
				};
				
				const int PID = ImageUV[1] * gTextures.TextureList[id].Image.Size[0] + ImageUV[0];
				
				RGBA Color = gTextures.TextureList[id].Image.pData[PID];

				L.FromRGB(ONE_OVER_255 * (float)Color.Data[0], ONE_OVER_255 * (float)Color.Data[1], ONE_OVER_255 * (float)Color.Data[2]);
			}
			/**/
			break;
		}
	}

	return L;
}

}