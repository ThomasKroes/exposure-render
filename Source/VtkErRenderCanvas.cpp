/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "ErCoreStable.h"

#include "vtkErRenderCanvas.h"
#include "vtkErVolumeMapper.h"

vtkCxxRevisionMacro(vtkErRenderCanvas, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkErRenderCanvas);

vtkErRenderCanvas::vtkErRenderCanvas()
{
	this->VolumeMapper = NULL;
}

vtkErRenderCanvas::~vtkErRenderCanvas()
{
}

int vtkErRenderCanvas::RenderOpaqueGeometry(vtkViewport* viewport)
{
	this->Render(vtkRenderer::SafeDownCast(viewport));
	return 1;
}

int vtkErRenderCanvas::RenderTranslucentPolygonalGeometry(vtkViewport* viewport)
{
	this->Render(vtkRenderer::SafeDownCast(viewport));
	return 1;

}
void vtkErRenderCanvas::Render(vtkRenderer* pRenderer)
{
	if (!VolumeMapper)
	{
		vtkErrorMacro("This rendering canvas is not associated with an exposure render volume mapper, use vtkErRenderCanvas::SetVolumeMapper()!");
		return;
	}

	// if the display extent has not been set, then compute one
	this->ComputedDisplayExtent[0] = 0.0;
	this->ComputedDisplayExtent[1] = 300.0;
	this->ComputedDisplayExtent[2] = 0.0;
	this->ComputedDisplayExtent[3] = 300.0;
	this->ComputedDisplayExtent[4] = 0.0;
	this->ComputedDisplayExtent[5] = 0.0;

	/*
	glBegin(GL_QUADS);
		glTexCoord2i(1,1);
		glVertex3d(0, 0, 0);
		glTexCoord2i(0,1);
		glVertex3d(0, 300, 0);
		glTexCoord2i(0,0);
		glVertex3d(300, 300, 0);
		glTexCoord2i(1,0);
		glVertex3d(coordinatesD);
	glEnd();
	*/
}