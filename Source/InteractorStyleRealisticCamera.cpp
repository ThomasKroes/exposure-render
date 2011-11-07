/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "InteractorStyleRealisticCamera.h"
#include "RenderThread.h"

// Mouse button flags
CFlags gMouseButtonFlags;

vtkStandardNewMacro(vtkRealisticCameraStyle);

float vtkRealisticCameraStyle::m_OrbitSpeed				= 1.0f;
float vtkRealisticCameraStyle::m_PanSpeed				= 1.0f;
float vtkRealisticCameraStyle::m_ZoomSpeed				= 1000.0f;
float vtkRealisticCameraStyle::m_ContinuousZoomSpeed	= 0.0000001f;
float vtkRealisticCameraStyle::m_ApertureSpeed			= 0.001f;
float vtkRealisticCameraStyle::m_FovSpeed				= 0.5f;

void vtkRealisticCameraStyle::OnLeftButtonDown(void) 
{
	vtkInteractorStyleUser::OnLeftButtonDown();

	gMouseButtonFlags.ClearAllFlags();
	gMouseButtonFlags.SetFlag(Left);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void vtkRealisticCameraStyle::OnLeftButtonUp(void) 
{
	vtkInteractorStyleUser::OnLeftButtonUp();

	gMouseButtonFlags.ClearFlag(Left);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void vtkRealisticCameraStyle::OnRightButtonDown(void) 
{
	vtkInteractorStyleUser::OnRightButtonDown();
		
	gMouseButtonFlags.ClearAllFlags();
	gMouseButtonFlags.SetFlag(Right);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void vtkRealisticCameraStyle::OnRightButtonUp(void) 
{
	vtkInteractorStyleUser::OnRightButtonUp();

	gMouseButtonFlags.ClearFlag(Right);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void vtkRealisticCameraStyle::OnMiddleButtonDown(void) 
{
	vtkInteractorStyleUser::OnMiddleButtonDown();

	gMouseButtonFlags.ClearAllFlags();
	gMouseButtonFlags.SetFlag(Middle);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void vtkRealisticCameraStyle::OnMiddleButtonUp(void) 
{
	vtkInteractorStyleUser::OnMiddleButtonUp();

	gMouseButtonFlags.ClearFlag(Middle);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void vtkRealisticCameraStyle::OnMouseWheelForward(void)
{
	vtkInteractorStyleUser::OnMouseWheelForward();

//	gScene.m_Camera.Zoom(-m_ZoomSpeed);

	// Flag the camera as dirty, this will restart the rendering
//	gScene.m_DirtyFlags.SetFlag(CameraDirty);
};
	
void vtkRealisticCameraStyle::OnMouseWheelBackward(void)
{
	vtkInteractorStyleUser::OnMouseWheelBackward();

//	gScene.m_Camera.Zoom(m_ZoomSpeed);

	// Flag the camera as dirty, this will restart the rendering
//	gScene.m_DirtyFlags.SetFlag(CameraDirty);
};

void vtkRealisticCameraStyle::OnMouseMove(void) 
{
	// Forward events
	vtkInteractorStyleUser::OnMouseMove();

	// Orbiting
	if (gMouseButtonFlags.HasFlag(Left))
	{
		if (GetShiftKey() && GetCtrlKey())
		{
			GetLastPos(m_NewPos[0], m_NewPos[1]);

//			gCamera.GetFocus().SetFocalDistance(max(0.0f, gScene.m_Camera.m_Focus.m_FocalDistance + m_ApertureSpeed * (float)(m_NewPos[1] - m_OldPos[1])));

			GetLastPos(m_OldPos[0], m_OldPos[1]);

			// Flag the camera as dirty, this will restart the rendering
//			gScene.m_DirtyFlags.SetFlag(CameraDirty);
		}
		else
		{
			if (GetShiftKey())
			{
				GetLastPos(m_NewPos[0], m_NewPos[1]);

//				gCamera.GetAperture().SetSize(max(0.0f, gScene.m_Camera.m_Aperture.m_Size + m_ApertureSpeed * (float)(m_NewPos[1] - m_OldPos[1])));

				GetLastPos(m_OldPos[0], m_OldPos[1]);

				// Flag the camera as dirty, this will restart the rendering
//				gScene.m_DirtyFlags.SetFlag(CameraDirty);
			}
			else if (GetCtrlKey())
			{
				GetLastPos(m_NewPos[0], m_NewPos[1]);

//				gCamera.GetProjection().SetFieldOfView(max(0.0f, gScene.m_Camera.m_FovV - m_FovSpeed * (float)(m_NewPos[1] - m_OldPos[1])));

				GetLastPos(m_OldPos[0], m_OldPos[1]);

				/// Flag the camera as dirty, this will restart the rendering
//				gScene.m_DirtyFlags.SetFlag(CameraDirty);
			}
			else
			{
				GetLastPos(m_NewPos[0], m_NewPos[1]);

//				gScene.m_Camera.Orbit(0.6f * m_OrbitSpeed * (float)(m_NewPos[1] - m_OldPos[1]), -m_OrbitSpeed * (float)(m_NewPos[0] - m_OldPos[0]));

				GetLastPos(m_OldPos[0], m_OldPos[1]);

				// Flag the camera as dirty, this will restart the rendering
//				gScene.m_DirtyFlags.SetFlag(CameraDirty);
			}
		}
	}

	// Panning
	if (gMouseButtonFlags.HasFlag(Middle))
	{
		GetLastPos(m_NewPos[0], m_NewPos[1]);

//		gScene.m_Camera.Pan(m_PanSpeed * (float)(m_NewPos[1] - m_OldPos[1]), -m_PanSpeed * ((float)(m_NewPos[0] - m_OldPos[0])));

		GetLastPos(m_OldPos[0], m_OldPos[1]);

		// Flag the camera as dirty, this will restart the rendering
//		gScene.m_DirtyFlags.SetFlag(CameraDirty);
	}

	// Zooming
	if (gMouseButtonFlags.HasFlag(Right))
	{
		GetLastPos(m_NewPos[0], m_NewPos[1]);

//		gScene.m_Camera.Zoom(-(float)(m_NewPos[1] - m_OldPos[1]));

		GetLastPos(m_OldPos[0], m_OldPos[1]);

		// Flag the camera as dirty, this will restart the rendering
//		gScene.m_DirtyFlags.SetFlag(CameraDirty);
	}
}