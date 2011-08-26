
#include "InteractorStyleRealisticCamera.h"
#include "RenderThread.h"

// Mouse button flags
CFlags gMouseButtonFlags;

vtkStandardNewMacro(CInteractorStyleRealisticCamera);

float CInteractorStyleRealisticCamera::m_OrbitSpeed				= 1000.0f;
float CInteractorStyleRealisticCamera::m_PanSpeed				= 1000.0f;
float CInteractorStyleRealisticCamera::m_ZoomSpeed				= 1000.0f;
float CInteractorStyleRealisticCamera::m_ContinuousZoomSpeed	= 0.000001f;
float CInteractorStyleRealisticCamera::m_ApertureSpeed			= 0.001f;
float CInteractorStyleRealisticCamera::m_FovSpeed				= 0.5f;

void CInteractorStyleRealisticCamera::OnLeftButtonDown(void) 
{
	vtkInteractorStyleUser::OnLeftButtonDown();

	gMouseButtonFlags.ClearAllFlags();
	gMouseButtonFlags.SetFlag(Left);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void CInteractorStyleRealisticCamera::OnLeftButtonUp(void) 
{
	vtkInteractorStyleUser::OnLeftButtonUp();

	gMouseButtonFlags.ClearFlag(Left);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void CInteractorStyleRealisticCamera::OnRightButtonDown(void) 
{
	vtkInteractorStyleUser::OnRightButtonDown();
		
	gMouseButtonFlags.ClearAllFlags();
	gMouseButtonFlags.SetFlag(Right);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void CInteractorStyleRealisticCamera::OnRightButtonUp(void) 
{
	vtkInteractorStyleUser::OnRightButtonUp();

	gMouseButtonFlags.ClearFlag(Right);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void CInteractorStyleRealisticCamera::OnMiddleButtonDown(void) 
{
	vtkInteractorStyleUser::OnMiddleButtonDown();

	gMouseButtonFlags.ClearAllFlags();
	gMouseButtonFlags.SetFlag(Middle);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void CInteractorStyleRealisticCamera::OnMiddleButtonUp(void) 
{
	vtkInteractorStyleUser::OnMiddleButtonUp();

	gMouseButtonFlags.ClearFlag(Middle);
	GetLastPos(m_OldPos[0], m_OldPos[1]);
}

void CInteractorStyleRealisticCamera::OnMouseWheelForward(void)
{
	vtkInteractorStyleUser::OnMouseWheelForward();

	if (!gpScene)
		return;

	gpScene->m_Camera.Zoom(-m_ZoomSpeed);

	// Flag the camera as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(CameraDirty);
};
	
void CInteractorStyleRealisticCamera::OnMouseWheelBackward(void)
{
	vtkInteractorStyleUser::OnMouseWheelBackward();

	if (!gpScene)
		return;

	gpScene->m_Camera.Zoom(m_ZoomSpeed);

	// Flag the camera as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(CameraDirty);
};

void CInteractorStyleRealisticCamera::OnMouseMove(void) 
{
	// Forward events
	vtkInteractorStyleUser::OnMouseMove();

	if (!gpScene)
		return;

	gpScene->m_Camera.m_Focus.m_FocalDistance = 1.0f;

	// Orbiting
	if (gMouseButtonFlags.HasFlag(Left))
	{
		if (GetShiftKey())
		{
			GetLastPos(m_NewPos[0], m_NewPos[1]);

			gpScene->m_Camera.m_Aperture.m_Size = max(0.0f, gpScene->m_Camera.m_Aperture.m_Size + m_ApertureSpeed * (float)(m_NewPos[1] - m_OldPos[1]));

			GetLastPos(m_OldPos[0], m_OldPos[1]);

			// Flag the camera as dirty, this will restart the rendering
			gpScene->m_DirtyFlags.SetFlag(CameraDirty);
		}
		else if (GetCtrlKey())
		{
			GetLastPos(m_NewPos[0], m_NewPos[1]);

			gpScene->m_Camera.m_FovV = max(0.0f, gpScene->m_Camera.m_FovV - m_FovSpeed * (float)(m_NewPos[1] - m_OldPos[1]));

			GetLastPos(m_OldPos[0], m_OldPos[1]);

			/// Flag the camera as dirty, this will restart the rendering
			gpScene->m_DirtyFlags.SetFlag(CameraDirty);
		}
		else
		{
			GetLastPos(m_NewPos[0], m_NewPos[1]);

			gpScene->m_Camera.Orbit(0.6f * m_OrbitSpeed * ((float)(m_NewPos[1] - m_OldPos[1]) / gpScene->m_Camera.m_Film.m_Resolution.Height()), -m_OrbitSpeed * ((float)(m_NewPos[0] - m_OldPos[0]) / gpScene->m_Camera.m_Film.m_Resolution.Width()));

			GetLastPos(m_OldPos[0], m_OldPos[1]);

			// Flag the camera as dirty, this will restart the rendering
			gpScene->m_DirtyFlags.SetFlag(CameraDirty);
		}
	}

	// Panning
	if (gMouseButtonFlags.HasFlag(Middle))
	{
		GetLastPos(m_NewPos[0], m_NewPos[1]);

		gpScene->m_Camera.Pan(m_PanSpeed * ((float)(m_NewPos[1] - m_OldPos[1]) / gpScene->m_Camera.m_Film.m_Resolution.Height()), -m_PanSpeed * ((float)(m_NewPos[0] - m_OldPos[0]) / gpScene->m_Camera.m_Film.m_Resolution.Width()));

		GetLastPos(m_OldPos[0], m_OldPos[1]);

		// Flag the camera as dirty, this will restart the rendering
		gpScene->m_DirtyFlags.SetFlag(CameraDirty);
	}

	// Zooming
	if (gMouseButtonFlags.HasFlag(Right))
	{
		GetLastPos(m_NewPos[0], m_NewPos[1]);

		gpScene->m_Camera.Zoom(-0.000001f * m_ContinuousZoomSpeed * (float)(m_NewPos[1] - m_OldPos[1]));

		GetLastPos(m_OldPos[0], m_OldPos[1]);

		// Flag the camera as dirty, this will restart the rendering
		gpScene->m_DirtyFlags.SetFlag(CameraDirty);
	}
}