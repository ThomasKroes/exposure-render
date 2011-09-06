#pragma once

// VTK
#include <vtkInteractorStyleUser.h>
#include <vtkObjectFactory.h>

#include "Flags.h"
#include "Camera.h"

// Define interaction style for controlling a realistic camera
class vtkRealisticCameraStyle : public vtkInteractorStyleUser
{
public:
	enum EMouseButtonFlag
	{
		Left	= 0x0001,
		Middle	= 0x0002,
		Right	= 0x0004
	};

	static vtkRealisticCameraStyle* New();
	vtkTypeMacro(vtkRealisticCameraStyle, vtkInteractorStyleUser);

	virtual void OnLeftButtonDown(void);
	virtual void OnLeftButtonUp(void);
	virtual void OnRightButtonDown(void);
	virtual void OnRightButtonUp(void);
	virtual void OnMiddleButtonDown(void);
	virtual void OnMiddleButtonUp(void);
	virtual void OnMouseWheelForward(void);
	virtual void OnMouseWheelBackward(void);
	virtual void OnMouseMove(void);

	int m_OldPos[2];
	int m_NewPos[2];

	// Camera sensitivity to mouse movement
	static float m_OrbitSpeed;			
	static float m_PanSpeed;
	static float m_ZoomSpeed;
	static float m_ContinuousZoomSpeed;
	static float m_ApertureSpeed;
	static float m_FovSpeed;
};