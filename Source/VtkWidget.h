#pragma once

#include <QVTKWidget.h>

// VTK
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkConeSource.h>
#include <vtkRenderWindow.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkImageImport.h>
#include <vtkImageActor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkVolumeMapper.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkImageData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPointData.h>

// Interactor
#include "InteractorStyleRealisticCamera.h"

#include "RenderThread.h"

#include "VolumeMapper.h"

// http://www.na-mic.org/svn/Slicer3/branches/cuda/Modules/VolumeRenderingCuda/

class CVtkWidget : public QWidget
{
    Q_OBJECT

public:
    CVtkWidget(QWidget* pParent = NULL);
	
	QVTKWidget*		GetQtVtkWidget(void);

	QFrameBuffer	m_FrameBuffer;

public slots:
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnRenderLoopTimer(void);

private:
	void SetupRenderView(void);
	
	QGridLayout									m_MainLayout;
	QVTKWidget									m_QtVtkWidget;
	unsigned char*								m_pPixels;
	QTimer										m_RenderLoopTimer;

public:
	vtkSmartPointer<vtkImageActor>				m_ImageActor;
	vtkSmartPointer<vtkImageImport>				m_ImageImport;
	vtkSmartPointer<vtkInteractorStyleImage>	m_InteractorStyleImage;
	vtkSmartPointer<vtkRenderer>				m_SceneRenderer;
	vtkSmartPointer<vtkRenderWindow>			m_RenderWindow;
	vtkSmartPointer<vtkRenderWindowInteractor>	m_RenderWindowInteractor;
	vtkSmartPointer<vtkCallbackCommand>			m_KeyPressCallback;
	vtkSmartPointer<vtkCallbackCommand>			m_KeyReleaseCallback;
	vtkSmartPointer<vtkRealisticCameraStyle>	m_InteractorStyleRealisticCamera;

	// Experimental
	vtkSmartPointer<vtkVolume>					m_Volume;
//	vtkSmartPointer<vtkVolumeCudaMapper>		m_VolumeMapper;
};