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

#include "VtkCudaVolumeMapper.h"

// http://www.na-mic.org/svn/Slicer3/branches/cuda/Modules/VolumeRenderingCuda/

class CVtkWidget : public QWidget
{
    Q_OBJECT

public:
    CVtkWidget(QWidget* pParent = NULL);
	
	QVTKWidget*		GetQtVtkWidget(void);

//	QFrameBuffer	m_FrameBuffer;

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