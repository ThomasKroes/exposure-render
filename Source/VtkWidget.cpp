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

#include "VtkWidget.h"


#include <vtkMetaImageReader.h>
#include <vtkCubeSource.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkLight.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLightActor.h>
#include <vtkPlaneWidget.h>
#include <vtkPointWidget.h>
//#include <vtkAxesTransformWidget.h>
//#include <vtkAxesTransformRepresentation.h>
#include <vtkLineWidget.h>
#include <vtkSphereSource.h>
#include <vtkAxesActor.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkImageActor.h>

#include "vtkErBackgroundLight.h"
#include "VtkErAreaLight.h"
#include "vtkErVolumeMapper.h"

//http://agl.unm.edu/rpf/

void TimerCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
 	vtkRenderWindowInteractor* pRenderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(pCaller);
 
	if (pRenderWindowInteractor)
		pRenderWindowInteractor->Render();

	
}

CVtkRenderWidget* gpActiveRenderWidget = NULL;

CVtkRenderWidget::CVtkRenderWidget(QWidget* pParent) :
	QDialog(pParent),
	m_MainLayout(),
	m_ViewFront(),
	m_QtVtkWidget(),
	m_Volume(),
	m_VolumeProperty(),
	m_VolumeMapper(),
	m_Renderer(),
	m_OverlayRenderer(),
	m_TimerCallback()
{
	setLayout(&m_MainLayout);
	
	m_MainLayout.addWidget(&m_QtVtkWidget);

	m_MainLayout.setContentsMargins(1, 1, 1, 1);
 	m_ViewFront.setText("Front");
	m_ViewBack.setText("Back");
	m_ViewLeft.setText("Left");
	m_ViewRight.setText("Right");
	m_ViewTop.setText("Top");
	m_ViewBottom.setText("Bottom");

	m_MainLayout.addWidget(&m_ViewFront);
	m_MainLayout.addWidget(&m_ViewBack);
	m_MainLayout.addWidget(&m_ViewLeft);
	m_MainLayout.addWidget(&m_ViewRight);
	m_MainLayout.addWidget(&m_ViewTop);
	m_MainLayout.addWidget(&m_ViewBottom);

	QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	QObject::connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	QObject::connect(&m_ViewFront, SIGNAL(clicked()), this, SLOT(OnViewFront()));
	QObject::connect(&m_ViewBack, SIGNAL(clicked()), this, SLOT(OnViewBack()));
	QObject::connect(&m_ViewLeft, SIGNAL(clicked()), this, SLOT(OnViewLeft()));
	QObject::connect(&m_ViewRight, SIGNAL(clicked()), this, SLOT(OnViewRight()));
	QObject::connect(&m_ViewTop, SIGNAL(clicked()), this, SLOT(OnViewTop()));
	QObject::connect(&m_ViewBottom, SIGNAL(clicked()), this, SLOT(OnViewBottom()));

	SetupVtk();
}

QVTKWidget* CVtkRenderWidget::GetQtVtkWidget(void)
{
	return &m_QtVtkWidget;
}

void CVtkRenderWidget::OnRenderBegin(void)
{
}

void CVtkRenderWidget::OnRenderEnd(void)
{
}

void CVtkRenderWidget::SetActive(void)
{
	gpActiveRenderWidget = this;
}

void CVtkRenderWidget::LoadVolume(const QString& FilePath)
{
	vtkSmartPointer<vtkMetaImageReader> MetaImageReader = vtkMetaImageReader::New();

	MetaImageReader->SetFileName(FilePath.toAscii());

	MetaImageReader->Update();

	m_VolumeMapper->SetInput(MetaImageReader->GetOutput());
	
    m_Volume->SetMapper(m_VolumeMapper);

    m_Volume->SetProperty(m_VolumeProperty);

	// Camera
	m_Camera->SetRenderer(m_Renderer);
	m_Renderer->SetActiveCamera(m_Camera);
	m_Renderer->ResetCamera();

	m_Renderer->AddViewProp(m_Volume);

	vtkErAreaLight* l1 = vtkErAreaLight::New();
	l1->SetPosition(-4.0,4.0,-1.0);
	l1->SetFocalPoint(0,0,0);
	l1->SetColor(100.0,100.0,100.0);
	l1->SetPositional(1);
	m_Renderer->AddLight(l1);
	l1->SetSwitch(1);

	vtkLightActor *la = vtkLightActor::New();
    la->SetLight(l1);
	
	// Environment light
	vtkErBackgroundLight* pErBackgroundLight = vtkErBackgroundLight::New();
	pErBackgroundLight->SetDiffuseColor(500, 500, 10000);
	m_Renderer->AddLight(pErBackgroundLight);

	vtkPlaneWidget* planeWidget = vtkPlaneWidget::New();
	planeWidget->SetInteractor(m_QtVtkWidget.GetRenderWindow()->GetInteractor());
	planeWidget->On();

	m_QtVtkWidget.GetRenderWindow()->GetInteractor()->CreateRepeatingTimer(0.1);
	m_QtVtkWidget.GetRenderWindow()->GetInteractor()->AddObserver(vtkCommand::TimerEvent, m_TimerCallback);

	m_Renderer->GetActiveCamera()->SetPosition(10, 10, 10);
	m_Renderer->GetActiveCamera()->SetFocalPoint(0, 0, 0);
	m_Renderer->GetActiveCamera()->SetFocalDisk(0.0);
	m_Renderer->GetActiveCamera()->SetClippingRange(0, 100000000);
	
	vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
	m_QtVtkWidget.GetRenderWindow()->GetInteractor()->SetInteractorStyle(style);
	
	
//	vtkSmartPointer<vtkInteractorStyleImage> style1 = vtkSmartPointer<vtkInteractorStyleImage>::New();
//	m_QtVtkWidget.GetRenderWindow()->GetInteractor()->SetInteractorStyle(style1);
}

void CVtkRenderWidget::SetupVtk(void)
{
	m_Volume			= vtkVolume::New();
	m_VolumeProperty	= vtkErVolumeProperty::New();
	m_VolumeMapper		= vtkErVolumeMapper::New();
	m_Renderer			= vtkRenderer::New();
	m_OverlayRenderer	= vtkRenderer::New();
	m_TimerCallback		= vtkCallbackCommand::New();
	m_Camera			= vtkErCamera::New();

//	m_Renderer->SetInteractive(1);
//	m_OverlayRenderer->SetInteractive(1);

//	m_Renderer->SetLayer(1);
//	m_OverlayRenderer->SetLayer(0);

//	m_QtVtkWidget.GetRenderWindow()->SetNumberOfLayers(2);
//	m_QtVtkWidget.GetRenderWindow()->AddRenderer(m_OverlayRenderer);
	m_QtVtkWidget.GetRenderWindow()->AddRenderer(m_Renderer);
	
//	m_QtVtkWidget.setGraphicsEffect(new QGraphicsDropShadowEffect());
	
	m_Renderer->SetBackground(0.1, 0.1, 0.1);
	
//	m_QtVtkWidget.GetRenderWindow()->SetPolygonSmoothing(1);

	m_TimerCallback->SetCallback(TimerCallbackFunction);
	m_TimerCallback->SetClientData((void*)this);

//	m_QtVtkWidget.GetRenderWindow()->SetAlphaBitPlanes(true);
//	m_QtVtkWidget.GetRenderWindow()->SetMultiSamples(0);

//	m_Renderer->SetBackground(0, 0, 0);
//	m_Renderer->GetActiveCamera()->SetPosition(0.0, 0.0, 10.0);
//	m_Renderer->GetActiveCamera()->SetFocalPoint(0.0, 0.0, 0.0);
//	m_Renderer->SetUseDepthPeeling(true);
//	m_Renderer->SetMaximumNumberOfPeels(4);
//	m_QtVtkWidget.setAutomaticImageCacheEnabled(false);
}

void CVtkRenderWidget::OnViewFront(void)
{
	m_Camera->SetViewFront();
}

void CVtkRenderWidget::OnViewBack(void)
{
	m_Camera->SetViewBack();
}

void CVtkRenderWidget::OnViewLeft(void)
{
	m_Camera->SetViewLeft();
}

void CVtkRenderWidget::OnViewRight(void)
{
	m_Camera->SetViewRight();
}

void CVtkRenderWidget::OnViewTop(void)
{
	m_Camera->SetViewTop();
}

void CVtkRenderWidget::OnViewBottom(void)
{
	m_Camera->SetViewBottom();
}

QRenderView::QRenderView(QWidget* pParent /*= NULL*/) :
	QGraphicsView(pParent),
	m_GraphicsScene(),
	m_RenderWidget()
{
	setScene(&m_GraphicsScene);
	setWindowTitle("Render Canvas");

	m_GraphicsScene.setBackgroundBrush(QBrush(Qt::black));

	m_GraphicsScene.addWidget(&m_RenderWidget, Qt::Dialog);

//	m_RenderWidget.setParent(this);
//	m_RenderWidget.setFocus();

//	m_GraphicsScene.addItem(new QVTKGraphicsItem(new QGLContext(QGLFormat())));
//	setViewport(&m_RenderWidget);
//    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
//	setInteractive(true);
//	renderHints() = 0;
}