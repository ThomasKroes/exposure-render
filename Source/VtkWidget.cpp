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
#include "MainWindow.h"

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

#include "vtkErBackgroundLight.h"
#include "vtkErAreaLightWidget.h"

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
	m_MainLayout.addWidget(new QPushButton("Hello World!"));

	QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	QObject::connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));

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

//	m_Renderer->RemoveAllLights();

//	vtkErBackgroundLight* pErBackgroundLight = vtkErBackgroundLight::New();

//	pErBackgroundLight->SetDiffuseColor(10000, 10000, 10000);

//	m_SceneRenderer->AddLight(pErBackgroundLight);

	vtkSphereSource* pBox = vtkSphereSource::New();

  pBox->SetRadius(0.1);
  
  pBox->Update();

  // The mapper is responsible for pushing the geometry into the graphics
  // library. It may also do color mapping, if scalars or other attributes
  // are defined.
  vtkPolyDataMapper *cylinderMapper = vtkPolyDataMapper::New();
  cylinderMapper->SetInputConnection(pBox->GetOutputPort());

  // The actor is a grouping mechanism: besides the geometry (mapper), it
  // also has a property, transformation matrix, and/or texture map.
  // Here we set its color and rotate it -22.5 degrees.
  vtkActor *cylinderActor = vtkActor::New();
  cylinderActor->SetMapper(cylinderMapper);
//  cylinderActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
//  cylinderActor->GetProperty()->SetOpacity(1.0);
//  cylinderActor->RotateX(30.0);
//  cylinderActor->RotateY(-45.0);
	
  m_Renderer->AddViewProp(m_Volume);
//	m_Renderer->AddActor(cylinderActor);
	

	vtkLight* pLight = vtkLight::New();
	pLight->SetPosition(2.0, 2.0, 2.0);
	pLight->SetDiffuseColor(1, 1, 1);

//	m_Renderer->RemoveAllLights();
//	m_Renderer->AddLight(pLight);
//	m_SceneRenderer->AddLight(pLight2);
//	m_SceneRenderer->AddLight(pLight3);

	// lighting the box.
  vtkErAreaLight* l1 = vtkErAreaLight::New();
  l1->SetPosition(-4.0,4.0,-1.0);
  l1->SetFocalPoint(0,0,0);
  l1->SetColor(10000.0,100.0,100.0);
  l1->SetPositional(1);
  m_Renderer->AddLight(l1);
  l1->SetSwitch(1);

	vtkLightActor *la = vtkLightActor::New();
    la->SetLight(l1);
//    m_Renderer->AddViewProp(la);

	
	m_Renderer->SetActiveCamera(m_Camera);
	m_Renderer->ResetCamera();

	vtkErAreaLightWidget* vtkLineWidget = vtkErAreaLightWidget::New();
  vtkLineWidget->SetInteractor(m_QtVtkWidget.GetRenderWindow()->GetInteractor());
  

//  vtkLineWidget->On();
//  vtkLineWidget->SetCurrentRenderer(m_OverlayRenderer);

//  vtkAxesTransformRepresentation* pRep = vtkAxesTransformRepresentation::New();

	vtkErBackgroundLight* pErBackgroundLight = vtkErBackgroundLight::New();

	pErBackgroundLight->SetDiffuseColor(5000, 5000, 10000);

	m_Renderer->AddLight(pErBackgroundLight);

	m_QtVtkWidget.GetRenderWindow()->GetInteractor()->CreateRepeatingTimer(0.001);
	m_QtVtkWidget.GetRenderWindow()->GetInteractor()->AddObserver(vtkCommand::TimerEvent, m_TimerCallback);

	m_Renderer->GetActiveCamera()->SetPosition(10, 10, 10);
	m_Renderer->GetActiveCamera()->SetFocalPoint(0, 0, 0);
	m_Renderer->GetActiveCamera()->SetFocalDisk(0.0);
	m_Renderer->GetActiveCamera()->SetClippingRange(1, 10);

	vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
	m_QtVtkWidget.GetRenderWindow()->GetInteractor()->SetInteractorStyle(style);
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
	
	
	
//	m_QtVtkWidget.GetRenderWindow()->SetPolygonSmoothing(1);

	m_TimerCallback->SetCallback(TimerCallbackFunction);
	m_TimerCallback->SetClientData((void*)this);

	m_QtVtkWidget.GetRenderWindow()->SetAlphaBitPlanes(true);
//	m_QtVtkWidget.GetRenderWindow()->SetMultiSamples(0);

//	m_Renderer->SetBackground(0, 0, 0);
//	m_Renderer->GetActiveCamera()->SetPosition(0.0, 0.0, 10.0);
//	m_Renderer->GetActiveCamera()->SetFocalPoint(0.0, 0.0, 0.0);
//	m_Renderer->SetUseDepthPeeling(true);
//	m_Renderer->SetMaximumNumberOfPeels(4);
//	m_QtVtkWidget.setAutomaticImageCacheEnabled(false);
}


QRenderView::QRenderView(QWidget* pParent /*= NULL*/) :
	QGraphicsView(pParent),
	m_GraphicsScene(),
	m_RenderWidget()
{
	setScene(&m_GraphicsScene);

	m_GraphicsScene.setBackgroundBrush(QBrush(Qt::black));

	m_GraphicsScene.addWidget(&m_RenderWidget, Qt::Dialog);

	m_RenderWidget.setFocus();
}