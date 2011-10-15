
// Precompiled headers
#include "Stable.h"

#include "VtkWidget.h"
#include "MainWindow.h"

#include <vtkMetaImageReader.h>
#include <vtkVolumeProperty.h>

void KeyPressCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
 	vtkRenderWindowInteractor* pRenderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(pCaller);
 
 	char* pKeySymbol = pRenderWindowInteractor->GetKeySym();
 
 	if (strcmp(pKeySymbol, "space") == 0)
 	{
		pRenderWindowInteractor->SetInteractorStyle(gpMainWindow->m_VtkWidget.m_InteractorStyleImage);
 		gpMainWindow->setCursor(QCursor(Qt::PointingHandCursor));
 	}
}

void KeyReleaseCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
 	vtkRenderWindowInteractor* pRenderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(pCaller);
 
 	char* pKeySymbol = pRenderWindowInteractor->GetKeySym();
 
 	if (strcmp(pKeySymbol, "space") == 0)
 	{
 		pRenderWindowInteractor->SetInteractorStyle(gpMainWindow->m_VtkWidget.m_InteractorStyleRealisticCamera);
 		gpMainWindow->setCursor(QCursor(Qt::ArrowCursor));
 	}
}

CVtkWidget::CVtkWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_QtVtkWidget(),
	m_pPixels(NULL),
	m_ImageActor(),
	m_ImageImport(),
	m_InteractorStyleImage(),
	m_SceneRenderer(),
	m_RenderWindow(),
	m_RenderWindowInteractor(),
	m_KeyPressCallback(),
	m_KeyReleaseCallback(),
	m_InteractorStyleRealisticCamera()
{
	setLayout(&m_MainLayout);

	QMenu* pMenu = new QMenu();
	pMenu->addAction("asd", this, SLOT(OnRenderBegin()));

	m_MainLayout.addWidget(pMenu);
	
	m_MainLayout.addWidget(&m_QtVtkWidget, 0, 0, 1, 2);

	QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	QObject::connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	QObject::connect(&m_RenderLoopTimer, SIGNAL(timeout()), this, SLOT(OnRenderLoopTimer()));

	SetupRenderView();
}

QVTKWidget* CVtkWidget::GetQtVtkWidget(void)
{
	return &m_QtVtkWidget;
}

void CVtkWidget::OnRenderBegin(void)
{
	
	// Scale
	m_SceneRenderer->GetActiveCamera()->SetParallelScale(500.0f);

	m_ImageImport->SetDataSpacing(1, 1, 1);
	m_ImageImport->SetDataOrigin(-0.5f * (float)gScene.m_Camera.m_Film.m_Resolution.GetResX(), -0.5f * (float)gScene.m_Camera.m_Film.m_Resolution.GetResY(), 0);

	m_pPixels = (unsigned char*)malloc(4 * 2048 * 2048 * sizeof(unsigned char));

	m_ImageImport->SetImportVoidPointer((void*)m_pPixels, 1);
	m_ImageImport->SetWholeExtent(0, gScene.m_Camera.m_Film.m_Resolution.GetResX() - 1, 0, gScene.m_Camera.m_Film.m_Resolution.GetResY() - 1, 0, 0);
	m_ImageImport->SetDataExtentToWholeExtent();
	m_ImageImport->SetDataScalarTypeToUnsignedChar();
	m_ImageImport->SetNumberOfScalarComponents(3);
	m_ImageImport->Update();

	m_ImageActor->SetInterpolate(1);
	m_ImageActor->SetInput(m_ImageImport->GetOutput());
	m_ImageActor->SetScale(1, -1, -1);
	m_ImageActor->VisibilityOn();

	// Add the image actor
	m_SceneRenderer->AddActor(m_ImageActor); 
	
	// Start the timer
	m_RenderLoopTimer.start(1000.0f / 25.0f);
	/*

	vtkSmartPointer<vtkMetaImageReader> MetaImageReader = vtkMetaImageReader::New();

	MetaImageReader->SetFileName("c:\\volumes\\manix_small.mhd");
	MetaImageReader->Update();

	m_VolumeMapper->SetInput(MetaImageReader->GetOutput());

    m_Volume->SetMapper(m_VolumeMapper);
    vtkVolumeProperty* prop = vtkVolumeProperty::New();
    m_Volume->SetProperty(prop);
    m_SceneRenderer->AddViewProp(m_Volume);*/
}

//http://agl.unm.edu/rpf/

void CVtkWidget::OnRenderEnd(void)
{
	m_ImageActor->VisibilityOff();
	m_RenderWindow->Render();
}

void CVtkWidget::SetupRenderView(void)
{
	// Create and configure scene renderer
	m_SceneRenderer = vtkRenderer::New();
	m_SceneRenderer->SetBackground(0.25, 0.25, 0.25);
	m_SceneRenderer->SetBackground2(0.25, 0.25, 0.25);
	m_SceneRenderer->SetGradientBackground(true);
	m_SceneRenderer->GetActiveCamera()->SetPosition(0.0, 0.0, 1.0);
	m_SceneRenderer->GetActiveCamera()->SetFocalPoint(0.0, 0.0, 0.0);
	m_SceneRenderer->GetActiveCamera()->ParallelProjectionOn();

	// Get render window and configure
	m_RenderWindow = GetQtVtkWidget()->GetRenderWindow();
	m_RenderWindow->AddRenderer(m_SceneRenderer);

	// Key press callback
	m_KeyPressCallback = vtkCallbackCommand::New();
	m_KeyPressCallback->SetCallback(KeyPressCallbackFunction);

	// Key press callback
	m_KeyReleaseCallback = vtkCallbackCommand::New();
	m_KeyReleaseCallback->SetCallback(KeyReleaseCallbackFunction);

	// Create interactor style for camera navigation
	m_InteractorStyleRealisticCamera = vtkSmartPointer<vtkRealisticCameraStyle>::New();
	m_InteractorStyleImage = vtkInteractorStyleImage::New();

	// Add observers
	m_RenderWindow->GetInteractor()->SetInteractorStyle(m_InteractorStyleRealisticCamera);
//	m_RenderWindow->GetInteractor()->SetInteractorStyle(m_InteractorStyleImage);

	m_RenderWindow->GetInteractor()->AddObserver(vtkCommand::KeyPressEvent, m_KeyPressCallback);
	m_RenderWindow->GetInteractor()->AddObserver(vtkCommand::KeyReleaseEvent, m_KeyReleaseCallback);

	m_ImageImport = vtkImageImport::New();
	m_ImageActor = vtkImageActor::New();
}

void CVtkWidget::OnRenderLoopTimer(void)
{
	if (!gpRenderThread)
		return;

	QMutexLocker(&gpRenderThread->m_Mutex);

	m_FrameBuffer = gFrameBuffer;

	if (m_FrameBuffer.GetPixels() == NULL)
		return;

	m_ImageImport->SetImportVoidPointer(NULL);
	m_ImageImport->SetImportVoidPointer(m_FrameBuffer.GetPixels());

	m_ImageImport->SetDataOrigin(-0.5f * (float)m_FrameBuffer.GetWidth(), -0.5f * (float)m_FrameBuffer.GetHeight(), 0);
	m_ImageImport->SetWholeExtent(0, m_FrameBuffer.GetWidth() - 1, 0, m_FrameBuffer.GetHeight() - 1, 0, 0);
	m_ImageImport->UpdateWholeExtent();
	m_ImageImport->SetDataExtentToWholeExtent();
	m_ImageImport->Update();
	
	m_ImageActor->SetInput(m_ImageImport->GetOutput());

	m_RenderWindow->GetInteractor()->Render();
}