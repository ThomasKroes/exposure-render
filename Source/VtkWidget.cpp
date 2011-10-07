
// Precompiled headers
#include "Stable.h"

#include "VtkWidget.h"
#include "MainWindow.h"

// Key press callback
void KeyPressCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
// 	vtkRenderWindowInteractor* pRenderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(pCaller);
// 
// 	char* pKeySymbol = pRenderWindowInteractor->GetKeySym();
// 
// 	if (strcmp(pKeySymbol, "space") == 0)
// 	{
// 		pRenderWindowInteractor->SetInteractorStyle(gpMainWindow->m_VtkWidget.m_InteractorStyleRealisticCamera);
// 
// 		// Change the cursor to a pointing, thus indicating the change in interaction mode
// 		gpMainWindow->setCursor(QCursor(Qt::CursorShape::PointingHandCursor));
// 	}
}

// Key press callback
void KeyReleaseCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
// 	vtkRenderWindowInteractor* pRenderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(pCaller);
// 
// 	char* pKeySymbol = pRenderWindowInteractor->GetKeySym();
// 
// 	if (strcmp(pKeySymbol, "space") == 0)
// 	{
// 		pRenderWindowInteractor->SetInteractorStyle(gpMainWindow->m_VtkWidget.m_InteractorStyleRealisticCamera);
// 
// 		// Change the cursor to a pointing, thus indicating the change in interaction mode
// 		gpMainWindow->setCursor(QCursor(Qt::CursorShape::ArrowCursor));
// 	}
}

CVtkWidget::CVtkWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_QtVtkWidget(),
	m_pPixels(NULL),
	m_Pause(false),
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
	// Create and apply main layout
	setLayout(&m_MainLayout);

	QMenu* pMenu = new QMenu();
	pMenu->addAction("asd", this, SLOT(OnRenderBegin()));

	m_MainLayout.addWidget(pMenu);
	
	// Add VTK widget 
	m_MainLayout.addWidget(&m_QtVtkWidget, 0, 0, 1, 2);

	// Notify us when rendering begins and ends, before/after each rendered frame, when stuff becomes dirty, when the rendering canvas is resized and when the timer has timed out
	QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	QObject::connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	QObject::connect(&m_RenderLoopTimer, SIGNAL(timeout()), this, SLOT(OnRenderLoopTimer()));
	QObject::connect(&gCamera.GetFilm(), SIGNAL(Changed(const QFilm&)), this, SLOT(OnFilmChanged(const QFilm&)));
	QObject::connect(&gStatus, SIGNAL(RenderPause(const bool&)), this, SLOT(OnRenderPause(const bool&)));

	// Setup the render view
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

	m_pPixels = (unsigned char*)malloc(3 * 2048 * 2048 * sizeof(unsigned char));

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
	
	// Add the image actor
	m_SceneRenderer->AddActor(m_ImageActor); 

	
//	m_UcharArray->Allocate(gScene.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr));
	// Start the timer
	m_RenderLoopTimer.start(1000.0f / 40.0f);
}

void CVtkWidget::OnRenderEnd(void)
{
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

void CVtkWidget::OnFilmChanged(const QFilm& Film)
{
// 	free(m_pPixels);
// 	m_pPixels = (unsigned char*)malloc(3 * Film.GetWidth() * Film.GetHeight() * sizeof(unsigned char));
// 
// 	m_ImageImport->SetDataOrigin(-0.5f * (float)Film.GetWidth(), -0.5f * (float)Film.GetHeight(), 0);
// 	m_ImageImport->SetImportVoidPointer((void*)m_pPixels, 1);
// 	m_ImageImport->SetDataExtent(0, Film.GetWidth() - 1, 0, Film.GetHeight() - 1, 0, 0);
// 	m_ImageImport->SetWholeExtent(0, Film.GetWidth() - 1, 0, Film.GetHeight() - 1, 0, 0);
// 	m_ImageImport->SetDataExtentToWholeExtent();

// 	m_ImageImport->SetImportVoidPointer(NULL);
// 	m_ImageImport->SetImportVoidPointer(gpRenderThread->GetRenderImage());
// 	// 
// 	m_ImageActor->SetInput(m_ImageImport->GetOutput());

	// 	m_ImageActor->SetDisplayExtent(0, gScene.m_Camera.m_Film.m_Resolution.GetWidth() - 1, 0, gScene.m_Camera.m_Film.m_Resolution.GetHeight() - 1, 0, 0);

//	m_ImageImport->SetImportVoidPointer(NULL);
	// 	m_pImageImport->SetImportVoidPointer(gpRenderThread->GetRenderImage());
	/**/
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
	m_ImageImport->SetDataExtentToWholeExtent();
	m_ImageImport->UpdateWholeExtent();
//	m_ImageImport->Update();
	
	m_ImageActor->SetInput(m_ImageImport->GetOutput());
	m_RenderWindow->GetInteractor()->Render();
}

void CVtkWidget::OnRenderPause(const bool& Pause)
{
	m_Pause = Pause;
}