
// Precompiled headers
#include "Stable.h"

#include "VtkWidget.h"
#include "RenderThread.h"
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
	m_InteractorStyleImage(),
	m_SceneRenderer(),
	m_RenderWindow(),
	m_RenderWindowInteractor(),
	m_KeyPressCallback(),
	m_KeyReleaseCallback(),
	m_InteractorStyleRealisticCamera(),
	m_TextActor(),
	m_RenderLoopTimer()
{
	// Create and apply main layout
	setLayout(&m_MainLayout);

	// Add VTK widget 
	m_MainLayout.addWidget(&m_QtVtkWidget);

	// Notify us when rendering begins and ends, before/after each rendered frame, when stuff becomes dirty, when the rendering canvas is resized and when the timer has timed out
	connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gStatus, SIGNAL(PreRenderFrame()), this, SLOT(OnPreRenderFrame()));
	connect(&gStatus, SIGNAL(PostRenderFrame()), this, SLOT(OnPostRenderFrame()));
	connect(&gStatus, SIGNAL(Resize()), this, SLOT(OnResize()));
//	connect(&m_RenderLoopTimer, SIGNAL(timeout()), this, SLOT(OnRenderLoopTimer()));

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
	m_SceneRenderer->GetActiveCamera()->SetParallelScale(600.0f);

	// Start the timer
	m_RenderLoopTimer.start(10.0f);
}

void CVtkWidget::OnRenderEnd(void)
{
	m_RenderWindow->Render();

	// Stop the timer
	m_RenderLoopTimer.stop();
}

void CVtkWidget::OnPreRenderFrame(void)
{
	if (!Scene())
		return;
}

void CVtkWidget::OnPostRenderFrame(void)
{
	if (!Scene())
		return;

	gpRenderThread->m_Mutex.lock();

	if (gpRenderThread->GetRenderImage())
	{
		// Decide where to blit the image
		const Vec2i CanvasSize(640, 480);//Scene()->m_Camera.m_Film.m_Resolution.GetResX(), Scene()->m_Camera.m_Film.m_Resolution.GetResY());
		const Vec2i Origin((int)(0.5f * (float)(width() - CanvasSize.x)), (int)(0.5f * (float)(height() - CanvasSize.y)));

		// Blit
		m_RenderWindow->SetRGBACharPixelData(Origin.x, Origin.y, Origin.x + CanvasSize.x - 1, Origin.y + CanvasSize.y - 1, (unsigned char*)gpRenderThread->GetRenderImage(), 1);
	}

	gpRenderThread->m_Mutex.unlock();
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

	// Setup the text and add it to the window
	vtkSmartPointer<vtkTextActor> m_TextActor = vtkSmartPointer<vtkTextActor>::New();
	m_TextActor->GetTextProperty()->SetFontSize(24);
	m_TextActor->SetPosition2(10, 40);
//	m_SceneRenderer->AddActor2D(m_TextActor);
	m_TextActor->SetInput ( "Hello world" );
	m_TextActor->GetTextProperty()->SetColor(1.0,0.0,0.0);
}

void CVtkWidget::OnRenderLoopTimer(void)
{
	
}

void CVtkWidget::OnResize(void)
{
	if (!Scene())
		return;
	/*
 	m_ImageImport->SetDataExtent(0, Scene()->m_Camera.m_Film.m_Resolution.GetResX() - 1, 0, Scene()->m_Camera.m_Film.m_Resolution.GetResY() - 1, 0, 0);
 	m_ImageImport->SetWholeExtent(0, Scene()->m_Camera.m_Film.m_Resolution.GetResX() - 1, 0, Scene()->m_Camera.m_Film.m_Resolution.GetResY() - 1, 0, 0);
// 	m_ImageActor->SetDisplayExtent(0, Scene()->m_Camera.m_Film.m_Resolution.GetWidth() - 1, 0, Scene()->m_Camera.m_Film.m_Resolution.GetHeight() - 1, 0, 0);

	m_pImageImport->SetImportVoidPointer(NULL);
	m_pImageImport->SetImportVoidPointer(gpRenderThread->GetRenderImage());
	*/
}