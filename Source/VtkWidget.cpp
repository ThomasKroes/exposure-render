
// Precompiled headers
#include "Stable.h"

#include "VtkWidget.h"
#include "RenderThread.h"

// Key press callback
void KeyPressCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
	/*
	vtkRenderWindowInteractor* pRenderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(pCaller);

	char* pKeySymbol = pRenderWindowInteractor->GetKeySym();

	if (strcmp(pKeySymbol, "space") == 0)
	{
		pRenderWindowInteractor->SetInteractorStyle(gpMainWindow->m_pInteractorStyleImage);

		// Change the cursor to a pointing, thus indicating the change in interaction mode
		gpMainWindow->setCursor(QCursor(Qt::CursorShape::PointingHandCursor));
	}
	*/
}

// Key press callback
void KeyReleaseCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
	/* 
	vtkRenderWindowInteractor* pRenderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(pCaller);

	char* pKeySymbol = pRenderWindowInteractor->GetKeySym();

	if (strcmp(pKeySymbol, "space") == 0)
	{
		pRenderWindowInteractor->SetInteractorStyle(gpMainWindow->m_pInteractorStyleRealisticCamera);

		// Change the cursor to a pointing, thus indicating the change in interaction mode
		gpMainWindow->setCursor(QCursor(Qt::CursorShape::ArrowCursor));
	}
	*/
}

CVtkWidget::CVtkWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_QtVtkWidget(),
	m_ImageImport(),
	m_ImageActor(),
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

	// Create the image importer and image actor, they provide the canvas for 
	m_ImageImport = vtkImageImport::New();
	m_ImageActor = vtkImageActor::New();

	// Notify us when rendering begins and ends, before/after each rendered frame, when stuff becomes dirty, when the rendering canvas is resized and when the timer has timed out
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gRenderStatus, SIGNAL(PreRenderFrame()), this, SLOT(OnPreRenderFrame()));
	connect(&gRenderStatus, SIGNAL(PostRenderFrame()), this, SLOT(OnPostRenderFrame()));
	connect(&gRenderStatus, SIGNAL(Resize()), this, SLOT(OnResize()));
	connect(&m_RenderLoopTimer, SIGNAL(timeout()), this, SLOT(OnRenderLoopTimer()));

	// Setup the render view
	SetupRenderView();
}

QVTKWidget* CVtkWidget::GetQtVtkWidget(void)
{
	return &m_QtVtkWidget;
}

void CVtkWidget::OnRenderBegin(void)
{
	
	m_ImageImport->SetDataSpacing(1, 1, 1);
	m_ImageImport->SetDataOrigin(-400, -300, 0);
	m_ImageImport->SetImportVoidPointer((void*)malloc(3 * 800 * 600 * sizeof(unsigned char)));
	m_ImageImport->SetWholeExtent(0, 800 - 1, 0, 600 - 1, 0, 0);
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

	// Scale
	m_SceneRenderer->GetActiveCamera()->SetParallelScale(600.0f);
	/**/

	// Start the timer
	m_RenderLoopTimer.start(10);
}

void CVtkWidget::OnRenderEnd(void)
{
	// Remove the image actor from the screen
	m_SceneRenderer->RemoveActor(m_ImageActor);

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
	//	m_pRenderWindow->GetInteractor()->SetInteractorStyle(m_pInteractorStyleImage);

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
	if (!Scene())
		return;

	if (gpRenderThread->GetRenderImage())
	{
		m_ImageImport->SetDataExtent(0, Scene()->m_Camera.m_Film.m_Resolution.GetResX() - 1, 0, Scene()->m_Camera.m_Film.m_Resolution.GetResY() - 1, 0, 0);
		m_ImageImport->SetWholeExtent(0, Scene()->m_Camera.m_Film.m_Resolution.GetResX() - 1, 0, Scene()->m_Camera.m_Film.m_Resolution.GetResY() - 1, 0, 0);
		m_ImageImport->SetImportVoidPointer(NULL);
		m_ImageImport->SetImportVoidPointer(gpRenderThread->GetRenderImage());

		m_ImageActor->SetInput(m_ImageImport->GetOutput());

		m_RenderWindow->GetInteractor()->Render();
	}
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