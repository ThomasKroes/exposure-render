
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
	m_ImageActor(),
	m_DisplayImage(),
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

	m_DisplayImage = vtkImageData::New();
 	m_DisplayImage->SetDimensions(640, 480, 1);
//	m_DisplayImage->SetDimensions(512,512,10);
	m_DisplayImage->SetSpacing(1,1,1);
	m_DisplayImage->SetScalarTypeToUnsignedChar();
	m_DisplayImage->SetNumberOfScalarComponents(3);// the r,g,b value.
	m_DisplayImage->SetOrigin(0.0,0.0,0.0);
	m_DisplayImage->AllocateScalars();
	

	m_ImageActor = vtkImageActor::New();

	// Configure image actor
	m_ImageActor->SetInterpolate(1);
	m_ImageActor->SetInput(m_DisplayImage);
	m_ImageActor->SetScale(1, -1, -1);
	m_ImageActor->VisibilityOn();

	// Add the image actor
	m_SceneRenderer->AddActor(m_ImageActor); 

	
//	m_UcharArray->Allocate(Scene()->m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr));
	// Start the timer
//	m_RenderLoopTimer.start(10.0f);
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
	
//	gpRenderThread->m_Mutex.lock();

	if (gpRenderThread->GetRenderImage())
	{/*
		const Vec2i CanvasSize(Scene()->m_Camera.m_Film.m_Resolution.GetResX(), Scene()->m_Camera.m_Film.m_Resolution.GetResY());
		const Vec2i Origin((int)(0.5f * (float)(width() - CanvasSize.x)), (int)(0.5f * (float)(height() - CanvasSize.y)));

		
		vtkSmartPointer<vtkUnsignedCharArray> m_UcharArray = vtkUnsignedCharArray::New();
		
//		m_UcharArray->Allocate(Scene()->m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr));
//		m_UcharArray->SetArray((unsigned char*)gpRenderThread->GetRenderImage(), Scene()->m_Camera.m_Film.m_Resolution.GetNoElements(), 0, 1);
//		m_UcharArray->SetVoidArray(NULL, Scene()->m_Camera.m_Film.m_Resolution.GetNoElements(), 0);
//		m_UcharArray->SetVoidArray((void*)gpRenderThread->GetRenderImage(), Scene()->m_Camera.m_Film.m_Resolution.GetNoElements(), 0);
		
		vtkUnsignedCharArray *_colors = (vtkUnsignedCharArray*)m_DisplayImage->GetPointData()->GetScalars();
		
		for (int i = 0; i < Scene()->m_Camera.m_Film.m_Resolution.GetNoElements(); i++)
		{
// 			_colors->SetValue(i, gpRenderThread->GetRenderImage()[i].r);
		}
//		m_DisplayImage->Update();
		m_DisplayImage->Modified();
//		m_DisplayImage->SetDimensions(640, 480, 1);
//		m_ImageActor->SetInput(m_DisplayImage);
//		m_ImageActor->Render(m_SceneRenderer);
		m_RenderWindow->Render();

		*/

		// Decide where to blit the image
		const Vec2i CanvasSize(Scene()->m_Camera.m_Film.m_Resolution.GetResX(), Scene()->m_Camera.m_Film.m_Resolution.GetResY());
		const Vec2i Origin((int)(0.5f * (float)(width() - CanvasSize.x)), (int)(0.5f * (float)(height() - CanvasSize.y)));

		// Blit
		m_RenderWindow->SetPixelData(max(0, Origin.x), max(0, Origin.y), Origin.x + CanvasSize.x - 1, Origin.y + CanvasSize.y - 1, (unsigned char*)gpRenderThread->GetRenderImage(), 1);

	}

// 	gpRenderThread->m_Mutex.unlock();
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