
#include <vtkMetaImageReader.h>
#include <vtkSmartPointer.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderer.h>
#include <vtkImageActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkTesting.h>
#include <vtkImageCast.h>
#include <vtkPolyDataMapper.h>
#include <vtkPlanes.h>
#include <vtkMapper.h>
#include <vtkCameraActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkDelaunay3D.h>
#include <vtkActor.h>
#include <vtkPolyData.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkImageBlend.h>
#include <vtkPointData.h>
#include <vtkPNGWriter.h>
#include <vtkTextProperty.h>
#include <vtkImageCanvasSource2D.h>
#include <vtkBarChartActor.h>
#include <vtkFieldData.h>
#include <vtkImageAccumulate.h>
#include <vtkImageExtractComponents.h>
#include <vtkIntArray.h>
#include <vtkJPEGReader.h>
#include <vtkLegendBoxActor.h>
#include <vtkProperty2D.h>
#include <vtkStdString.h>
#include <vtkImageData.h>
#include <vtkImageResample.h>
#include <vtkImageViewer2.h>
#include <vtkAngleWidget.h>
#include <vtkImageImport.h>
#include <vtkPointPicker.h>
#include <vtkObjectFactory.h>
#include <vtkInteractorStyleUser.h>
#include <vtkInteractorStyleTrackball.h>
#include <vtkUnsignedCharArray.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkTextActor.h>
#include <vtkDataArrayTemplate.h>
#include <vtkSphereWidget.h>
#include <vtkProp3D.h>
#include <vtkIntArray.h>
#include <vtkDataSetAttributes.h>
#include <vtkGraphLayoutView.h>
#include <vtkMutableDirectedGraph.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkGraphLayoutView.h>
#include <vtkGraphWriter.h>
#include <vtkRandomGraphSource.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkConeSource.h>
#include <vtkPlaneSource.h>
#include <vtkAffineRepresentation2D.h>
#include <vtkLogoRepresentation.h>
#include <vtkInteractorStyleUnicam.h>
#include <vtkProperty.h>

// VTK Widgets
#include <vtkPlaneWidget.h>
#include <vtkAffineWidget.h>
#include <vtkLogoWidget.h>

// VTK sources
#include <vtkConeSource.h>
#include <vtkCylinderSource.h>

// ====

#include <vtkBoxWidget.h>
#include <vtkActor.h>
#include <vtkBoxWidget.h>
#include <vtkCamera.h>
#include <vtkCommand.h>

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>

#include "Flags.h"
#include "TransferFunction.h"
#include "VolumeTracer.cuh"
#include "Statistics.h"
#include "Light.h"

#include "cutil_inline.h"
#include "curand_kernel.h"

#include <vtkAnnotationLink.h>


#include "vtkMath.h"
#include "vtkActor.h"
#include "vtkPolyDataMapper.h"
#include "vtkRenderWindow.h"
#include "vtkXMLPolyDataReader.h"

#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"

#include <vtksys/SystemTools.hxx>
#include <vtksys/CommandLineArguments.hxx>

using namespace std;

float gCudaTotalMemorySize = 0;

bool					gAbort					= false;

// CInteractorStyleRealisticCamera*					gpInteractorStyleRealisticCamera	= NULL;






/*
void TimerCallbackFunction(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData)
{
	gpImageImport->SetImportVoidPointer(NULL);
 	gpImageImport->SetImportVoidPointer(gpImageCanvas);
 	gpImageImport->Update();
 
 	gpImageActor->SetInput(gpImageImport->GetOutput());
 	
 	gpRenderWindowInteractor->Render();

	double RenderTime = gpSceneRenderer->GetLastRenderTimeInSeconds();

	char FPS[255];

	sprintf_s(FPS, "%0.2f", (float)(1.0 / RenderTime));

	gStatistics.SetValue("FPS (VTK)", FPS);
}
*/

void LoadVolume(const char* pFilePath)
{
//	BindVolumeData((short*)pImageDataResampled->GetScalarPointer(), gResolution);
}

// Main render thread which runs in the background
void RenderThreadMain(void)
{
	/*
	const int SizeRandomStates		= gCamera.m_Film.m_Resolution.m_NoElements * sizeof(curandStateXORWOW_t);
	const int SizeAccEstXyz			= gCamera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz);
	const int SizeEstFrameXyz		= gCamera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz);
	const int SizeEstFrameBlurXyz	= gCamera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz);
	const int SizeEstRgbLdr			= 3 * gCamera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char);

	gCudaTotalMemorySize += (float)SizeRandomStates / powf(1024.0f, 2.0f);
	gCudaTotalMemorySize += (float)SizeAccEstXyz / powf(1024.0f, 2.0f);
	gCudaTotalMemorySize += (float)SizeEstFrameXyz / powf(1024.0f, 2.0f);
	gCudaTotalMemorySize += (float)SizeEstFrameBlurXyz / powf(1024.0f, 2.0f);
	gCudaTotalMemorySize += (float)SizeEstRgbLdr / powf(1024.0f, 2.0f);

		// Allocate
	cudaMalloc((void**)&gpDevRandomStates, SizeRandomStates);
	cudaMalloc((void**)&gpDevAccEstXyz, SizeAccEstXyz);
	cudaMalloc((void**)&gpDevEstFrameXyz, SizeEstFrameXyz);
	cudaMalloc((void**)&gpDevEstFrameBlurXyz, SizeEstFrameBlurXyz);
	cudaMalloc((void**)&gpDevEstRgbLdr, SizeEstRgbLdr);

	// Load the volume
	LoadVolume(gpFile);

	// Set up the random number generator
	SetupRNG(gpDevRandomStates, gWidth, gHeight);

	gCamera.m_Aperture.m_Size = 0.01f;
	gCamera.m_Focus.m_FocalDistance = (gCamera.m_Target - gCamera.m_From).Length();
	gCamera.m_SceneBoundingBox = gBoundingBox;
	gCamera.SetViewMode(ViewModeFront);
	gCamera.Update();

	// Memory
	char ByteSizeRandomStates[255];
	sprintf_s(ByteSizeRandomStates, "%0.2f", (float)SizeRandomStates / powf(1024.0f, 2.0f));
	gStatistics.Add("RandomStates", ByteSizeRandomStates, "Mb", 0);

	// Buffer:AccEstXyz
	char ByteSizeAccEstXyz[255];
	sprintf_s(ByteSizeAccEstXyz, "%0.2f", (float)SizeAccEstXyz / powf(1024.0f, 2.0f));
	gStatistics.Add("AccEstXyz", ByteSizeAccEstXyz, "Mb", 0);

	// Buffer:EstFrameXyz
	char ByteSizeEstFrameXyz[255];
	sprintf_s(ByteSizeEstFrameXyz, "%0.2f", (float)SizeEstFrameXyz / powf(1024.0f, 2.0f));
	gStatistics.Add("EstFrameXyz", ByteSizeEstFrameXyz, "Mb", 0);

	// Buffer:EstFrameBlurXyz
	char ByteSizeEstFrameBlurXyz[255];
	sprintf_s(ByteSizeEstFrameBlurXyz, "%0.2f", (float)SizeEstFrameBlurXyz / powf(1024.0f, 2.0f));
	gStatistics.Add("EstFrameBlurXyz", ByteSizeEstFrameBlurXyz, "Mb", 0);

	// Buffer:EstRgbLdr
	char ByteSizeEstRgbLdr[255];
	sprintf_s(ByteSizeEstRgbLdr, "%0.2f", (float)SizeEstRgbLdr / powf(1024.0f, 2.0f));
	gStatistics.Add("EstRgbLdr", ByteSizeEstRgbLdr, "Mb", 0);

	// Buffer:Total
	char ByteSizeTotal[255];
	sprintf_s(ByteSizeTotal, "%0.2f", gCudaTotalMemorySize);
	gStatistics.Add("Total", ByteSizeTotal, "Mb", 0);
	
	gStatistics.Add("CUDA Memory");

	gStatistics.m_Offset += 10.0;

	gStatistics.Add("No. iterations", "iter");
	gStatistics.Add("FPS (VTK)", "0.00", "fps");
	gStatistics.Add("FPS (Tracer)", "0.00", "fps");
	gStatistics.Add("Max No. bounces", "0", "bounces");
	gStatistics.Add("Phase G", "0.0");

	gStatistics.Add("Aperture Size", "0.00");

	gStatistics.Add("Performance");

	gStatistics.m_Offset += 10.0;

	char Size[255];

	sprintf_s(Size, "%0.3f x %0.3f x %0.3f", gBoundingBox.LengthX(), gBoundingBox.LengthY(), gBoundingBox.LengthZ());

	char Spacing[255];

	sprintf_s(Spacing, "%0.3f x %0.3f x %0.3f", gSpacing.x, gSpacing.y, gSpacing.z);

	char Resolution[255];

	sprintf_s(Resolution, "%d x %d x %d", gResolution.m_XYZ.x, gResolution.m_XYZ.y, gResolution.m_XYZ.z);

	gStatistics.Add("Resolution", Resolution);
	gStatistics.Add("Size", Size, "m");
	gStatistics.Add("Spacing", Spacing, "m");
	gStatistics.Add("File", gpFile);

	gStatistics.SetItalicValue("File", 1);

	for (;;)
	{
		if (gAbort)
			break;

		if (gDirty)
		{
			gN = 0.0f;
			cudaMemset(gpDevAccEstXyz, 0, gWidth * gHeight * sizeof(CColorXyz));
			cudaMemset(gpDevEstFrameXyz, 0, gWidth * gHeight * sizeof(CColorXyz));
			
			// Rendering is not dirty anymore
			gDirty = false;
		}

		gN++;

		char NoIterations[255];

		sprintf_s(NoIterations, "%0.0f", gN);

		gStatistics.SetValue("No. iterations", NoIterations);

		gCamera.Update();

		CCudaTimer CudaTimer;
		
		RenderVolume(gpDevRandomStates, gWidth, gHeight, gMaxD, gMaxNoScatteringEvents, gPhaseG, gCamera, gLight, gBoundingBox, gResolution, gTransferFunctions, gpDevEstFrameXyz);
		BlurImageXyz(gpDevEstFrameXyz, gpDevEstFrameBlurXyz, CResolution2D(gWidth, gHeight), 1.3f);
		ComputeEstimate(gWidth, gHeight, gpDevEstFrameXyz, gpDevAccEstXyz, gN, 100.0f, gpDevEstRgbLdr);

		float DT = CudaTimer.StopTimer();

		char FpsTrace[255];

		sprintf_s(FpsTrace, "%0.2f", 1000.0f / DT);

		gStatistics.SetValue("FPS (Tracer)", FpsTrace);

		char MaxNoBounces[255];

		sprintf_s(MaxNoBounces, "%d", gMaxNoScatteringEvents);

		gStatistics.SetValue("Max No. bounces", MaxNoBounces);

		// Phase G
		char PhaseG[255];
		sprintf_s(PhaseG, "%0.2f", gPhaseG);
		gStatistics.SetValue("Phase G", PhaseG);

		// Aperture size
		char ApertureSize[255];
		sprintf_s(ApertureSize, "%0.3f", gCamera.m_Aperture.m_Size);
		gStatistics.SetValue("Aperture Size", ApertureSize);

		// Blit
		cudaMemcpy(gpImageCanvas, gpDevEstRgbLdr, 3 * gCamera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	}

	// Free
	cudaFree(gpDevRandomStates);
	cudaFree(gpDevAccEstXyz);
	cudaFree(gpDevEstFrameXyz);
	cudaFree(gpDevEstFrameBlurXyz);
	cudaFree(gpDevEstRgbLdr);
	*/
}

/*
// Application entry function
int main(int argc, char *argv[] )
{
	// Create opacity piece wise function
	gpPfOpacity = vtkPiecewiseFunction::New();
	gpPfOpacity->AddPoint(0.0, 0.0);
	gpPfOpacity->AddPoint(100, 1.0);

	// Create color transfer fucntion
	gpCtfDiffuse = vtkColorTransferFunction::New();
	gpCtfDiffuse->SetColorSpaceToHSV();
	gpCtfDiffuse->AddHSVPoint(0.0, 0.66, 1.0, 1.0);
	gpCtfDiffuse->AddHSVPoint(512.0, 0.33, 1.0, 1.0);
	gpCtfDiffuse->AddHSVPoint(1024.0, 0.0, 1.0, 1.0);

	if (!InitializeCuda())
		return EXIT_SUCCESS;

	gpImageCanvas = (unsigned char*)malloc(3 * gCamera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char));
	memset(gpImageCanvas, 0, 3 * gCamera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char));

	gCamera.m_Film.m_Resolution.m_XY.x	= gWidth;
	gCamera.m_Film.m_Resolution.m_XY.y	= gHeight;
	
	// Create and configure image importer
	gpImageImport = vtkImageImport::New();
	gpImageImport->SetDataSpacing(1, 1, 1);
	gpImageImport->SetDataOrigin(0, 0, 0);
	gpImageImport->SetWholeExtent(0, gWidth - 1, 0, gHeight - 1, 0, 0);
	gpImageImport->SetDataExtentToWholeExtent();
	gpImageImport->SetDataScalarTypeToUnsignedChar();
	gpImageImport->SetNumberOfScalarComponents(3);
	gpImageImport->SetImportVoidPointer(gpImageCanvas);
	gpImageImport->Update();
	
	// Create and configure background image actor
	gpImageActor = vtkImageActor::New();
	gpImageActor->SetInterpolate(1);
	gpImageActor->SetInput(gpImageImport->GetOutput());
	gpImageActor->SetScale(1,-1,-1);

	// Create background renderer
	gpBackgroundRenderer = vtkRenderer::New();
	gpBackgroundRenderer->SetLayer(0);
	gpBackgroundRenderer->InteractiveOff();
	gpBackgroundRenderer->AddActor(gpImageActor);
	gpBackgroundRenderer->SetBackground(0.2, 0.2, 0.2);

	// Create and configure scene renderer
	gpSceneRenderer = vtkRenderer::New();
	gpSceneRenderer->GetActiveCamera()->SetPosition(1.0, 1.0, 1.0);
	gpSceneRenderer->GetActiveCamera()->SetFocalPoint(0.0, 0.0, 0.0);
	gpSceneRenderer->GetActiveCamera()->SetRoll(0.0);
	gpSceneRenderer->SetLayer(1);
	gStatistics.m_pRenderer = gpSceneRenderer;

	// Create and configure render window
//	gpRenderWidget->GetRenderWindow()->SetNumberOfLayers(2);
//	gpRenderWidget->GetRenderWindow()->AddRenderer(gpBackgroundRenderer);
//	gpRenderWidget->GetRenderWindow()->AddRenderer(gpSceneRenderer);
	
	gpRenderWindowInteractor			= vtkRenderWindowInteractor::New();
	gpStyleImage						= vtkInteractorStyleImage::New();
	gpInteractorStyleRealisticCamera	= CInteractorStyleRealisticCamera::New();

	// Callbacks
	gpKeyPressCallback	= vtkCallbackCommand::New();
	gpTimerCallback		= vtkCallbackCommand::New();

	gpKeyPressCallback->SetCallback(KeyPressCallbackFunction);
	gpTimerCallback->SetCallback(TimerCallbackFunction);

	// Add observers
	gpRenderWindowInteractor->AddObserver(vtkCommand::KeyPressEvent, gpKeyPressCallback);
	gpRenderWindowInteractor->AddObserver(vtkCommand::TimerEvent, gpTimerCallback);
	
	// Render and start interaction
//	gpRenderWindowInteractor->SetRenderWindow(gpRenderWidget->GetRenderWindow());
	
	gpRenderWindowInteractor->SetInteractorStyle(gpInteractorStyleRealisticCamera);

	gpRenderWindowInteractor->Initialize();

	// Create a timer
	gpRenderWindowInteractor->CreateRepeatingTimer(1000.0f / 50.0f);

	// Start the the render that will render in the background, VTK will pickup the frames at every timer interval
	boost::thread RenderThread = boost::thread(RenderThreadMain);

	gpPfOpacity->AddPoint(512.0, 1.0);

	// Clean up memory
	gpInteractorStyleRealisticCamera->Delete();
	gpImageActor->Delete();
	gpRenderWindowInteractor->Delete();
	gpKeyPressCallback->Delete();
	gpTimerCallback->Delete();
	gpImageImport->Delete();
	gpPfOpacity->Delete();			
	gpTfOpacityChanged->Delete();
	gpTfDiffuseChanged->Delete();
	gpCtfDiffuse->Delete();
	gpSceneRenderer->Delete();
	gpBackgroundRenderer->Delete();

	free(gpImageCanvas);

	return EXIT_SUCCESS;
}
*/
