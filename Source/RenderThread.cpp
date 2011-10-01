
// Precompiled headers
#include "Stable.h"

#include "RenderThread.h"
#include "CudaUtilities.h"
#include "Scene.h"
#include "MainWindow.h"
#include "LoadSettingsDialog.h"
#include "Lighting.h"
#include "Timing.h"

// CUDA kernels
#include "Core.cuh"

// VTK
#include <vtkSmartPointer.h>
#include <vtkMetaImageReader.h>
#include <vtkImageCast.h>
#include <vtkImageResample.h>
#include <vtkImageData.h>
#include <vtkImageGradientMagnitude.h>
#include <vtkCallbackCommand.h>
#include <vtkImageAccumulate.h>
#include <vtkIntArray.h>
#include <vtkImageShiftScale.h>
#include <vtkErrorCode.h>
#include <vtkImageGradient.h>
#include <vtkExtractVectorComponents.h>
// Render thread
QRenderThread* gpRenderThread = NULL;

QRenderThread::QRenderThread(const QString& FileName, QObject* pParent /*= NULL*/) :
	QThread(pParent),
	m_FileName(FileName),
	m_CudaFrameBuffers(),
	m_pDevScene(NULL),
	m_pRenderImage(NULL),
	m_Variance(),
	m_pDevVariance(NULL),
	m_pDensityBuffer(NULL),
	m_pGradientMagnitudeBuffer(NULL),
	m_Abort(false),
	m_Scene(),
	m_Pause(false),
	m_SaveFrames(),
	m_SaveBaseName("phase_function")
{
//	m_SaveFrames << 0 << 100 << 200;

	m_Scene.m_DenoiseParams.m_Enabled = false;
}

QRenderThread::QRenderThread(const QRenderThread& Other)
{
	*this = Other;
}

QRenderThread::~QRenderThread(void)
{
	free(m_pDensityBuffer);
}

QRenderThread& QRenderThread::operator=(const QRenderThread& Other)
{
	m_FileName					= Other.m_FileName;
	m_CudaFrameBuffers			= Other.m_CudaFrameBuffers;
	m_pDevScene					= Other.m_pDevScene;
	m_Variance					= Other.m_Variance;
	m_pDevVariance				= Other.m_pDevVariance;
	m_pRenderImage				= Other.m_pRenderImage;
	m_pDensityBuffer			= Other.m_pDensityBuffer;
	m_pGradientMagnitudeBuffer	= Other.m_pGradientMagnitudeBuffer;
	m_Abort						= Other.m_Abort;
	m_Scene						= Other.m_Scene;
	m_Pause						= Other.m_Pause;
	m_SaveFrames				= Other.m_SaveFrames;
	m_SaveBaseName				= Other.m_SaveBaseName;

	return *this;
}

void QRenderThread::run()
{
	Log("Initializing CUDA...", "graphic-card");

 	if (!InitializeCuda())
 	{
		Log("Unable to initialize CUDA, rendering cannot start", QLogger::Critical);
 		QMessageBox::critical(gpMainWindow, "An error has occurred", "Unable to locate a CUDA capable device");
 		return;
 	}

 	m_Scene.m_Camera.m_Film.m_Resolution.SetResX(512);
 	m_Scene.m_Camera.m_Film.m_Resolution.SetResY(512);
 	m_Scene.m_Camera.m_SceneBoundingBox = m_Scene.m_BoundingBox;
 	m_Scene.m_Camera.SetViewMode(ViewModeFront);
 	m_Scene.m_Camera.Update();

	// Force the render thread to allocate the necessary buffers, do not remove this line
	m_Scene.m_DirtyFlags.SetFlag(FilmResolutionDirty | CameraDirty);

	gStatus.SetStatisticChanged("Memory", "CUDA Memory", "", "", "memory");
	gStatus.SetStatisticChanged("Memory", "Host Memory", "", "", "memory");

	// Bind density buffer to texture
	Log("Copying density volume to device", "grid");
	gStatus.SetStatisticChanged("CUDA Memory", "Density Buffer", QString::number(m_Scene.m_Resolution.GetNoElements() * sizeof(short) / MB, 'f', 2), "MB");
	BindDensityBuffer((short*)m_pDensityBuffer, make_cudaExtent(m_Scene.m_Resolution[0], m_Scene.m_Resolution[1], m_Scene.m_Resolution[2]));

	// Bind gradient magnitude buffer to texture
	Log("Copying gradient magnitude to device", "grid");
	gStatus.SetStatisticChanged("CUDA Memory", "Gradient Magnitude Buffer", QString::number(m_Scene.m_Resolution.GetNoElements() * sizeof(short) / MB, 'f', 2), "MB");
	BindGradientMagnitudeBuffer((short*)m_pGradientMagnitudeBuffer, make_cudaExtent(m_Scene.m_Resolution[0], m_Scene.m_Resolution[1], m_Scene.m_Resolution[2]));

	gStatus.SetStatisticChanged("Performance", "Timings", "");

	// Allocate CUDA memory for scene
	HandleCudaError(cudaMalloc((void**)&m_pDevScene, sizeof(CScene)));

	// Allocate CUDA memory for variance class
	HandleCudaError(cudaMalloc((void**)&m_pDevVariance, sizeof(CVariance)));
	
	gStatus.SetStatisticChanged("CUDA Memory", "Scene", QString::number(sizeof(CScene) / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("CUDA Memory", "Frame Buffers", "", "");

	// Let others know that we are starting with rendering
	gStatus.SetRenderBegin();
	
	Log("Device memory: " + QString::number(GetUsedCudaMemory() / MB, 'f', 2) + "/" + QString::number(GetTotalCudaMemory() / MB, 'f', 2) + " MB", "memory");

	QObject::connect(&gTransferFunction, SIGNAL(Changed()), this, SLOT(OnUpdateTransferFunction()));
	QObject::connect(&gCamera, SIGNAL(Changed()), this, SLOT(OnUpdateCamera()));
	QObject::connect(&gLighting, SIGNAL(Changed()), this, SLOT(OnUpdateLighting()));
	QObject::connect(&gLighting.Background(), SIGNAL(Changed()), this, SLOT(OnUpdateLighting()));

	// Try to load appearance/lighting/camera presets with the same name as the loaded file
	gStatus.SetLoadPreset(QFileInfo(m_FileName).baseName());

	// Keep track of frames/second
	CTiming FPS, RenderImage, BlurImage, PostProcessImage, DenoiseImage;

	try
	{
		while (!m_Abort)
		{
			if (m_Pause)
				continue;

			// Let others know we are starting with a new frame
			gStatus.SetPreRenderFrame();

			// CUDA time for profiling
 			CCudaTimer TmrFps;

			// Make a local copy of the scene, this to prevent modification to the scene from outside this thread
			CScene SceneCopy = m_Scene;

			// Update the camera, do not remove
			SceneCopy.m_Camera.Update();
		
			gStatus.SetStatisticChanged("Camera", "Position", FormatVector(m_Scene.m_Camera.m_From));
			gStatus.SetStatisticChanged("Camera", "Target", FormatVector(m_Scene.m_Camera.m_Target));
			gStatus.SetStatisticChanged("Camera", "Up Vector", FormatVector(m_Scene.m_Camera.m_Up));

			// Resizing the image canvas requires special attention
			if (SceneCopy.m_DirtyFlags.HasFlag(FilmResolutionDirty))
			{
				// Allocate host image buffer, this thread will blit it's frames to this buffer
				free(m_pRenderImage);
				m_pRenderImage = NULL;

				m_pRenderImage = (CColorRgbaLdr*)malloc(SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbaLdr));

				if (m_pRenderImage)
					memset(m_pRenderImage, 0, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbaLdr));
			
				gStatus.SetStatisticChanged("Host Memory", "LDR Frame Buffer", QString::number(3 * SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbaLdr) / MB, 'f', 2), "MB");

				m_CudaFrameBuffers.Resize(Vec2i(SceneCopy.m_Camera.m_Film.GetWidth(), SceneCopy.m_Camera.m_Film.GetHeight()));

				// Reset no. iterations
				m_Scene.SetNoIterations(0);

				// Notify Inform others about the memory allocations
				gStatus.SetResize();

				BindEstimateRgbLdr(m_CudaFrameBuffers.m_pDevEstRgbaLdr, SceneCopy.m_Camera.m_Film.m_Resolution.GetResX(), SceneCopy.m_Camera.m_Film.m_Resolution.GetResY());

				m_Variance.Resize(SceneCopy.m_Camera.m_Film.m_Resolution.GetResX(), SceneCopy.m_Camera.m_Film.m_Resolution.GetResY());
			}

			// Restart the rendering when when the camera, lights and render params are dirty
			if (SceneCopy.m_DirtyFlags.HasFlag(CameraDirty | LightsDirty | RenderParamsDirty | TransferFunctionDirty))
			{
				m_CudaFrameBuffers.Reset();

				// Reset no. iterations
				m_Scene.SetNoIterations(0);

				// Reset variance
	//			m_Variance.Reset();
			}

			gStatus.SetStatisticChanged("Film", "Width m_Scene", QString::number(m_Scene.m_Camera.m_Film.m_Resolution.GetResX()), "Pixels");
			gStatus.SetStatisticChanged("Film", "Height m_Scene", QString::number(m_Scene.m_Camera.m_Film.m_Resolution.GetResY()), "Pixels");

			gStatus.SetStatisticChanged("Film", "Width SceneCopy", QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResX()), "Pixels");
			gStatus.SetStatisticChanged("Film", "Height SceneCopy", QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResY()), "Pixels");

			// At this point, all dirty flags should have been taken care of, since the flags in the original scene are now cleared
			m_Scene.m_DirtyFlags.ClearAllFlags();

			// Increase the number of iterations performed so far
			m_Scene.SetNoIterations(m_Scene.GetNoIterations() + 1);

			// Adjust de-noising parameters
			const float Radius = 0.015 * (float)m_Scene.GetNoIterations();

			gStatus.SetStatisticChanged("Denoise", "Radius", QString::number(Radius, 'f', 2), "ms.");

			if (Radius < 1.0f)
			{
				m_Scene.m_DenoiseParams.m_Enabled = true;
				m_Scene.m_DenoiseParams.SetWindowRadius(6.0f);
				m_Scene.m_DenoiseParams.m_LerpC = Radius;
			}
			else
			{
				m_Scene.m_DenoiseParams.m_Enabled = false;
			}
			
		//	m_Scene.m_DenoiseParams.m_LerpC = 1.0f - Radius;

			cudaMemcpy(m_pDevScene, &SceneCopy, sizeof(CScene), cudaMemcpyHostToDevice);
			cudaMemcpy(m_pDevVariance, &m_Variance, sizeof(CVariance), cudaMemcpyHostToDevice);

			// Execute the rendering kernels
  			Render(0, &SceneCopy, m_pDevScene, m_CudaFrameBuffers, m_Scene.GetNoIterations(), m_pDevVariance, m_Variance.GetVarianceBuffer(), RenderImage, BlurImage, PostProcessImage, DenoiseImage);
			HandleCudaError(cudaGetLastError());
		
			gStatus.SetStatisticChanged("Timings", "Render Image", QString::number(RenderImage.m_FilteredDuration, 'f', 2), "ms.");
			gStatus.SetStatisticChanged("Timings", "Blur Estimate", QString::number(BlurImage.m_FilteredDuration, 'f', 2), "ms.");
			gStatus.SetStatisticChanged("Timings", "Post Process Estimate", QString::number(PostProcessImage.m_FilteredDuration, 'f', 2), "ms.");
			gStatus.SetStatisticChanged("Timings", "De-noise Image", QString::number(DenoiseImage.m_FilteredDuration, 'f', 2), "ms.");

			FPS.AddDuration(1000.0f / TmrFps.ElapsedTime());

 			gStatus.SetStatisticChanged("Performance", "FPS", QString::number(FPS.m_FilteredDuration, 'f', 2), "Frames/Sec.");
 			gStatus.SetStatisticChanged("Performance", "No. Iterations", QString::number(m_Scene.GetNoIterations()), "Iterations");

	// 		gStatus.SetStatisticChanged("Test", "Magnitude", QString::number(SceneCopy.m_Variance), "ms.");
	//		printf("%.2f", SceneCopy.m_Variance);

			HandleCudaError(cudaMemcpy(m_pRenderImage, m_CudaFrameBuffers.m_pDevRgbLdrDisp, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr), cudaMemcpyDeviceToHost));

			if (m_SaveFrames.indexOf(m_Scene.GetNoIterations()) > 0)
			{
				const QString ImageFilePath = QApplication::applicationDirPath() + "/Output/" + m_SaveBaseName + "_" + QString::number(m_Scene.GetNoIterations()) + ".png";

				SaveImage((unsigned char*)m_pRenderImage, m_Scene.m_Camera.m_Film.m_Resolution.GetResX(), m_Scene.m_Camera.m_Film.m_Resolution.GetResY(), ImageFilePath);
			}

			// Let others know we are finished with a frame
 			gStatus.SetPostRenderFrame();
		}
	}
	catch (QString* pMessage)
	{
		Log("A critical error has occured: " + *pMessage);
	}

	// Free CUDA buffers
	HandleCudaError(cudaFree(m_pDevScene));
	HandleCudaError(cudaFree(m_pDevVariance));
	
	m_pDevScene		= NULL;
	m_pDevVariance	= NULL;

	// Free render image buffer
	free(m_pRenderImage);
	m_pRenderImage = NULL;

	UnbindDensityBuffer();

	// Let others know that we have stopped rendering
	gStatus.SetRenderEnd();

	Log("Device memory: " + QString::number(GetUsedCudaMemory() / MB, 'f', 2) + "/" + QString::number(GetTotalCudaMemory() / MB, 'f', 2) + " MB", "memory");

	// Load default appearance, lighting and camera presets
	gStatus.SetLoadPreset("Default");

	// Clear the histogram
	gHistogram.Reset();
}

bool QRenderThread::Load(QString& FileName)
{
	m_FileName = FileName;

	// Create meta image reader
	vtkSmartPointer<vtkMetaImageReader> MetaImageReader = vtkMetaImageReader::New();

	QFileInfo FileInfo(FileName);

	if (!FileInfo.exists())
	{
		Log(QString(QFileInfo(FileName).filePath().replace("//", "/")).toAscii() + "  does not exist!", QLogger::Critical);
		return false;
	}

	// Exit if the reader can't read the file
	if (!MetaImageReader->CanReadFile(m_FileName.toAscii()))
	{
		Log(QString("Meta image reader can't read file " + QFileInfo(FileName).fileName()).toAscii(), QLogger::Critical);
		return false;
	}

	MetaImageReader->SetFileName(m_FileName.toAscii());

	Log(QString("Loading " + QFileInfo(FileName).fileName()).toAscii());

	MetaImageReader->Update();

	if (MetaImageReader->GetErrorCode() != vtkErrorCode::NoError)
	{
		Log("Error loading file " + QString(vtkErrorCode::GetStringFromErrorCode(MetaImageReader->GetErrorCode())));
		return false;
	}

	vtkSmartPointer<vtkImageCast> ImageCast = vtkImageCast::New();
	
	Log("Casting volume data type to short", "grid");

	ImageCast->SetInput(MetaImageReader->GetOutput());
	ImageCast->SetOutputScalarTypeToShort();
	ImageCast->Update();

	if (ImageCast->GetErrorCode() != vtkErrorCode::NoError)
	{
		Log("vtkImageCast error: " + QString(vtkErrorCode::GetStringFromErrorCode(MetaImageReader->GetErrorCode())));
		return false;
	}
	
	// Volume resolution
	int* pVolumeResolution = ImageCast->GetOutput()->GetExtent();
	m_Scene.m_Resolution.SetResXYZ(Vec3i(pVolumeResolution[1] + 1, pVolumeResolution[3] + 1, pVolumeResolution[5] + 1));

	Log("Resolution: " + FormatSize(m_Scene.m_Resolution.GetResXYZ()) + "", "grid");

	// Intensity range
	double* pIntensityRange = ImageCast->GetOutput()->GetScalarRange();
	m_Scene.m_IntensityRange.SetMin((float)pIntensityRange[0]);
	m_Scene.m_IntensityRange.SetMax((float)pIntensityRange[1]);

	Log("Intensity range: [" + QString::number(m_Scene.m_IntensityRange.GetMin()) + ", " + QString::number(m_Scene.m_IntensityRange.GetMax()) + "]", "grid");

	// Spacing
	double* pSpacing = ImageCast->GetOutput()->GetSpacing();

	m_Scene.m_Spacing.x = (float)pSpacing[0];
	m_Scene.m_Spacing.y = (float)pSpacing[1];
	m_Scene.m_Spacing.z = (float)pSpacing[2];

	Log("Spacing: " + FormatSize(m_Scene.m_Spacing, 2), "grid");

	// Compute physical size
	const Vec3f PhysicalSize(Vec3f(m_Scene.m_Spacing.x * (float)m_Scene.m_Resolution.GetResX(), m_Scene.m_Spacing.y * (float)m_Scene.m_Resolution.GetResY(), m_Scene.m_Spacing.z * (float)m_Scene.m_Resolution.GetResZ()));

	// Compute the volume's bounding box
	m_Scene.m_BoundingBox.m_MinP	= Vec3f(0.0f);
	m_Scene.m_BoundingBox.m_MaxP	= PhysicalSize / PhysicalSize.Max();

	m_Scene.m_GradientDelta = 1.0f / (float)m_Scene.m_Resolution.GetMax();
	
	Log("Bounding box: " + FormatVector(m_Scene.m_BoundingBox.m_MinP, 2) + " - " + FormatVector(m_Scene.m_BoundingBox.m_MaxP), "grid");
	
	const int DensityBufferSize = m_Scene.m_Resolution.GetNoElements() * sizeof(short);

 	m_pDensityBuffer = (short*)malloc(DensityBufferSize);
  	memcpy(m_pDensityBuffer, ImageCast->GetOutput()->GetScalarPointer(), DensityBufferSize);

	// Gradient magnitude volume
	vtkSmartPointer<vtkImageGradientMagnitude> GradientMagnitude = vtkImageGradientMagnitude::New();
	
	Log("Creating gradient magnitude volume", "grid");
		
	GradientMagnitude->SetDimensionality(3);
	GradientMagnitude->SetInput(ImageCast->GetOutput());
	GradientMagnitude->Update();

	vtkImageData* GradientMagnitudeBuffer = GradientMagnitude->GetOutput();
	
	// Scalar range of the gradient magnitude
	double* pGradientMagnitudeRange = GradientMagnitudeBuffer->GetScalarRange();
	
	m_Scene.m_GradientMagnitudeRange.SetMin((float)pGradientMagnitudeRange[0]);
	m_Scene.m_GradientMagnitudeRange.SetMax((float)pGradientMagnitudeRange[1]);
	
	Log("Gradient magnitude range: [" + QString::number(m_Scene.m_GradientMagnitudeRange.GetMin(), 'f', 2) + " - " + QString::number(m_Scene.m_GradientMagnitudeRange.GetMax(), 'f', 2) + "]", "grid");
	
	const int GradientMagnitudeBufferSize = m_Scene.m_Resolution.GetNoElements() * sizeof(short);
	
	m_pGradientMagnitudeBuffer = (short*)malloc(GradientMagnitudeBufferSize);
	memcpy(m_pGradientMagnitudeBuffer, GradientMagnitudeBuffer->GetScalarPointer(), GradientMagnitudeBufferSize);

	// Build the histogram
	Log("Creating gradient magnitude histogram", "grid");

	vtkSmartPointer<vtkImageAccumulate> GradMagHistogram = vtkSmartPointer<vtkImageAccumulate>::New();

	GradMagHistogram->SetInputConnection(GradientMagnitude->GetOutputPort());
	GradMagHistogram->SetComponentExtent(0, 255, 0, 0, 0, 0);
	GradMagHistogram->SetComponentOrigin(0, 0, 0);
	GradMagHistogram->SetComponentSpacing(m_Scene.m_GradientMagnitudeRange.GetLength() / 256.0f, 0, 0);
//	GradMagHistogram->IgnoreZeroOn();
	GradMagHistogram->Update();

	m_Scene.m_GradMagMean = (float)GradMagHistogram->GetMean()[0];
	m_Scene.m_GradientFactor = m_Scene.m_GradMagMean;

	Log("Mean gradient magnitude: " + QString::number(m_Scene.m_GradMagMean, 'f', 2), "grid");

	Log("Creating density histogram", "grid");

	// Build the histogram
	vtkSmartPointer<vtkImageAccumulate> Histogram = vtkSmartPointer<vtkImageAccumulate>::New();

	Log("Creating histogram", "grid");

 	Histogram->SetInputConnection(ImageCast->GetOutputPort());
 	Histogram->SetComponentExtent(0, 256, 0, 0, 0, 0);
 	Histogram->SetComponentOrigin(m_Scene.m_IntensityRange.GetMin(), 0, 0);
 	Histogram->SetComponentSpacing(m_Scene.m_IntensityRange.GetLength() / 256.0f, 0, 0);
 	Histogram->IgnoreZeroOn();
 	Histogram->Update();
 
	// Update the histogram in the transfer function
	gHistogram.SetBins((int*)Histogram->GetOutput()->GetScalarPointer(), 256);
	
	gStatus.SetStatisticChanged("Volume", "File", QFileInfo(m_FileName).fileName(), "");
	gStatus.SetStatisticChanged("Volume", "Bounding Box", "", "");
	gStatus.SetStatisticChanged("Bounding Box", "Min", FormatVector(m_Scene.m_BoundingBox.m_MinP, 2), "m");
	gStatus.SetStatisticChanged("Bounding Box", "Max", FormatVector(m_Scene.m_BoundingBox.m_MaxP, 2), "m");
	gStatus.SetStatisticChanged("Volume", "Physical Size", FormatSize(PhysicalSize, 2), "mm");
	gStatus.SetStatisticChanged("Volume", "Resolution", FormatSize(m_Scene.m_Resolution.GetResXYZ()), "Voxels");
//	gStatus.SetStatisticChanged("Volume", "Extinction Resolution", FormatSize(Vec3i((int)m_Scene.m_ExtinctionSize.width, (int)m_Scene.m_ExtinctionSize.height, (int)m_Scene.m_ExtinctionSize.depth)), "Voxels");
	gStatus.SetStatisticChanged("Volume", "Spacing", FormatSize(m_Scene.m_Spacing, 2), "mm");
	gStatus.SetStatisticChanged("Volume", "No. Voxels", QString::number(m_Scene.m_Resolution.GetNoElements()), "Voxels");
	gStatus.SetStatisticChanged("Volume", "Density Range", "[" + QString::number(m_Scene.m_IntensityRange.GetMin()) + ", " + QString::number(m_Scene.m_IntensityRange.GetMax()) + "]", "");

	return true;
}

void QRenderThread::OnUpdateTransferFunction(void)
{
	QTransferFunction TransferFunction = gTransferFunction;

	m_Scene.m_TransferFunctions.m_Opacity.m_NoNodes		= TransferFunction.GetNodes().size();
	m_Scene.m_TransferFunctions.m_Diffuse.m_NoNodes		= TransferFunction.GetNodes().size();
	m_Scene.m_TransferFunctions.m_Specular.m_NoNodes	= TransferFunction.GetNodes().size();
	m_Scene.m_TransferFunctions.m_Emission.m_NoNodes	= TransferFunction.GetNodes().size();
	m_Scene.m_TransferFunctions.m_Roughness.m_NoNodes	= TransferFunction.GetNodes().size();

	for (int i = 0; i < TransferFunction.GetNodes().size(); i++)
	{
		QNode& Node = TransferFunction.GetNode(i);

		const float Intensity = Scene()->m_IntensityRange.GetMin() + Scene()->m_IntensityRange.GetLength() * Node.GetIntensity();

		// Positions
		m_Scene.m_TransferFunctions.m_Opacity.m_P[i]	= Intensity;
		m_Scene.m_TransferFunctions.m_Diffuse.m_P[i]	= Intensity;
		m_Scene.m_TransferFunctions.m_Specular.m_P[i]	= Intensity;
		m_Scene.m_TransferFunctions.m_Emission.m_P[i]	= Intensity;
		m_Scene.m_TransferFunctions.m_Roughness.m_P[i]	= Intensity;

		// Colors
		m_Scene.m_TransferFunctions.m_Opacity.m_C[i]	= CColorRgbHdr(Node.GetOpacity());
		m_Scene.m_TransferFunctions.m_Diffuse.m_C[i]	= CColorRgbHdr(Node.GetDiffuse().redF(), Node.GetDiffuse().greenF(), Node.GetDiffuse().blueF());
		m_Scene.m_TransferFunctions.m_Specular.m_C[i]	= CColorRgbHdr(Node.GetSpecular().redF(), Node.GetSpecular().greenF(), Node.GetSpecular().blueF());
		m_Scene.m_TransferFunctions.m_Emission.m_C[i]	= 500.0f * CColorRgbHdr(Node.GetEmission().redF(), Node.GetEmission().greenF(), Node.GetEmission().blueF());

		const float Roughness = 1.0f - expf(-Node.GetGlossiness());

		m_Scene.m_TransferFunctions.m_Roughness.m_C[i]	= CColorRgbHdr(Roughness * 100);
	}

	m_Scene.m_DensityScale	= TransferFunction.GetDensityScale();
	m_Scene.m_ShadingType	= TransferFunction.GetShadingType();

	m_Scene.m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void QRenderThread::OnUpdateCamera(void)
{
	if (!Scene())
		return;

	// Film
	Scene()->m_Camera.m_Film.m_Exposure = (1.001f - gCamera.GetFilm().GetExposure()) * 1000.0f;

	if (gCamera.GetFilm().IsDirty())
	{
// 		Scene()->m_Camera.m_Film.m_Resolution.SetResX(gCamera.GetFilm().GetWidth());
// 		// 
// 		Scene()->m_DirtyFlags.SetFlag(FilmResolutionDirty);
	}

// 	Scene()->m_Camera.m_From	= gCamera.GetFrom();
// 	Scene()->m_Camera.m_Target	= gCamera.GetTarget();
// 	Scene()->m_Camera.m_Up		= gCamera.GetUp();

	Scene()->m_Camera.Update();

	// Aperture
	Scene()->m_Camera.m_Aperture.m_Size	= gCamera.GetAperture().GetSize();

	// Projection
	Scene()->m_Camera.m_FovV = gCamera.GetProjection().GetFieldOfView();

	// Focus
	Scene()->m_Camera.m_Focus.m_FocalDistance = gCamera.GetFocus().GetFocalDistance();

	Scene()->m_DirtyFlags.SetFlag(CameraDirty);
}

void QRenderThread::OnUpdateLighting(void)
{
	if (!Scene())
		return;

	Scene()->m_Lighting.Reset();

	if (gLighting.Background().GetEnabled())
	{
		CLight BackgroundLight;

		BackgroundLight.m_T	= 1;

		BackgroundLight.m_ColorTop		= gLighting.Background().GetIntensity() * CColorRgbHdr(gLighting.Background().GetTopColor().redF(), gLighting.Background().GetTopColor().greenF(), gLighting.Background().GetTopColor().blueF());
		BackgroundLight.m_ColorMiddle	= gLighting.Background().GetIntensity() * CColorRgbHdr(gLighting.Background().GetMiddleColor().redF(), gLighting.Background().GetMiddleColor().greenF(), gLighting.Background().GetMiddleColor().blueF());
		BackgroundLight.m_ColorBottom	= gLighting.Background().GetIntensity() * CColorRgbHdr(gLighting.Background().GetBottomColor().redF(), gLighting.Background().GetBottomColor().greenF(), gLighting.Background().GetBottomColor().blueF());

		BackgroundLight.Update(Scene()->m_BoundingBox);

		Scene()->m_Lighting.AddLight(BackgroundLight);
	}

	for (int i = 0; i < gLighting.GetLights().size(); i++)
	{
		QLight& Light = gLighting.GetLights()[i];

		CLight AreaLight;

		AreaLight.m_T			= 0;
		AreaLight.m_Theta		= Light.GetTheta() / RAD_F;
		AreaLight.m_Phi			= Light.GetPhi() / RAD_F;
		AreaLight.m_Width		= Light.GetWidth();
		AreaLight.m_Height		= Light.GetHeight();
		AreaLight.m_Distance	= Light.GetDistance();
		AreaLight.m_Color		= Light.GetIntensity() * CColorRgbHdr(Light.GetColor().redF(), Light.GetColor().greenF(), Light.GetColor().blueF());

		AreaLight.Update(Scene()->m_BoundingBox);

		Scene()->m_Lighting.AddLight(AreaLight);
	}

	Scene()->m_DirtyFlags.SetFlag(LightsDirty);
}

CColorRgbaLdr* QRenderThread::GetRenderImage(void) const
{
	return m_pRenderImage;
}

CScene* Scene(void)
{
	return gpRenderThread ? gpRenderThread->GetScene() : NULL;
}

void StartRenderThread(QString& FileName)
{
	// Create new render thread
 	gpRenderThread = new QRenderThread(FileName);
  //	gpRenderThread->setStackSize(1500000000);

	// Load the volume
 	if (!gpRenderThread->Load(FileName))
 		return;
 
	// Start the render thread
	gpRenderThread->start();
}

void KillRenderThread(void)
{
 	if (!gpRenderThread)
 		return;
 
	// Kill the render thread
	gpRenderThread->Close();

	// Wait for thread to end
	gpRenderThread->wait();

	// Remove the render thread
	delete gpRenderThread;
	gpRenderThread = NULL;
}

