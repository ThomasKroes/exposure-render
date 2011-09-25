
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

// Render thread
QRenderThread* gpRenderThread = NULL;

QRenderThread::QRenderThread(const QString& FileName, QObject* pParent /*= NULL*/) :
	QThread(pParent),
	m_FileName(FileName),
	m_pScene(NULL),
	m_pDevAccEstXyz(NULL),
	m_pDevEstFrameXyz(NULL),
	m_pDevEstFrameBlurXyz(NULL),
	m_pDevEstRgbaLdr(NULL),
	m_pDevRgbLdrDisp(NULL),
	m_pRenderImage(NULL),
	m_pDevSeeds(NULL),
	m_pDensityBuffer(NULL),
	m_pGradientMagnitudeBuffer(NULL),
	m_Mutex(),
	m_Abort(false),
	m_Scene(),
	m_Pause(false)
{
}

QRenderThread::QRenderThread(const QRenderThread& Other)
{
	*this = Other;
}

QRenderThread& QRenderThread::operator=(const QRenderThread& Other)
{
	m_FileName						= Other.m_FileName;
	m_pScene						= Other.m_pScene;
	m_pDevAccEstXyz					= Other.m_pDevAccEstXyz;
	m_pDevEstFrameXyz				= Other.m_pDevEstFrameXyz;
	m_pDevEstFrameBlurXyz			= Other.m_pDevEstFrameBlurXyz;
	m_pDevEstRgbaLdr				= Other.m_pDevEstRgbaLdr;
	m_pDevRgbLdrDisp				= Other.m_pDevRgbLdrDisp;
	m_pRenderImage					= Other.m_pRenderImage;
	m_pDevSeeds						= Other.m_pDevSeeds;
	m_pDensityBuffer				= Other.m_pDensityBuffer;
	m_pGradientMagnitudeBuffer		= Other.m_pGradientMagnitudeBuffer;
//	m_Mutex							= Other.m_Mutex;
	m_Abort							= Other.m_Abort;
	m_Scene							= Other.m_Scene;
	m_Pause							= Other.m_Pause;

	return *this;
}

QRenderThread::~QRenderThread(void)
{
	free(m_pDensityBuffer);
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

	m_Scene.m_Camera.m_Film.m_Resolution.Set(Vec2i(640, 480));
 	m_Scene.m_Camera.m_SceneBoundingBox = m_Scene.m_BoundingBox;
 	m_Scene.m_Camera.SetViewMode(ViewModeFront);
 	m_Scene.m_Camera.Update();

	// Force the render thread to allocate the necessary buffers, do not remove this line
	m_Scene.m_DirtyFlags.SetFlag(FilmResolutionDirty | CameraDirty);

	gStatus.SetStatisticChanged("Memory", "CUDA Memory", "", "", ":/Images/memory.png");
	gStatus.SetStatisticChanged("Memory", "Host Memory", "", "", ":/Images/memory.png");

//	Load(m_FileName);

	// Bind density buffer to texture
	Log("Copying density volume to device", "grid");
	gStatus.SetStatisticChanged("CUDA Memory", "Density Buffer", QString::number(m_Scene.m_Resolution.GetNoElements() * sizeof(short) / MB, 'f', 2), "MB", ":/Images/memory.png");
	BindDensityBuffer((short*)m_pDensityBuffer, make_cudaExtent(m_Scene.m_Resolution[0], m_Scene.m_Resolution[1], m_Scene.m_Resolution[2]));

	// Bind gradient magnitude buffer to texture
	Log("Copying gradient magnitude to device", "grid");
	gStatus.SetStatisticChanged("CUDA Memory", "Gradient Magnitude Buffer", QString::number(m_Scene.m_Resolution.GetNoElements() * sizeof(short) / MB, 'f', 2), "MB", ":/Images/memory.png");
	BindGradientMagnitudeBuffer((short*)m_pGradientMagnitudeBuffer, make_cudaExtent(m_Scene.m_Resolution[0], m_Scene.m_Resolution[1], m_Scene.m_Resolution[2]));

	gStatus.SetStatisticChanged("Performance", "Timings", "");

	// Allocate CUDA memory for scene
	HandleCudaError(cudaMalloc((void**)&m_pScene, sizeof(CScene)));

	gStatus.SetStatisticChanged("CUDA Memory", "Scene", QString::number(sizeof(CScene) / MB, 'f', 2), "MB");

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

	// Performance
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

		// Copy scene from host to device
		cudaMemcpy(m_pScene, &SceneCopy, sizeof(CScene), cudaMemcpyHostToDevice);

		// Resizing the image canvas requires special attention
		if (SceneCopy.m_DirtyFlags.HasFlag(FilmResolutionDirty))
		{
			m_Mutex.lock();

			// Allocate host image buffer, this thread will blit it's frames to this buffer
			free(m_pRenderImage);
			m_pRenderImage = NULL;

			m_pRenderImage = (CColorRgbaLdr*)malloc(SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbaLdr));

			if (m_pRenderImage)
				memset(m_pRenderImage, 0, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbaLdr));
			
			gStatus.SetStatisticChanged("Host Memory", "LDR Frame Buffer", QString::number(3 * SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbaLdr) / MB, 'f', 2), "MB");

			m_Mutex.unlock();

			const int NoRandomSeeds = SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * 2;

			// Compute size of the CUDA buffers
 			float SizeRandomSeeds				= NoRandomSeeds * sizeof(unsigned int);
			float SizeHdrAccumulationBuffer		= SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorXyz);
			float SizeHdrFrameBuffer			= SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorXyz);
			float SizeHdrBlurFrameBuffer		= SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorXyz);
			float SizeRgbaLdrFrameBuffer		= SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbaLdr);
			
			HandleCudaError(cudaFree(m_pDevSeeds));
			HandleCudaError(cudaFree(m_pDevAccEstXyz));
			HandleCudaError(cudaFree(m_pDevEstFrameXyz));
			HandleCudaError(cudaFree(m_pDevEstFrameBlurXyz));
			HandleCudaError(cudaFree(m_pDevEstRgbaLdr));
			HandleCudaError(cudaFree(m_pDevRgbLdrDisp));

			// Allocate device buffers
			HandleCudaError(cudaMalloc((void**)&m_pDevSeeds, SizeRandomSeeds));
			HandleCudaError(cudaMalloc((void**)&m_pDevAccEstXyz, SizeHdrAccumulationBuffer));
			HandleCudaError(cudaMalloc((void**)&m_pDevEstFrameXyz, SizeHdrFrameBuffer));
			HandleCudaError(cudaMalloc((void**)&m_pDevEstFrameBlurXyz, SizeHdrBlurFrameBuffer));
			HandleCudaError(cudaMalloc((void**)&m_pDevEstRgbaLdr, SizeRgbaLdrFrameBuffer));
			HandleCudaError(cudaMalloc((void**)&m_pDevRgbLdrDisp, SizeRgbaLdrFrameBuffer));

			gStatus.SetStatisticChanged("CUDA Memory", "Random Seeds", QString::number(SizeRandomSeeds / MB, 'f', 2), "MB");
			gStatus.SetStatisticChanged("CUDA Memory", "HDR Accumulation Buffer", QString::number(SizeHdrFrameBuffer / MB, 'f', 2), "MB");
			gStatus.SetStatisticChanged("CUDA Memory", "HDR Frame Buffer Blur", QString::number(SizeHdrBlurFrameBuffer / MB, 'f', 2), "MB");
			gStatus.SetStatisticChanged("CUDA Memory", "LDRA Estimate", QString::number(SizeRgbaLdrFrameBuffer / MB, 'f', 2), "MB");
			gStatus.SetStatisticChanged("CUDA Memory", "LDR Estimate", QString::number(SizeRgbaLdrFrameBuffer / MB, 'f', 2), "MB");
			
			// Create random seeds
			unsigned int* pSeeds = (unsigned int*)malloc(SizeRandomSeeds);

			for (int i = 0; i < NoRandomSeeds; i++)
				pSeeds[i] = qrand();

			HandleCudaError(cudaMemcpy(m_pDevSeeds, pSeeds, SizeRandomSeeds, cudaMemcpyHostToDevice));

			free(pSeeds);

			// Reset buffers to black
			HandleCudaError(cudaMemset(m_pDevAccEstXyz, 0, SizeHdrAccumulationBuffer));
			HandleCudaError(cudaMemset(m_pDevEstFrameXyz, 0, SizeHdrFrameBuffer));
			HandleCudaError(cudaMemset(m_pDevEstFrameBlurXyz, 0, SizeHdrBlurFrameBuffer));
			HandleCudaError(cudaMemset(m_pDevEstRgbaLdr, 0, SizeRgbaLdrFrameBuffer));
			HandleCudaError(cudaMemset(m_pDevRgbLdrDisp, 0, SizeRgbaLdrFrameBuffer));

			// Reset no. iterations
			m_Scene.SetNoIterations(0);

			// Notify Inform others about the memory allocations
			gStatus.SetResize();

			BindEstimateRgbLdr(m_pDevEstRgbaLdr, m_Scene.m_Camera.m_Film.m_Resolution.GetResX(), m_Scene.m_Camera.m_Film.m_Resolution.GetResY());

			gStatus.SetStatisticChanged("Camera", "Resolution", QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResX()) + " x " + QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResY()), "Pixels");
		}

		// Restart the rendering when when the camera, lights and render params are dirty
		if (SceneCopy.m_DirtyFlags.HasFlag(CameraDirty | LightsDirty | RenderParamsDirty | TransferFunctionDirty))
		{
			// Reset buffers to black
			HandleCudaError(cudaMemset(m_pDevAccEstXyz, 0, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorXyz)));

			// Reset no. iterations
			m_Scene.SetNoIterations(0);
		}

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

		// Execute the rendering kernels
  		Render(0, &SceneCopy, m_pScene, m_pDevSeeds, m_pDevEstFrameXyz, m_pDevEstFrameBlurXyz, m_pDevAccEstXyz, m_pDevEstRgbaLdr, m_pDevRgbLdrDisp, m_Scene.GetNoIterations(), RenderImage, BlurImage, PostProcessImage, DenoiseImage);
		HandleCudaError(cudaGetLastError());
		
		gStatus.SetStatisticChanged("Timings", "Render Image", QString::number(RenderImage.m_FilteredDuration, 'f', 2), "ms.");
		gStatus.SetStatisticChanged("Timings", "Blur Estimate", QString::number(BlurImage.m_FilteredDuration, 'f', 2), "ms.");
		gStatus.SetStatisticChanged("Timings", "Post Process Estimate", QString::number(PostProcessImage.m_FilteredDuration, 'f', 2), "ms.");
		gStatus.SetStatisticChanged("Timings", "De-noise Image", QString::number(DenoiseImage.m_FilteredDuration, 'f', 2), "ms.");

		FPS.AddDuration(1000.0f / TmrFps.ElapsedTime());

 		gStatus.SetStatisticChanged("Performance", "FPS", QString::number(FPS.m_FilteredDuration, 'f', 2), "Frames/Sec.");
 		gStatus.SetStatisticChanged("Performance", "No. Iterations", QString::number(m_Scene.GetNoIterations()), "Iterations");

		HandleCudaError(cudaMemcpy(m_pRenderImage, m_pDevRgbLdrDisp, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbaLdr), cudaMemcpyDeviceToHost));

		// Let others know we are finished with a frame
 		gStatus.SetPostRenderFrame();
	}

	// Free CUDA buffers
	HandleCudaError(cudaFree(m_pScene));
	HandleCudaError(cudaFree(m_pDevAccEstXyz));
	HandleCudaError(cudaFree(m_pDevEstFrameXyz));
	HandleCudaError(cudaFree(m_pDevEstFrameBlurXyz));
	HandleCudaError(cudaFree(m_pDevEstRgbaLdr));
	HandleCudaError(cudaFree(m_pDevRgbLdrDisp));
	HandleCudaError(cudaFree(m_pDevSeeds));
	
	m_pScene				= NULL;
	m_pDevSeeds				= NULL;
	m_pDevAccEstXyz			= NULL;
	m_pDevEstFrameXyz		= NULL;
	m_pDevEstFrameBlurXyz	= NULL;
	m_pDevEstRgbaLdr		= NULL;
	m_pDevSeeds				= NULL;

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
		m_Scene.m_TransferFunctions.m_Roughness.m_C[i]	= CColorRgbHdr(0.0001f + (10000.0f * powf(Node.GetGlossiness(), 2.0f)));
	}

	m_Scene.m_DensityScale = TransferFunction.GetDensityScale();

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
		// 
		// 		Scene()->m_DirtyFlags.SetFlag(FilmResolutionDirty);
	}

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

