
// Precompiled headers
#include "Stable.h"

#include "RenderThread.h"
#include "CudaUtilities.h"
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

QMutex gSceneMutex;

QRenderThread::QRenderThread(const QString& FileName, QObject* pParent /*= NULL*/) :
	QThread(pParent),
	m_FileName(FileName),
	m_CudaFrameBuffers(),
	m_pDevScene(NULL),
	m_pRenderImage(NULL),
	m_pDensityBuffer(NULL),
	m_pGradientMagnitudeBuffer(NULL),
	m_Abort(false),
	m_Pause(false),
	m_SaveFrames(),
	m_SaveBaseName("phase_function")
{
//	m_SaveFrames << 0 << 100 << 200;
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
	m_pRenderImage				= Other.m_pRenderImage;
	m_pDensityBuffer			= Other.m_pDensityBuffer;
	m_pGradientMagnitudeBuffer	= Other.m_pGradientMagnitudeBuffer;
	m_Abort						= Other.m_Abort;
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

	CScene SceneCopy;
	
 	gScene.m_Camera.m_SceneBoundingBox = gScene.m_BoundingBox;
	gScene.m_Camera.SetViewMode(ViewModeFront);
 	gScene.m_Camera.Update();

	// Force the render thread to allocate the necessary buffers, do not remove this line
	gScene.m_DirtyFlags.SetFlag(FilmResolutionDirty | CameraDirty);

	gStatus.SetStatisticChanged("Memory", "CUDA Memory", "", "", "memory");
	gStatus.SetStatisticChanged("Memory", "Host Memory", "", "", "memory");

	// Bind density buffer to texture
	Log("Copying density volume to device", "grid");
	gStatus.SetStatisticChanged("CUDA Memory", "Density Buffer", QString::number(gScene.m_Resolution.GetNoElements() * sizeof(short) / MB, 'f', 2), "MB");
	BindDensityBuffer((short*)m_pDensityBuffer, make_cudaExtent(gScene.m_Resolution[0], gScene.m_Resolution[1], gScene.m_Resolution[2]));

	// Bind gradient magnitude buffer to texture
	Log("Copying gradient magnitude to device", "grid");
	gStatus.SetStatisticChanged("CUDA Memory", "Gradient Magnitude Buffer", QString::number(gScene.m_Resolution.GetNoElements() * sizeof(short) / MB, 'f', 2), "MB");
	BindGradientMagnitudeBuffer((short*)m_pGradientMagnitudeBuffer, make_cudaExtent(gScene.m_Resolution[0], gScene.m_Resolution[1], gScene.m_Resolution[2]));

	gStatus.SetStatisticChanged("Performance", "Timings", "");

	// Allocate CUDA memory for scene
	HandleCudaError(cudaMalloc((void**)&m_pDevScene, sizeof(CScene)));

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

			gSceneMutex.lock();
			SceneCopy = gScene;
			gSceneMutex.unlock();
//			SceneCopy.m_Camera.Update();

			gStatus.SetStatisticChanged("Camera", "Position", FormatVector(SceneCopy.m_Camera.m_From));
			gStatus.SetStatisticChanged("Camera", "Target", FormatVector(SceneCopy.m_Camera.m_Target));
			gStatus.SetStatisticChanged("Camera", "Up Vector", FormatVector(SceneCopy.m_Camera.m_Up));

			// Resizing the image canvas requires special attention
			if (SceneCopy.m_DirtyFlags.HasFlag(FilmResolutionDirty))
			{
				// Allocate host image buffer, this thread will blit it's frames to this buffer
				free(m_pRenderImage);
				m_pRenderImage = NULL;

				m_pRenderImage = (CColorRgbaLdr*)malloc(SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr));

				if (m_pRenderImage)
					memset(m_pRenderImage, 0, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr));
			
				gStatus.SetStatisticChanged("Host Memory", "LDR Frame Buffer", QString::number(3 * SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr) / MB, 'f', 2), "MB");

				m_CudaFrameBuffers.Resize(Vec2i(SceneCopy.m_Camera.m_Film.GetWidth(), SceneCopy.m_Camera.m_Film.GetHeight()));

				// Reset no. iterations
				SceneCopy.SetNoIterations(0);

				// Notify Inform others about the memory allocations
//				gStatus.SetResize();

				BindEstimateRgbLdr(m_CudaFrameBuffers.m_pDevEstRgbaLdr, SceneCopy.m_Camera.m_Film.m_Resolution.GetResX(), SceneCopy.m_Camera.m_Film.m_Resolution.GetResY());

				Log("Render canvas resized to: " + QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResX()) + " x " + QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResY()) + " pixels", "application-resize");
			}

			// Restart the rendering when when the camera, lights and render params are dirty
			if (SceneCopy.m_DirtyFlags.HasFlag(CameraDirty | LightsDirty | RenderParamsDirty | TransferFunctionDirty))
			{
				m_CudaFrameBuffers.Reset();

				// Reset no. iterations
				gScene.SetNoIterations(0);
			}

			// At this point, all dirty flags should have been taken care of, since the flags in the original scene are now cleared
			gScene.m_DirtyFlags.ClearAllFlags();

			// Increase the number of iterations performed so far
			gScene.SetNoIterations(gScene.GetNoIterations() + 1);

			// Adjust de-noising parameters
			const float Radius = 0.015 * (float)SceneCopy.GetNoIterations();

			gStatus.SetStatisticChanged("Denoise", "Radius", QString::number(Radius, 'f', 2), "ms.");

			if (Radius < 1.0f)
			{
				SceneCopy.m_DenoiseParams.m_Enabled = true;
				SceneCopy.m_DenoiseParams.SetWindowRadius(6.0f);
				SceneCopy.m_DenoiseParams.m_LerpC = Radius;
			}
			else
			{
				SceneCopy.m_DenoiseParams.m_Enabled = false;
			}
			
		//	SceneCopy.m_DenoiseParams.m_LerpC = 1.0f - Radius;

			SceneCopy.m_Camera.Update();

			HandleCudaError(cudaMemcpy(m_pDevScene, &SceneCopy, sizeof(CScene), cudaMemcpyHostToDevice));

//			msleep(10);
			
			// Execute the rendering kernels
  			Render(0, &SceneCopy, m_pDevScene, m_CudaFrameBuffers, gScene.GetNoIterations(), RenderImage, BlurImage, PostProcessImage, DenoiseImage);
			HandleCudaError(cudaGetLastError());
		
			gStatus.SetStatisticChanged("Timings", "Render Image", QString::number(RenderImage.m_FilteredDuration, 'f', 2), "ms.");
			gStatus.SetStatisticChanged("Timings", "Blur Estimate", QString::number(BlurImage.m_FilteredDuration, 'f', 2), "ms.");
			gStatus.SetStatisticChanged("Timings", "Post Process Estimate", QString::number(PostProcessImage.m_FilteredDuration, 'f', 2), "ms.");
			gStatus.SetStatisticChanged("Timings", "De-noise Image", QString::number(DenoiseImage.m_FilteredDuration, 'f', 2), "ms.");

			FPS.AddDuration(1000.0f / TmrFps.ElapsedTime());

 			gStatus.SetStatisticChanged("Performance", "FPS", QString::number(FPS.m_FilteredDuration, 'f', 2), "Frames/Sec.");
 			gStatus.SetStatisticChanged("Performance", "No. Iterations", QString::number(SceneCopy.GetNoIterations()), "Iterations");

			HandleCudaError(cudaMemcpy(m_pRenderImage, m_CudaFrameBuffers.m_pDevRgbLdrDisp, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr), cudaMemcpyDeviceToHost));

			if (m_SaveFrames.indexOf(SceneCopy.GetNoIterations()) > 0)
			{
				const QString ImageFilePath = QApplication::applicationDirPath() + "/Output/" + m_SaveBaseName + "_" + QString::number(SceneCopy.GetNoIterations()) + ".png";

				SaveImage((unsigned char*)m_pRenderImage, SceneCopy.m_Camera.m_Film.m_Resolution.GetResX(), SceneCopy.m_Camera.m_Film.m_Resolution.GetResY(), ImageFilePath);
			}
			/**/
			// Let others know we are finished with a frame
 			gStatus.SetPostRenderFrame();
		}
	}
	catch (QString* pMessage)
	{
		Log(*pMessage + ", rendering will be aborted");

		// Free render image buffer
		free(m_pRenderImage);
		m_pRenderImage = NULL;

		// Let others know that we have stopped rendering
		gStatus.SetRenderEnd();

		return;
	}

	// Free CUDA buffers
	HandleCudaError(cudaFree(m_pDevScene));
	
	m_pDevScene = NULL;

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
	gScene.m_Resolution.SetResXYZ(Vec3i(pVolumeResolution[1] + 1, pVolumeResolution[3] + 1, pVolumeResolution[5] + 1));

	Log("Resolution: " + FormatSize(gScene.m_Resolution.GetResXYZ()) + "", "grid");

	// Intensity range
	double* pIntensityRange = ImageCast->GetOutput()->GetScalarRange();
	gScene.m_IntensityRange.SetMin((float)pIntensityRange[0]);
	gScene.m_IntensityRange.SetMax((float)pIntensityRange[1]);

	Log("Intensity range: [" + QString::number(gScene.m_IntensityRange.GetMin()) + ", " + QString::number(gScene.m_IntensityRange.GetMax()) + "]", "grid");

	// Spacing
	double* pSpacing = ImageCast->GetOutput()->GetSpacing();

	gScene.m_Spacing.x = (float)pSpacing[0];
	gScene.m_Spacing.y = (float)pSpacing[1];
	gScene.m_Spacing.z = (float)pSpacing[2];

	Log("Spacing: " + FormatSize(gScene.m_Spacing, 2), "grid");

	// Compute physical size
	const Vec3f PhysicalSize(Vec3f(gScene.m_Spacing.x * (float)gScene.m_Resolution.GetResX(), gScene.m_Spacing.y * (float)gScene.m_Resolution.GetResY(), gScene.m_Spacing.z * (float)gScene.m_Resolution.GetResZ()));

	// Compute the volume's bounding box
	gScene.m_BoundingBox.m_MinP	= Vec3f(0.0f);
	gScene.m_BoundingBox.m_MaxP	= PhysicalSize / PhysicalSize.Max();

	gScene.m_GradientDelta = 1.0f / (float)gScene.m_Resolution.GetMax();
	
	Log("Bounding box: " + FormatVector(gScene.m_BoundingBox.m_MinP, 2) + " - " + FormatVector(gScene.m_BoundingBox.m_MaxP), "grid");
	
	const int DensityBufferSize = gScene.m_Resolution.GetNoElements() * sizeof(short);

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
	
	gScene.m_GradientMagnitudeRange.SetMin((float)pGradientMagnitudeRange[0]);
	gScene.m_GradientMagnitudeRange.SetMax((float)pGradientMagnitudeRange[1]);
	
	Log("Gradient magnitude range: [" + QString::number(gScene.m_GradientMagnitudeRange.GetMin(), 'f', 2) + " - " + QString::number(gScene.m_GradientMagnitudeRange.GetMax(), 'f', 2) + "]", "grid");
	
	const int GradientMagnitudeBufferSize = gScene.m_Resolution.GetNoElements() * sizeof(short);
	
	m_pGradientMagnitudeBuffer = (short*)malloc(GradientMagnitudeBufferSize);
	memcpy(m_pGradientMagnitudeBuffer, GradientMagnitudeBuffer->GetScalarPointer(), GradientMagnitudeBufferSize);

	// Build the histogram
	Log("Creating gradient magnitude histogram", "grid");

	vtkSmartPointer<vtkImageAccumulate> GradMagHistogram = vtkSmartPointer<vtkImageAccumulate>::New();

	GradMagHistogram->SetInputConnection(GradientMagnitude->GetOutputPort());
	GradMagHistogram->SetComponentExtent(0, 255, 0, 0, 0, 0);
	GradMagHistogram->SetComponentOrigin(0, 0, 0);
	GradMagHistogram->SetComponentSpacing(gScene.m_GradientMagnitudeRange.GetLength() / 256.0f, 0, 0);
//	GradMagHistogram->IgnoreZeroOn();
	GradMagHistogram->Update();

	gScene.m_GradMagMean = (float)GradMagHistogram->GetMean()[0];
	gScene.m_GradientFactor = gScene.m_GradMagMean;

	Log("Mean gradient magnitude: " + QString::number(gScene.m_GradMagMean, 'f', 2), "grid");

	Log("Creating density histogram", "grid");

	// Build the histogram
	vtkSmartPointer<vtkImageAccumulate> Histogram = vtkSmartPointer<vtkImageAccumulate>::New();

	Log("Creating histogram", "grid");

 	Histogram->SetInputConnection(ImageCast->GetOutputPort());
 	Histogram->SetComponentExtent(0, 256, 0, 0, 0, 0);
 	Histogram->SetComponentOrigin(gScene.m_IntensityRange.GetMin(), 0, 0);
 	Histogram->SetComponentSpacing(gScene.m_IntensityRange.GetLength() / 256.0f, 0, 0);
 	Histogram->IgnoreZeroOn();
 	Histogram->Update();
 
	// Update the histogram in the transfer function
	gHistogram.SetBins((int*)Histogram->GetOutput()->GetScalarPointer(), 256);
	
	gStatus.SetStatisticChanged("Volume", "File", QFileInfo(m_FileName).fileName(), "");
	gStatus.SetStatisticChanged("Volume", "Bounding Box", "", "");
	gStatus.SetStatisticChanged("Bounding Box", "Min", FormatVector(gScene.m_BoundingBox.m_MinP, 2), "m");
	gStatus.SetStatisticChanged("Bounding Box", "Max", FormatVector(gScene.m_BoundingBox.m_MaxP, 2), "m");
	gStatus.SetStatisticChanged("Volume", "Physical Size", FormatSize(PhysicalSize, 2), "mm");
	gStatus.SetStatisticChanged("Volume", "Resolution", FormatSize(gScene.m_Resolution.GetResXYZ()), "Voxels");
//	gStatus.SetStatisticChanged("Volume", "Extinction Resolution", FormatSize(Vec3i((int)gScene.m_ExtinctionSize.width, (int)gScene.m_ExtinctionSize.height, (int)gScene.m_ExtinctionSize.depth)), "Voxels");
	gStatus.SetStatisticChanged("Volume", "Spacing", FormatSize(gScene.m_Spacing, 2), "mm");
	gStatus.SetStatisticChanged("Volume", "No. Voxels", QString::number(gScene.m_Resolution.GetNoElements()), "Voxels");
	gStatus.SetStatisticChanged("Volume", "Density Range", "[" + QString::number(gScene.m_IntensityRange.GetMin()) + ", " + QString::number(gScene.m_IntensityRange.GetMax()) + "]", "");
	/**/
	
	return true;
}

void QRenderThread::OnUpdateTransferFunction(void)
{
	QMutexLocker Locker(&gSceneMutex);

	QTransferFunction TransferFunction = gTransferFunction;

	gScene.m_TransferFunctions.m_Opacity.m_NoNodes		= TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Diffuse.m_NoNodes		= TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Specular.m_NoNodes		= TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Emission.m_NoNodes		= TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Roughness.m_NoNodes	= TransferFunction.GetNodes().size();

	for (int i = 0; i < TransferFunction.GetNodes().size(); i++)
	{
		QNode& Node = TransferFunction.GetNode(i);

		const float Intensity = gScene.m_IntensityRange.GetMin() + gScene.m_IntensityRange.GetLength() * Node.GetIntensity();

		// Positions
		gScene.m_TransferFunctions.m_Opacity.m_P[i]		= Intensity;
		gScene.m_TransferFunctions.m_Diffuse.m_P[i]		= Intensity;
		gScene.m_TransferFunctions.m_Specular.m_P[i]	= Intensity;
		gScene.m_TransferFunctions.m_Emission.m_P[i]	= Intensity;
		gScene.m_TransferFunctions.m_Roughness.m_P[i]	= Intensity;

		// Colors
		gScene.m_TransferFunctions.m_Opacity.m_C[i]		= CColorRgbHdr(Node.GetOpacity());
		gScene.m_TransferFunctions.m_Diffuse.m_C[i]		= CColorRgbHdr(Node.GetDiffuse().redF(), Node.GetDiffuse().greenF(), Node.GetDiffuse().blueF());
		gScene.m_TransferFunctions.m_Specular.m_C[i]	= CColorRgbHdr(Node.GetSpecular().redF(), Node.GetSpecular().greenF(), Node.GetSpecular().blueF());
		gScene.m_TransferFunctions.m_Emission.m_C[i]	= 500.0f * CColorRgbHdr(Node.GetEmission().redF(), Node.GetEmission().greenF(), Node.GetEmission().blueF());

		const float Roughness = 1.0f - expf(-Node.GetGlossiness());

		gScene.m_TransferFunctions.m_Roughness.m_C[i]	= CColorRgbHdr(Roughness * 100);
	}

	gScene.m_DensityScale	= TransferFunction.GetDensityScale();
	gScene.m_ShadingType	= TransferFunction.GetShadingType();

	gScene.m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void QRenderThread::OnUpdateCamera(void)
{
	QMutexLocker Locker(&gSceneMutex);

	gScene.m_Camera.m_Film.m_Exposure = (1.001f - gCamera.GetFilm().GetExposure()) * 1000.0f;

	if (gCamera.GetFilm().IsDirty())
	{
		const int FilmWidth	= gCamera.GetFilm().GetWidth();
		const int FilmHeight = gCamera.GetFilm().GetHeight();

 		gScene.m_Camera.m_Film.m_Resolution.SetResX(FilmWidth);
		gScene.m_Camera.m_Film.m_Resolution.SetResY(FilmHeight);
		gScene.m_Camera.Update();
		gCamera.GetFilm().UnDirty();
// 		// 
 		gScene.m_DirtyFlags.SetFlag(FilmResolutionDirty);
	}

// 	gScene.m_Camera.m_From	= gCamera.GetFrom();
// 	gScene.m_Camera.m_Target	= gCamera.GetTarget();
// 	gScene.m_Camera.m_Up		= gCamera.GetUp();

	gScene.m_Camera.Update();

	// Aperture
	gScene.m_Camera.m_Aperture.m_Size	= gCamera.GetAperture().GetSize();

	// Projection
	gScene.m_Camera.m_FovV = gCamera.GetProjection().GetFieldOfView();

	// Focus
	gScene.m_Camera.m_Focus.m_FocalDistance = gCamera.GetFocus().GetFocalDistance();

	gScene.m_DirtyFlags.SetFlag(CameraDirty);
}

void QRenderThread::OnUpdateLighting(void)
{
	QMutexLocker Locker(&gSceneMutex);

	gScene.m_Lighting.Reset();

	if (gLighting.Background().GetEnabled())
	{
		CLight BackgroundLight;

		BackgroundLight.m_T	= 1;

		BackgroundLight.m_ColorTop		= gLighting.Background().GetIntensity() * CColorRgbHdr(gLighting.Background().GetTopColor().redF(), gLighting.Background().GetTopColor().greenF(), gLighting.Background().GetTopColor().blueF());
		BackgroundLight.m_ColorMiddle	= gLighting.Background().GetIntensity() * CColorRgbHdr(gLighting.Background().GetMiddleColor().redF(), gLighting.Background().GetMiddleColor().greenF(), gLighting.Background().GetMiddleColor().blueF());
		BackgroundLight.m_ColorBottom	= gLighting.Background().GetIntensity() * CColorRgbHdr(gLighting.Background().GetBottomColor().redF(), gLighting.Background().GetBottomColor().greenF(), gLighting.Background().GetBottomColor().blueF());

		BackgroundLight.Update(gScene.m_BoundingBox);

		gScene.m_Lighting.AddLight(BackgroundLight);
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

		AreaLight.Update(gScene.m_BoundingBox);

		gScene.m_Lighting.AddLight(AreaLight);
	}

	gScene.m_DirtyFlags.SetFlag(LightsDirty);
}

CColorRgbaLdr* QRenderThread::GetRenderImage(void) const
{
	return m_pRenderImage;
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

