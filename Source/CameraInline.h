#pragma once

#include "Geometry.h"
#include "RNG.h"
#include "Light.h"
#include "Flags.h"

class EXPOSURE_RENDER_DLL CCameraSample
{
public:
	Vec2f	m_ImageXY;
	Vec2f	m_LensUV;

	DEV CCameraSample(void)
	{
		m_ImageXY	= Vec2f(0.0f);
		m_LensUV	= Vec2f(0.0f);
	}

	DEV CCameraSample& CCameraSample::operator=(const CCameraSample& Other)
	{
		m_ImageXY	= Other.m_ImageXY;
		m_LensUV	= Other.m_LensUV;

		return *this;
	}

	DEV void LargeStep(Vec2f& ImageUV, Vec2f& LensUV, const int& X, const int& Y, const int& KernelSize)
	{
		m_ImageXY	= Vec2f(X + ImageUV.x, Y + ImageUV.y);
		m_LensUV	= LensUV;
	}
};

#define DEF_FOCUS_TYPE					CenterScreen
#define DEF_FOCUS_SENSOR_POS_CANVAS		Vec2f(0.0f)
#define DEF_FOCUS_P						Vec3f(0.0f)
#define DEF_FOCUS_FOCAL_DISTANCE		100.0f
#define	DEF_FOCUS_T						0.0f
#define DEF_FOCUS_N						Vec3f(0.0f)
#define DEF_FOCUS_DOT_WN				0.0f

class EXPOSURE_RENDER_DLL CFocus
{
public:
	enum EType
	{
		CenterScreen,
		ScreenPoint,
		Probed,
		Manual
	};

	EType		m_Type;
	Vec2f		m_SensorPosCanvas;
	float		m_FocalDistance;
	float		m_T;
	Vec3f		m_P;
	Vec3f		m_N;
	float		m_DotWN;

	HOD CFocus(void)
	{
		m_Type				= DEF_FOCUS_TYPE;
		m_SensorPosCanvas	= DEF_FOCUS_SENSOR_POS_CANVAS;
		m_FocalDistance		= DEF_FOCUS_FOCAL_DISTANCE;
		m_T					= DEF_FOCUS_T;
		m_P					= DEF_FOCUS_P;
		m_N					= DEF_FOCUS_N;
		m_DotWN				= DEF_FOCUS_DOT_WN;
	}

	HOD CFocus& CFocus::operator=(const CFocus& Other)
	{
		m_Type				= Other.m_Type;
		m_SensorPosCanvas	= Other.m_SensorPosCanvas;
		m_FocalDistance		= Other.m_FocalDistance;
		m_P					= Other.m_P;
		m_T					= Other.m_T;
		m_N					= Other.m_N;
		m_DotWN				= Other.m_DotWN;

		return *this;
	}
};

#define DEF_APERTURE_SIZE			0.0f
#define DEF_APERTURE_NO_BLADES		5
#define DEF_APERTURE_BIAS			BiasNone
#define DEF_APERTURE_ROTATION		0.0f

class EXPOSURE_RENDER_DLL CAperture
{
public:
	enum EBias
	{
		BiasCenter,
		BiasEdge,
		BiasNone
	};

	float			m_Size;
	int				m_NoBlades;
	EBias			m_Bias;
	float			m_Rotation;
	float			m_Data[MAX_BOKEH_DATA];

	HOD CAperture(void)
	{
		m_Size		= DEF_APERTURE_SIZE;
		m_NoBlades	= DEF_APERTURE_NO_BLADES;
		m_Bias		= DEF_APERTURE_BIAS;
		m_Rotation	= DEF_APERTURE_ROTATION;

		for (int i = 0; i < MAX_BOKEH_DATA; i++)
			m_Data[i] = 0.0f;
	}

	CAperture& CAperture::operator=(const CAperture& Other)
	{
		m_Size		= Other.m_Size;
		m_NoBlades	= Other.m_NoBlades;
		m_Bias		= Other.m_Bias;
		m_Rotation	= Other.m_Rotation;

		for (int i = 0; i < MAX_BOKEH_DATA; i++)
			m_Data[i] = Other.m_Data[i];

		return *this;
	}

	HOD void Update(const float& FStop)
	{
		// Update bokeh
		int Ns = (int)m_NoBlades;

		if ((Ns >= 3) && (Ns <= 6))
		{
			float w = m_Rotation * PI_F / 180.0f, wi = (2.0f * PI_F) / (float)Ns;

			Ns = (Ns + 2) * 2;

			for (int i = 0; i < Ns; i += 2)
			{
				m_Data[i]		= cos(w);
				m_Data[i + 1]	= sin(w);
				w += wi;
			}
		}
	}
};

class EXPOSURE_RENDER_DLL CFilm
{
public:
	CResolution2D	m_Resolution;
	float			m_Screen[2][2];
	Vec2f			m_InvScreen;
	float			m_Iso;
	float			m_Exposure;
	float			m_FStop;
	float			m_Gamma;

	// ToDo: Add description
	HOD CFilm(void)
	{
		m_Screen[0][0]	= 0.0f;
		m_Screen[0][1]	= 0.0f;
		m_Screen[1][0]	= 0.0f;
		m_Screen[1][1]	= 0.0f;
		m_InvScreen		= Vec2f(0.0f);
		m_Iso			= 400.0f;
		m_Exposure		= 10.0f;
		m_FStop			= 8.0f;
		m_Gamma			= 2.2f;
	}

	CFilm& CFilm::operator=(const CFilm& Other)
	{
		m_Resolution		= Other.m_Resolution;
		m_Screen[0][0]		= Other.m_Screen[0][0];
		m_Screen[0][1]		= Other.m_Screen[0][1];
		m_Screen[1][0]		= Other.m_Screen[1][0];
		m_Screen[1][1]		= Other.m_Screen[1][1];
		m_InvScreen			= Other.m_InvScreen;
		m_Iso				= Other.m_Iso;
		m_Exposure			= Other.m_Exposure;
		m_FStop				= Other.m_FStop;
		m_Gamma				= Other.m_Gamma;

		return *this;
	}

	HOD void Update(const float& FovV, const float& Aperture)
	{
		float Scale = 0.0f;

		Scale = tanf(0.5f * (FovV / RAD_F));

		if (m_Resolution.GetAspectRatio() > 1.0f)
		{
			m_Screen[0][0] = -Scale;
			m_Screen[0][1] = Scale;
			m_Screen[1][0] = -Scale * m_Resolution.GetAspectRatio();
			m_Screen[1][1] = Scale * m_Resolution.GetAspectRatio();
		}
		else
		{
			m_Screen[0][0] = -Scale / m_Resolution.GetAspectRatio();
			m_Screen[0][1] = Scale / m_Resolution.GetAspectRatio();
			m_Screen[1][0] = -Scale;
			m_Screen[1][1] = Scale;
		}

		m_InvScreen.x = (m_Screen[0][1] - m_Screen[0][0]) / m_Resolution.GetResX();
		m_InvScreen.y = (m_Screen[1][1] - m_Screen[1][0]) / m_Resolution.GetResY();

		m_Resolution.Update();
	}

	HOD int GetWidth(void) const
	{
		return m_Resolution.GetResX();
	}

	HOD int GetHeight(void) const
	{
		return m_Resolution.GetResY();
	}
};

#define FPS1 30.0f

#define DEF_CAMERA_TYPE						Perspective
#define DEF_CAMERA_OPERATOR					CameraOperatorUndefined
#define DEF_CAMERA_VIEW_MODE				ViewModeBack
#define DEF_CAMERA_HITHER					1.0f
#define DEF_CAMERA_YON						50000.0f
#define DEF_CAMERA_ENABLE_CLIPPING			true
#define DEF_CAMERA_GAMMA					2.2f
#define DEF_CAMERA_FIELD_OF_VIEW			55.0f
#define DEF_CAMERA_NUM_APERTURE_BLADES		4
#define DEF_CAMERA_APERTURE_BLADES_ANGLE	0.0f
#define DEF_CAMERA_ASPECT_RATIO				1.0f
#define DEF_CAMERA_ZOOM_SPEED				1.0f
#define DEF_CAMERA_ORBIT_SPEED				5.0f
#define DEF_CAMERA_APERTURE_SPEED			0.25f
#define DEF_CAMERA_FOCAL_DISTANCE_SPEED		10.0f

class EXPOSURE_RENDER_DLL CCamera 
{
public:
	CBoundingBox		m_SceneBoundingBox;
	float				m_Hither;
	float				m_Yon;
	bool				m_EnableClippingPlanes;
	Vec3f				m_From;
	Vec3f				m_Target;
	Vec3f				m_Up;
	float				m_FovV;
	float				m_AreaPixel;
	Vec3f 				m_N;
	Vec3f 				m_U;
	Vec3f 				m_V;
	CFilm				m_Film;
	CFocus				m_Focus;
	CAperture			m_Aperture;
	bool				m_Dirty;

	HOD CCamera(void)
	{
		m_Hither				= DEF_CAMERA_HITHER;
		m_Yon					= DEF_CAMERA_YON;
		m_EnableClippingPlanes	= DEF_CAMERA_ENABLE_CLIPPING;
		m_From					= Vec3f(500.0f, 500.0f, 500.0f);
		m_Target				= Vec3f(0.0f, 0.0f, 0.0f);
		m_Up					= Vec3f(0.0f, 1.0f, 0.0f);
		m_FovV					= DEF_CAMERA_FIELD_OF_VIEW;
		m_N						= Vec3f(0.0f, 0.0f, 1.0f);
		m_U						= Vec3f(1.0f, 0.0f, 0.0f);
		m_V						= Vec3f(0.0f, 1.0f, 0.0f);
		m_Dirty					= true;
	}

	CCamera& CCamera::operator=(const CCamera& Other)
	{
		m_SceneBoundingBox		= Other.m_SceneBoundingBox;
		m_Hither				= Other.m_Hither;
		m_Yon					= Other.m_Yon;
		m_EnableClippingPlanes	= Other.m_EnableClippingPlanes;
		m_From					= Other.m_From;
		m_Target				= Other.m_Target;
		m_Up					= Other.m_Up;
		m_FovV					= Other.m_FovV;
		m_AreaPixel				= Other.m_AreaPixel;
		m_N						= Other.m_N;
		m_U						= Other.m_U;
		m_V						= Other.m_V;
		m_Film					= Other.m_Film;
		m_Focus					= Other.m_Focus;
		m_Aperture				= Other.m_Aperture;
		m_Dirty					= Other.m_Dirty;

		return *this;
	}

	HOD void Update(void)
	{
		m_N	= Normalize(m_Target - m_From);
		m_U	= Normalize(Cross(m_Up, m_N));
		m_V	= Normalize(Cross(m_N, m_U));

		m_Film.Update(m_FovV, m_Aperture.m_Size);

		m_AreaPixel = m_Film.m_Resolution.GetAspectRatio() / (m_Focus.m_FocalDistance * m_Focus.m_FocalDistance);

		m_Aperture.Update(m_Film.m_FStop);

		m_Film.Update(m_FovV, m_Aperture.m_Size);
	}

	HO void Zoom(float amount)
	{
		Vec3f reverseLoS = m_From - m_Target;

		if (amount > 0)
		{	
			reverseLoS.ScaleBy(1.1f);
		}
		else if (amount < 0)
		{	
			if (reverseLoS.Length() > 0.0005f)
			{ 
				reverseLoS.ScaleBy(0.9f);
			}
		}

		m_From = reverseLoS + m_Target;
	}

	// Pan operator
	HO void Pan(float DownDegrees, float RightDegrees)
	{
		Vec3f LoS = m_Target - m_From;

		Vec3f right		= LoS.Cross(m_Up);
		Vec3f orthogUp	= LoS.Cross(right);

		right.Normalize();
		orthogUp.Normalize();

		const float Length = (m_Target - m_From).Length();

		const unsigned int WindowWidth	= m_Film.m_Resolution.GetResX();

		const float U = Length * (RightDegrees / WindowWidth);
		const float V = Length * (DownDegrees / WindowWidth);

		m_From		= m_From + right * U - m_Up * V;
		m_Target	= m_Target + right * U - m_Up * V;
	}

	HO void Orbit(float DownDegrees, float RightDegrees)
	{
		Vec3f ReverseLoS = m_From - m_Target;

		Vec3f right		= m_Up.Cross(ReverseLoS);
		Vec3f orthogUp	= ReverseLoS.Cross(right);
		Vec3f Up = Vec3f(0.0f, 1.0f, 0.0f);
		
		ReverseLoS.RotateAxis(right, DownDegrees);
		ReverseLoS.RotateAxis(Up, RightDegrees);
		m_Up.RotateAxis(right, DownDegrees);
		m_Up.RotateAxis(Up, RightDegrees);

		m_From = ReverseLoS + m_Target;
	}

	HO void SetViewMode(const EViewMode ViewMode)
	{
		if (ViewMode == ViewModeUser)
			return;

		m_Target	= m_SceneBoundingBox.GetCenter();
		m_Up		= Vec3f(0.0f, 1.0f, 0.0f);

		const float Distance = 1.5f;

		const float Length = Distance * m_SceneBoundingBox.GetMaxLength();

		m_From = m_Target;

		switch (ViewMode)
		{
		case ViewModeFront:							m_From.z -= Length;												break;
		case ViewModeBack:							m_From.z += Length;												break;
		case ViewModeLeft:							m_From.x += Length;												break;
		case ViewModeRight:							m_From.x -= -Length;											break;
		case ViewModeTop:							m_From.y += Length;		m_Up = Vec3f(0.0f, 0.0f, 1.0f);			break;
		case ViewModeBottom:						m_From.y -= -Length;	m_Up = Vec3f(0.0f, 0.0f, -1.0f);		break;
		case ViewModeIsometricFrontLeftTop:			m_From = Vec3f(Length, Length, -Length);						break;
		case ViewModeIsometricFrontRightTop:		m_From = m_Target + Vec3f(-Length, Length, -Length);			break;
		case ViewModeIsometricFrontLeftBottom:		m_From = m_Target + Vec3f(Length, -Length, -Length);			break;
		case ViewModeIsometricFrontRightBottom:		m_From = m_Target + Vec3f(-Length, -Length, -Length);			break;
		case ViewModeIsometricBackLeftTop:			m_From = m_Target + Vec3f(Length, Length, Length);				break;
		case ViewModeIsometricBackRightTop:			m_From = m_Target + Vec3f(-Length, Length, Length);				break;
		case ViewModeIsometricBackLeftBottom:		m_From = m_Target + Vec3f(Length, -Length, Length);				break;
		case ViewModeIsometricBackRightBottom:		m_From = m_Target + Vec3f(-Length, -Length, Length);			break;
		}

		Update();
	}

	HOD void GenerateRay(const Vec2f& Pixel, const Vec2f& ApertureRnd, Vec3f& RayO, Vec3f& RayD)
	{
		Vec2f ScreenPoint;

		ScreenPoint.x = m_Film.m_Screen[0][0] + (m_Film.m_InvScreen.x * Pixel.x);
		ScreenPoint.y = m_Film.m_Screen[1][0] + (m_Film.m_InvScreen.y * Pixel.y);

		RayO	= m_From;
		RayD	= Normalize(m_N + (-ScreenPoint.x * m_U) + (-ScreenPoint.y * m_V));

		if (m_Aperture.m_Size != 0.0f)
		{
			Vec2f LensUV = m_Aperture.m_Size * ConcentricSampleDisk(ApertureRnd);

			Vec3f LI = m_U * LensUV.x + m_V * LensUV.y;
			RayO += LI;
			RayD = Normalize((RayD * m_Focus.m_FocalDistance) - LI);
		}
	}
};