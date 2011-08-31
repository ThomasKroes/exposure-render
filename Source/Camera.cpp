
#include "Camera.h"

HOD void CCameraAnimation::Sample(CCamera& Camera)
{
	switch (m_Type)
	{
		case TurnTable:
		{
			const float Theta = (m_InitialAngle / RAD_F) + (m_DeltaTheta * (float)m_CurrentFrame);

			Camera.m_From	= Camera.m_Target + Vec3f((cosf(m_Latitude / RAD_F) * m_Distance) * cosf(Theta), (sinf(m_Latitude / RAD_F) * m_Distance) * sinf(m_Latitude / RAD_F), (cosf(m_Latitude / RAD_F) * m_Distance) * sinf(Theta));
			Camera.m_Up		= Vec3f(0.0f, 1.0f, 0.0f);

			break;
		}

		default:
		{
			break;
		}
	}
}
