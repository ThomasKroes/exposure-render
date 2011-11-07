/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Preset.h"

class QLight : public QPresetXML
{
	Q_OBJECT

public:
	QLight(QObject* pParent = NULL);
	virtual ~QLight(void);

	QLight::QLight(const QLight& Other);
	
	QLight& QLight::operator=(const QLight& Other);

	bool operator == (const QLight& Other) const;

	float			GetTheta(void) const;
	void			SetTheta(const float& Theta);
	float			GetPhi(void) const;
	void			SetPhi(const float& Phi);
	float			GetWidth(void) const;
	void			SetWidth(const float& Width);
	float			GetHeight(void) const;
	void			SetHeight(const float& Height);
	bool			GetLockSize(void) const;
	void			SetLockSize(const bool& LockSize);
	float			GetDistance(void) const;
	void			SetDistance(const float& Distance);
	QColor			GetColor(void) const;
	void			SetColor(const QColor& Color);
	float			GetIntensity(void) const;
	void			SetIntensity(const float& Intensity);
	void			ReadXML(QDomElement& Parent);
	QDomElement		WriteXML(QDomDocument& DOM, QDomElement& Parent);

	static QLight	Default(void);

signals:
	void LightPropertiesChanged(QLight*);
	void ThetaChanged(QLight*);
	void PhiChanged(QLight*);
	void DistanceChanged(QLight*);
	void WidthChanged(QLight*);
	void HeightChanged(QLight*);
	void LockSizeChanged(QLight*);
	void ColorChanged(QLight*);
	void IntensityChanged(QLight*);

protected:
	float		m_Theta;
	float		m_Phi;
	float		m_Distance;
	float		m_Width;
	float		m_Height;
	bool		m_LockSize;
	QColor		m_Color;
	float		m_Intensity;
	
	friend class QLightItem;
};

typedef QList<QLight> QLightList;