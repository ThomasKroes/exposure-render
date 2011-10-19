/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "Lighting.h"

QLighting gLighting;

QLighting::QLighting(QObject* pParent /*= NULL*/) :
	QPresetXML(pParent),
	m_Lights(),
	m_pSelectedLight(NULL),
	m_Background()
{
}

QLighting::~QLighting(void)
{
}

QLighting::QLighting(const QLighting& Other)
{
	*this = Other;
}

QLighting& QLighting::operator=(const QLighting& Other)
{
	// Clear light selection, do not remove this line!
	SetSelectedLight((QLight*)NULL);

	blockSignals(true);

	QPresetXML::operator=(Other);

	QObject::disconnect(this, SLOT(OnLightPropertiesChanged(QLight*)));

	m_Lights		= Other.m_Lights;
	
	blockSignals(false);

	m_Background = Other.m_Background;

	emit Changed();

	for (int i = 0; i < m_Lights.size(); i++)
	{
		QObject::connect(&m_Lights[i], SIGNAL(LightPropertiesChanged(QLight*)), this, SLOT(OnLightPropertiesChanged(QLight*)));
	}

	return *this;
}

void QLighting::OnLightPropertiesChanged(QLight* pLight)
{
	emit Changed();
}

void QLighting::OnBackgroundChanged(void)
{
	emit Changed();
}

void QLighting::AddLight(QLight& Light)
{
	// Add to list
	m_Lights.append(Light);

	// Select
	SetSelectedLight(&m_Lights.back());

	// Connect
	connect(&m_Lights.back(), SIGNAL(LightPropertiesChanged(QLight*)), this, SLOT(OnLightPropertiesChanged(QLight*)));

	Log("'" + Light.GetName() + "' added to the scene", "light-bulb");

	// Let others know the lighting has changed
	emit Changed();
}

void QLighting::RemoveLight(QLight* pLight)
{
	if (!pLight)
		return;

	Log("'" + pLight->GetName() + "' removed from the scene", "light-bulb");

	// Remove from light list
	m_Lights.remove(*pLight);

	m_pSelectedLight = NULL;

	// Deselect
	SetSelectedLight(NULL);

	// Let others know the lighting has changed
	emit Changed();
}

void QLighting::RemoveLight(const int& Index)
{
	if (Index < 0 || Index >= m_Lights.size())
		return;

	RemoveLight(&m_Lights[Index]);
}

QBackground& QLighting::Background(void)
{
	return m_Background;
}

QLightList& QLighting::GetLights(void)
{
	return m_Lights;
}

void QLighting::SetSelectedLight(QLight* pSelectedLight)
{
	m_pSelectedLight = pSelectedLight;
	emit LightSelectionChanged(pSelectedLight);
}

void QLighting::SetSelectedLight(const int& Index)
{
	if (m_Lights.size() <= 0)
	{
		SetSelectedLight((QLight*)NULL);
	}
	else
	{
		// Compute new index
		const int NewIndex = qMin(m_Lights.size() - 1, qMax(0, Index));

		if (GetSelectedLight() && m_Lights.indexOf(*GetSelectedLight()) == NewIndex)
			return;

		// Set selected node
		SetSelectedLight(&m_Lights[NewIndex]);
	}
}

QLight* QLighting::GetSelectedLight(void)
{
	return m_pSelectedLight;
}

void QLighting::SelectPreviousLight(void)
{
	if (!m_pSelectedLight)
		return;

	int Index = m_Lights.indexOf(*GetSelectedLight());

	if (Index < 0)
		return;

	// Compute new index
	const int NewIndex = qMin(m_Lights.size() - 1, qMax(0, Index - 1));

	// Set selected node
	SetSelectedLight(&m_Lights[NewIndex]);
}

void QLighting::SelectNextLight(void)
{
	if (!m_pSelectedLight)
		return;

	int Index = m_Lights.indexOf(*GetSelectedLight());

	if (Index < 0)
		return;

	// Compute new index
	const int NewIndex = qMin(m_Lights.size() - 1, qMax(0, Index + 1));

	// Set selected node
	SetSelectedLight(&m_Lights[NewIndex]);
}

void QLighting::CopyLight(QLight* pLight)
{
	if (!pLight)
		return;

	QLight LightCopy = *pLight;

	// Rename
	LightCopy.SetName("Copy of " + pLight->GetName());

	// Add
	AddLight(LightCopy);

	// Let others know the lighting has changed
	emit Changed();
}

void QLighting::CopySelectedLight(void)
{
	CopyLight(m_pSelectedLight);
}

void QLighting::RenameLight(const int& Index, const QString& Name)
{
	if (Index < 0 || Index >= m_Lights.size() || Name.isEmpty())
		return;

	Log("'" + m_Lights[Index].GetName() + " renamed to '" + Name + "'", "light-bulb");

	m_Lights[Index].SetName(Name);

	// Let others know the lighting has changed
	emit Changed();
}

void QLighting::ReadXML(QDomElement& Parent)
{
	SetSelectedLight(NULL);

	QPresetXML::ReadXML(Parent);

	QDomElement Lights = Parent.firstChild().toElement();

	// Read child nodes
	for (QDomNode DomNode = Lights.firstChild(); !DomNode.isNull(); DomNode = DomNode.nextSibling())
	{
		// Create new light preset
		QLight LightPreset(this);

		m_Lights.append(LightPreset);

		// Load preset into it
		m_Lights.back().ReadXML(DomNode.toElement());
	}

	QDomElement Background = Parent.firstChildElement("Background").toElement();
	m_Background.ReadXML(Background);

	SetSelectedLight(0);
}

QDomElement QLighting::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Preset
	QDomElement Preset = DOM.createElement("Preset");
	Parent.appendChild(Preset);

	QPresetXML::WriteXML(DOM, Preset);

	QDomElement Lights = DOM.createElement("Lights");
	Preset.appendChild(Lights);

	for (int i = 0; i < m_Lights.size(); i++)
		m_Lights[i].WriteXML(DOM, Lights);

	m_Background.WriteXML(DOM, Preset);

	return Preset;
}

QLighting QLighting::Default(void)
{
	QLighting DefaultLighting;

	DefaultLighting.SetName("Default");

	DefaultLighting.AddLight(QLight::Default());

	DefaultLighting.Background().SetTopColor(QColor(85, 170, 255));
	DefaultLighting.Background().SetMiddleColor(QColor(255, 170, 127));
	DefaultLighting.Background().SetBottomColor(QColor(63, 31, 0));

	DefaultLighting.Background().SetIntensity(2.0f);

	return DefaultLighting;
}