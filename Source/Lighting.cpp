
#include "Lighting.h"
#include "Scene.h"

QLighting gLighting;

QLight::QLight(QObject* pParent) :
	QPresetXML(pParent),
	m_Theta(),
	m_Phi(),
	m_Distance(),
	m_Width(),
	m_Height(),
	m_LockSize(),
	m_Color(),
	m_Intensity()
{
}

float QLight::GetTheta(void) const
{
	return m_Theta;
}

void QLight::SetTheta(const float& Theta)
{
	m_Theta = Theta;

	emit ThetaChanged(this);
	emit LightPropertiesChanged(this);
}

float QLight::GetPhi(void) const
{
	return m_Phi;
}

void QLight::SetPhi(const float& Phi)
{
	m_Phi = Phi;

	emit PhiChanged(this);
	emit LightPropertiesChanged(this);
}

float QLight::GetWidth(void) const
{
	return m_Width;
}

void QLight::SetWidth(const float& Width)
{
	m_Width = Width;

	emit WidthChanged(this);
	emit LightPropertiesChanged(this);
}

float QLight::GetHeight(void) const
{
	return m_Height;
}

void QLight::SetHeight(const float& Height)
{
	m_Height = Height;

	emit HeightChanged(this);
	emit LightPropertiesChanged(this);
}

bool QLight::GetLockSize(void) const
{
	return m_LockSize;
}

void QLight::SetLockSize(const bool& LockSize)
{
	m_LockSize = LockSize;
}

float QLight::GetDistance(void) const
{
	return m_Distance;
}

void QLight::SetDistance(const float& Distance)
{
	m_Distance = Distance;

	emit DistanceChanged(this);
	emit LightPropertiesChanged(this);
}

QColor QLight::GetColor(void) const
{
	return m_Color;
}

void QLight::SetColor(const QColor& Color)
{
	m_Color = Color;

	emit ColorChanged(this);
	emit LightPropertiesChanged(this);
}

float QLight::GetIntensity(void) const
{
	return m_Intensity;
}

void QLight::SetIntensity(const float& Intensity)
{
	m_Intensity = Intensity;

	emit IntensityChanged(this);
	emit LightPropertiesChanged(this);
}

void QLight::ReadXML(QDomElement& Parent)
{
	QPresetXML::ReadXML(Parent);

	m_Theta		= Parent.firstChildElement("Theta").attribute("Value").toFloat();
	m_Phi		= Parent.firstChildElement("Phi").attribute("Value").toFloat();
	m_Distance	= Parent.firstChildElement("Distance").attribute("Value").toFloat();
	m_Width		= Parent.firstChildElement("Width").attribute("Value").toFloat();
	m_Height	= Parent.firstChildElement("Height").attribute("Value").toFloat();
	m_LockSize	= Parent.firstChildElement("LockSize").attribute("Value").toInt();

	QDomElement Color = Parent.firstChildElement("Color");

	m_Color.setRed(Color.attribute("R").toInt());
	m_Color.setGreen(Color.attribute("G").toInt());
	m_Color.setBlue(Color.attribute("B").toInt());

	m_Intensity = Parent.firstChildElement("Intensity").attribute("Value").toFloat();
}

QDomElement QLight::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Light
	QDomElement Light = DOM.createElement("Light");
	Parent.appendChild(Light);

	QPresetXML::WriteXML(DOM, Light);

	// Theta
	QDomElement Theta = DOM.createElement("Theta");
	Theta.setAttribute("Value", m_Theta);
	Light.appendChild(Theta);

	// Phi
	QDomElement Phi = DOM.createElement("Phi");
	Phi.setAttribute("Value", m_Phi);
	Light.appendChild(Phi);

	// Distance
	QDomElement Distance = DOM.createElement("Distance");
	Distance.setAttribute("Value", m_Distance);
	Light.appendChild(Distance);

	// Width
	QDomElement Width = DOM.createElement("Width");
	Width.setAttribute("Value", m_Width);
	Light.appendChild(Width);

	// Height
	QDomElement Height = DOM.createElement("Height");
	Height.setAttribute("Value", m_Height);
	Light.appendChild(Height);

	// LockSize
	QDomElement LockSize = DOM.createElement("LockSize");
	LockSize.setAttribute("Value", m_LockSize);
	Light.appendChild(LockSize);

	// Color
	QDomElement Color = DOM.createElement("Color");
	Color.setAttribute("R", m_Color.red());
	Color.setAttribute("G", m_Color.green());
	Color.setAttribute("B", m_Color.blue());
	Light.appendChild(Color);

	// Intensity
	QDomElement Intensity = DOM.createElement("Intensity");
	Intensity.setAttribute("Value", m_Intensity);
	Light.appendChild(Intensity);

	return Light;
}

QBackground::QBackground(QObject* pParent) :
	QPresetXML(pParent),
	m_Enable(true),
	m_Color(Qt::white),
	m_Intensity(100.0),
	m_UseTexture(false),
	m_File("")
{
}

bool QBackground::GetEnabled(void) const
{
	return m_Enable;
}

void QBackground::SetEnabled(const bool& Enable)
{
	m_Enable = Enable;

	emit BackgroundChanged();
}

QColor QBackground::GetColor(void) const
{
	return m_Color;
}

void QBackground::SetColor(const QColor& Color)
{
	m_Color = Color;

	emit BackgroundChanged();
}

float QBackground::GetIntensity(void) const
{
	return m_Intensity;
}

void QBackground::SetIntensity(const float& Intensity)
{
	m_Intensity = Intensity;

	emit BackgroundChanged();
}

bool QBackground::GetUseTexture(void) const
{
	return m_UseTexture;
}

void QBackground::SetUseTexture(const bool& Texture)
{
	m_UseTexture = Texture;

	emit BackgroundChanged();
}

QString QBackground::GetFile(void) const
{
	return m_File;
}

void QBackground::SetFile(const QString& File)
{
	m_File = File;

	emit BackgroundChanged();
}

void QBackground::ReadXML(QDomElement& Parent)
{
	QPresetXML::ReadXML(Parent);

	SetEnabled(Parent.firstChildElement("Enable").attribute("Value").toInt());
	
	QDomElement Color = Parent.firstChildElement("Color");

	SetColor(QColor(Color.attribute("R").toInt(), Color.attribute("G").toInt(), Color.attribute("B").toInt()));

	SetIntensity(Parent.firstChildElement("Intensity").attribute("Value").toFloat());
	SetUseTexture(Parent.firstChildElement("UseTexture").attribute("Value").toInt());
	SetFile(Parent.firstChildElement("Height").attribute("Value"));
}

QDomElement QBackground::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Background
	QDomElement Background = DOM.createElement("Background");
	Parent.appendChild(Background);

//	QPresetXML::WriteXML(DOM, Background);

	// Enable
	QDomElement Enable = DOM.createElement("Enable");
	Enable.setAttribute("Value", m_Enable);
	Background.appendChild(Enable);

	// Color
	QDomElement Color = DOM.createElement("Color");
	Color.setAttribute("R", m_Color.red());
	Color.setAttribute("G", m_Color.green());
	Color.setAttribute("B", m_Color.blue());
	Background.appendChild(Color);

	// Intensity
	QDomElement Intensity = DOM.createElement("Intensity");
	Intensity.setAttribute("Value", m_Intensity);
	Background.appendChild(Intensity);

	// Use texture
	QDomElement UseTexture = DOM.createElement("UseTexture");
	UseTexture.setAttribute("Value", m_UseTexture);
	Background.appendChild(UseTexture);

	// File
	QDomElement File = DOM.createElement("File");
	File.setAttribute("Value", m_File);
	Background.appendChild(File);

	return Background;
}

QLighting::QLighting(QObject* pParent /*= NULL*/) :
	QPresetXML(pParent),
	m_Lights(),
	m_pSelectedLight(NULL),
	m_Background()
{
	connect(&m_Background, SIGNAL(BackgroundChanged()), this, SLOT(Update()));
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

void QLighting::OnLightPropertiesChanged(QLight* pLight)
{
	Update();
}

void QLighting::Update(void)
{
	if (gpScene == NULL)
		return;

	gpScene->m_Lighting.Reset();

	if (Background().GetEnabled())
	{
		CLight BackgroundLight;

		BackgroundLight.m_Type	= CLight::Background;
		BackgroundLight.m_Color	= CColorRgbHdr(Background().GetColor().redF(), Background().GetColor().greenF(), Background().GetColor().blueF());

		gpScene->m_Lighting.AddLight(BackgroundLight);
	}

	for (int i = 0; i < m_Lights.size(); i++)
	{
		QLight& Light = m_Lights[i];

		CLight AreaLight;

		AreaLight.m_Type		= CLight::Area;
		AreaLight.m_Theta		= Light.GetTheta();
		AreaLight.m_Phi			= Light.GetPhi();
		AreaLight.m_Width		= Light.GetWidth();
		AreaLight.m_Height		= Light.GetHeight();
		AreaLight.m_Distance	= Light.GetDistance();
		AreaLight.m_Color		= CColorRgbHdr(Light.GetColor().redF(), Light.GetColor().greenF(), Light.GetColor().blueF());

		AreaLight.Update();

		gpScene->m_Lighting.AddLight(AreaLight);
	}

	gpScene->m_DirtyFlags.SetFlag(LightsDirty);
}

void QLighting::AddLight(QLight& Light)
{
	m_Lights.append(Light);

	connect(&m_Lights.back(), SIGNAL(LightPropertiesChanged(QLight*)), this, SLOT(OnLightPropertiesChanged(QLight*)));
}

QBackground& QLighting::Background(void)
{
	return m_Background;
}

void QLighting::SetSelectedLight(QLight* pSelectedLight)
{
	QLight* pOldLight = m_pSelectedLight;
	m_pSelectedLight = pSelectedLight;
	emit LightSelectionChanged(pOldLight, m_pSelectedLight);
}

void QLighting::SetSelectedLight(const int& Index)
{
	QLight* pOldLight = m_pSelectedLight;

	if (m_Lights.size() <= 0)
	{
		m_pSelectedLight = NULL;
	}
	else
	{
		// Compute new index
		const int NewIndex = qMin(m_Lights.size() - 1, qMax(0, Index));

		// Set selected node
		m_pSelectedLight = &m_Lights[NewIndex];
	}

	// Notify others that our selection has changed
	emit LightSelectionChanged(pOldLight, m_pSelectedLight);
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
