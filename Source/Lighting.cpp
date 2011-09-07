
#include "Lighting.h"
#include "RenderThread.h"

QLighting gLighting;

QLight::QLight(QObject* pParent) :
	QPresetXML(pParent),
	m_Theta(0.0f),
	m_Phi(0.0f),
	m_Distance(10.0f),
	m_Width(1.0f),
	m_Height(1.0f),
	m_LockSize(true),
	m_Color(QColor(250, 231, 154)),
	m_Intensity(100.0f)
{
}

QLight::~QLight(void)
{
}

QLight::QLight(const QLight& Other)
{
	*this = Other;
};

QLight& QLight::operator=(const QLight& Other)
{
	QPresetXML::operator=(Other);

	m_Theta					= Other.m_Theta;
	m_Phi					= Other.m_Phi;
	m_Distance				= Other.m_Distance;
	m_Width					= Other.m_Width;
	m_Height				= Other.m_Height;
	m_LockSize				= Other.m_LockSize;
	m_Color					= Other.m_Color;
	m_Intensity				= Other.m_Intensity;

	return *this;
}

bool QLight::operator == (const QLight& Other) const
{
	return GetName() == Other.GetName();
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

	emit LockSizeChanged(this);
	emit LightPropertiesChanged(this);
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
	QPresetXML::operator=(Other);

	foreach (QLight Light, m_Lights)
		disconnect(&m_Lights.back(), SIGNAL(LightPropertiesChanged(QLight*)), this, SLOT(OnLightPropertiesChanged(QLight*)));

	m_Lights			= Other.m_Lights;

	foreach (QLight Light, m_Lights)
		connect(&m_Lights.back(), SIGNAL(LightPropertiesChanged(QLight*)), this, SLOT(OnLightPropertiesChanged(QLight*)));

	m_pSelectedLight	= Other.m_pSelectedLight;
	m_Background		= Other.m_Background;

	emit LightingChanged();

	return *this;
}

void QLighting::OnLightPropertiesChanged(QLight* pLight)
{
	Update();
}

void QLighting::Update(void)
{
	if (!Scene())
		return;

	Scene()->m_Lighting.Reset();

	/*
	if (Background().GetEnabled())
	{
		CLight BackgroundLight;

		BackgroundLight.m_Type	= CLight::Background;
		BackgroundLight.m_Color	= CColorRgbHdr(Background().GetColor().redF(), Background().GetColor().greenF(), Background().GetColor().blueF());

		Scene()->m_Lighting.AddLight(BackgroundLight);
	}
	*/
	for (int i = 0; i < m_Lights.size(); i++)
	{
		QLight& Light = m_Lights[i];

		CLight AreaLight;

		AreaLight.m_Type		= CLight::Area;
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

void QLighting::AddLight(QLight& Light)
{
	// Add to list
	m_Lights.append(Light);

	// Select
	SetSelectedLight(&m_Lights.back());

	// Connect
	connect(&m_Lights.back(), SIGNAL(LightPropertiesChanged(QLight*)), this, SLOT(OnLightPropertiesChanged(QLight*)));

	// Let others know the lighting has changed
	emit LightingChanged();
}

void QLighting::RemoveLight(QLight* pLight)
{
	// Remove from light list
	m_Lights.remove(*pLight);

	m_pSelectedLight = NULL;

	// Deselect
	SetSelectedLight(NULL);

	// Let others know the lighting has changed
	emit LightingChanged();
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
	emit LightingChanged();
}

void QLighting::CopySelectedLight(void)
{
	CopyLight(m_pSelectedLight);
}

void QLighting::RenameLight(const int& Index, const QString& Name)
{
	if (Index < 0 || Index >= m_Lights.size() || Name.isEmpty())
		return;

	m_Lights[Index].SetName(Name);

	// Let others know the lighting has changed
	emit LightingChanged();
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

	QLight Light;
	Light.SetName("Key");

	DefaultLighting.AddLight(Light);

	return DefaultLighting;
}