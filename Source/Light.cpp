
// Precompiled headers
#include "Stable.h"

#include "Light.h"

QLight::QLight(QObject* pParent) :
	QPresetXML(pParent),
	m_Theta(0.0f),
	m_Phi(0.0f),
	m_Distance(5.0f),
	m_Width(1.0f),
	m_Height(1.0f),
	m_LockSize(true),
	m_Color(QColor(250, 231, 154)),
	m_Intensity(1000.0f)
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