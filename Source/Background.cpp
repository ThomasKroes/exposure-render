
#include "Background.h"

QBackground::QBackground(QObject* pParent) :
	QPresetXML(pParent),
	m_Enable(true),
	m_Color(Qt::white),
	m_Intensity(100.0),
	m_UseTexture(false),
	m_File("")
{
}

QBackground::~QBackground(void)
{
}

QBackground::QBackground(const QBackground& Other)
{
	*this = Other;
};

QBackground& QBackground::operator=(const QBackground& Other)
{
	QPresetXML::operator=(Other);

	m_Enable		= Other.m_Enable;
	m_Color			= Other.m_Color;
	m_Intensity		= Other.m_Intensity;
	m_UseTexture	= Other.m_UseTexture;
	m_File			= Other.m_File;

	emit BackgroundChanged();

	return *this;
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