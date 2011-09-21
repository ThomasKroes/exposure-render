
// Precompiled headers
#include "Stable.h"

#include "Node.h"
#include "TransferFunction.h"

QNode::QNode(const QNode& Other)
{
	*this = Other;
};

QNode::QNode(QTransferFunction* pTransferFunction, const float& Intensity, const float& Opacity, const QColor& Diffuse, const QColor& Specular, const QColor& Emission, const float& Roughness) :
	QPresetXML(pTransferFunction),
	m_pTransferFunction(pTransferFunction),
	m_Intensity(Intensity),
	m_Opacity(Opacity),
	m_Diffuse(Diffuse),
	m_Specular(Specular),
	m_Emission(Emission),
	m_Glossiness(Roughness),
	m_MinX(0.0f),
	m_MaxX(1.0f),
	m_MinY(0.0f),
	m_MaxY(1.0f),
	m_ID(0)
{
}

QNode& QNode::operator=(const QNode& Other)
{
	m_pTransferFunction	= Other.m_pTransferFunction;
	m_Intensity			= Other.m_Intensity;
	m_Opacity			= Other.m_Opacity;
	m_Diffuse			= Other.m_Diffuse;
	m_Specular			= Other.m_Specular;
	m_Emission			= Other.m_Emission;
	m_Glossiness			= Other.m_Glossiness;
	m_MinX				= Other.m_MinX;
	m_MaxX				= Other.m_MaxX;
	m_MinY				= Other.m_MinY;
	m_MaxY				= Other.m_MaxY;
	m_ID				= Other.m_ID;

	return *this;
}

bool QNode::operator==(const QNode& Other) const
{
	return m_ID == Other.m_ID;
}

QTransferFunction* QNode::GetTransferFunction(void)
{
	return m_pTransferFunction;
}

float QNode::GetIntensity(void) const
{
	return m_Intensity;
}

void QNode::SetIntensity(const float& Intensity)
{
	if (Intensity == m_Intensity)
		return;

	m_Intensity = qMin(m_MaxX, qMax(Intensity, m_MinX));

	SetDirty(true);

	emit NodeChanged(this);
	emit IntensityChanged(this);
}

float QNode::GetOpacity(void) const
{
	return m_Opacity;
}

void QNode::SetOpacity(const float& Opacity)
{
	if (Opacity == m_Opacity)
		return;

	m_Opacity = qMin(m_MaxY, qMax(Opacity, m_MinY));

	SetDirty(true);

	emit NodeChanged(this);
	emit OpacityChanged(this);
}

QColor QNode::GetDiffuse(void) const
{
	return m_Diffuse;
}

void QNode::SetDiffuse(const QColor& Diffuse)
{
	if (Diffuse == m_Diffuse)
		return;

	m_Diffuse = Diffuse;

	SetDirty(true);

	emit NodeChanged(this);
	emit DiffuseChanged(this);
}

QColor QNode::GetSpecular(void) const
{
	return m_Specular;
}

void QNode::SetSpecular(const QColor& Specular)
{
	if (Specular == m_Specular)
		return;

	m_Specular = Specular;

	SetDirty(true);

	emit NodeChanged(this);
	emit SpecularChanged(this);
}

QColor QNode::GetEmission(void) const
{
	return m_Emission;
}

void QNode::SetEmission(const QColor& Emission)
{
	if (Emission == m_Emission)
		return;

	m_Emission = Emission;

	SetDirty(true);

	emit NodeChanged(this);
	emit EmissionChanged(this);
}

float QNode::GetGlossiness(void) const
{
	return m_Glossiness;
}

void QNode::SetGlossiness(const float& Roughness)
{
	if (Roughness == m_Glossiness)
		return;

	m_Glossiness = Roughness;

	SetDirty(true);

	emit NodeChanged(this);
	emit RoughnessChanged(this);
}

float QNode::GetMinX(void) const
{
	return m_MinX;
}

void QNode::SetMinX(const float& MinX)
{
	if (MinX == m_MinX)
		return;

	m_MinX = MinX;

	SetDirty(true);

	emit RangeChanged(this);
}

float QNode::GetMaxX(void) const
{
	return m_MaxX;
}

void QNode::SetMaxX(const float& MaxX)
{
	if (MaxX == m_MaxX)
		return;

	m_MaxX = MaxX;

	SetDirty(true);

	emit RangeChanged(this);
}

float QNode::GetMinY(void) const
{
	return m_MinY;
}

void QNode::SetMinY(const float& MinY)
{
	if (MinY == m_MinY)
		return;

	m_MinY = MinY;

	SetDirty(true);

	emit RangeChanged(this);
}

float QNode::GetMaxY(void) const
{
	return m_MaxY;
}

void QNode::SetMaxY(const float& MaxY)
{
	if (MaxY == m_MaxY)
		return;

	m_MaxY = MaxY;

	SetDirty(true);

	emit RangeChanged(this);
}

bool QNode::InRange(const QPointF& Point)
{
	return Point.x() >= m_MinX && Point.x() <= m_MaxX && Point.y() >= m_MinY && Point.y() <= m_MaxY;
}

int QNode::GetID(void) const
{
	return m_ID;
}

bool QNode::GetDirty(void) const
{
	return m_Dirty;
}

void QNode::SetDirty(const bool& Dirty)
{
	m_Dirty = Dirty;
}

void QNode::ReadXML(QDomElement& Parent)
{
	// Intensity
	m_Intensity = Parent.firstChildElement("NormalizedIntensity").attribute("Value").toFloat();

	// Opacity
	m_Opacity = Parent.firstChildElement("Opacity").attribute("Value").toFloat();

	// Diffuse
	QDomElement Diffuse = Parent.firstChildElement("Diffuse");
	m_Diffuse.setRed(Diffuse.attribute("R").toInt());
	m_Diffuse.setGreen(Diffuse.attribute("G").toInt());
	m_Diffuse.setBlue(Diffuse.attribute("B").toInt());

	// Specular
	QDomElement Specular = Parent.firstChildElement("Specular");
	m_Specular.setRed(Specular.attribute("R").toInt());
	m_Specular.setGreen(Specular.attribute("G").toInt());
	m_Specular.setBlue(Specular.attribute("B").toInt());

	// Emission
	QDomElement Emission = Parent.firstChildElement("Emission");
	m_Emission.setRed(Emission.attribute("R").toInt());
	m_Emission.setGreen(Emission.attribute("G").toInt());
	m_Emission.setBlue(Emission.attribute("B").toInt());

	// Roughness
	m_Glossiness = Parent.firstChildElement("Roughness").attribute("Value").toFloat();
}

QDomElement QNode::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Node
	QDomElement Node = DOM.createElement("Node");
	Parent.appendChild(Node);

	// Intensity
	QDomElement Intensity = DOM.createElement("NormalizedIntensity");
	Intensity.setAttribute("Value", GetIntensity());
	Node.appendChild(Intensity);

	// Opacity
	QDomElement Opacity = DOM.createElement("Opacity");
	Opacity.setAttribute("Value", GetOpacity());
	Node.appendChild(Opacity);

	// Diffuse
	QDomElement DiffuseColor = DOM.createElement("Diffuse");
	DiffuseColor.setAttribute("R", m_Diffuse.red());
	DiffuseColor.setAttribute("G", m_Diffuse.green());
	DiffuseColor.setAttribute("B", m_Diffuse.blue());
	Node.appendChild(DiffuseColor);

	// Specular
	QDomElement SpecularColor = DOM.createElement("Specular");
	SpecularColor.setAttribute("R", m_Specular.red());
	SpecularColor.setAttribute("G", m_Specular.green());
	SpecularColor.setAttribute("B", m_Specular.blue());
	Node.appendChild(SpecularColor);

	// Emission
	QDomElement EmissionColor = DOM.createElement("Emission");
	EmissionColor.setAttribute("R", m_Emission.red());
	EmissionColor.setAttribute("G", m_Emission.green());
	EmissionColor.setAttribute("B", m_Emission.blue());
	Node.appendChild(EmissionColor);

	// Roughness
	QDomElement Roughness = DOM.createElement("Roughness");
	Roughness.setAttribute("Value", m_Glossiness);
	Node.appendChild(Roughness);

	return Node;
}