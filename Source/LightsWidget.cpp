
#include "LightsWidget.h"
#include "Scene.h"

QLight::QLight(QObject* pParent) :
	QObject(pParent)
{
	SetName("Undefined");
	SetTheta(RandomFloat() * TWO_RAD_F);
	SetPhi(RandomFloat() * RAD_F);
	SetWidth(0.5f);
	SetHeight(0.5f);
	SetDistance(1.0f);
	SetIntensity(1.0f);
}

QString QLight::GetName(void)
{
	return m_Name;
}

void QLight::SetName(const QString& Name)
{
	m_Name = Name;

	emit LightPropertiesChanged(this);
}

float QLight::GetTheta(void) const
{
	return m_Theta;
}

void QLight::SetTheta(const float& Theta)
{
	m_Theta = Theta;

	emit LightPropertiesChanged(this);
}

float QLight::GetPhi(void) const
{
	return m_Phi;
}

void QLight::SetPhi(const float& Phi)
{
	m_Phi = Phi;

	emit LightPropertiesChanged(this);
}

float QLight::GetWidth(void) const
{
	return m_Width;
}

void QLight::SetWidth(const float& Width)
{
	m_Width = Width;

	emit LightPropertiesChanged(this);
}

float QLight::GetHeight(void) const
{
	return m_Height;
}

void QLight::SetHeight(const float& Height)
{
	m_Height = Height;

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

	emit LightPropertiesChanged(this);
}

QColor QLight::GetColor(void) const
{
	return m_Color;
}

void QLight::SetColor(const QColor& Color)
{
	m_Color = Color;

	emit LightPropertiesChanged(this);
}

float QLight::GetIntensity(void) const
{
	return m_Intensity;
}

void QLight::SetIntensity(const float& Intensity)
{
	m_Intensity = Intensity;

	emit LightPropertiesChanged(this);
}

QLightsWidget::QLightsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_Lights(),
	m_LightList(),
	m_LightName(),
	m_AddLight(),
	m_RemoveLight()
{
	// Title, status and tooltip
	setTitle("Lights");
	setToolTip("Lights");
	setStatusTip("Lights");

	// Apply main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	// Lights list
//	m_LightList.setParent(this);
	m_LightList.setSelectionMode(QAbstractItemView::SingleSelection);
	m_LightList.setAlternatingRowColors(true);
	m_LightList.setSortingEnabled(true);
	m_MainLayout.addWidget(&m_LightList, 0, 0, 1, 3);

	// Light name
	m_MainLayout.addWidget(&m_LightName, 1, 0);

	// Add light
	m_AddLight.setParent(this);
	m_AddLight.setEnabled(false);
	m_AddLight.setText("Add");
	m_AddLight.setToolTip("Add light");
	m_AddLight.setStatusTip("Add a new light to the scene");
	m_AddLight.setFixedWidth(20);
	m_AddLight.setFixedHeight(20);
	m_MainLayout.addWidget(&m_AddLight, 1, 1);

	// Remove light
	m_RemoveLight.setParent(this);
	m_RemoveLight.setEnabled(false);
	m_RemoveLight.setText("Remove");
	m_RemoveLight.setToolTip("Remove light");
	m_RemoveLight.setStatusTip("Remove a light from the scene");
	m_RemoveLight.setFixedWidth(20);
	m_RemoveLight.setFixedHeight(20);
	m_MainLayout.addWidget(&m_RemoveLight, 1, 2);

	connect(&m_LightList, SIGNAL(itemSelectionChanged()), this, SLOT(OnLightSelectionChanged()));
	connect(&m_AddLight, SIGNAL(clicked()), this, SLOT(OnAddLight()));
	connect(&m_RemoveLight, SIGNAL(clicked()), this, SLOT(OnRemoveLight()));
	connect(&m_LightName, SIGNAL(textChanged(const QString&)), this, SLOT(OnPresetNameChanged(const QString&)));
	connect(&m_LightList, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(OnLightItemChanged(QListWidgetItem*)));
}

void QLightsWidget::UpdateLightList(void)
{
	m_LightList.clear();

	for (int i = 0; i < m_Lights.size(); i++)
	{
		// Create new list item
		QLightItem* pLightItem = new QLightItem(&m_LightList, &m_Lights[i]);

		pLightItem->setFlags(pLightItem->flags() | Qt::ItemIsEditable);

		// Add the item
		m_LightList.addItem(pLightItem);
	}

	// Get current row
	const int CurrentRow = m_LightList.currentRow();

	m_AddLight.setEnabled(m_LightName.text().isEmpty() ? false : true);
	m_RemoveLight.setEnabled(CurrentRow >= 0);
}

void QLightsWidget::OnLightSelectionChanged(void)
{
	// Get current row
	int CurrentRow = m_LightList.currentRow();

	m_RemoveLight.setEnabled(CurrentRow >= 0);

	if (CurrentRow < 0)
		return;

	m_LightName.setText(m_LightList.currentItem()->text());

	QLightItem* pLightItem = dynamic_cast<QLightItem*>(m_LightList.currentItem());

	if (pLightItem)
		emit LightSelectionChanged(pLightItem->m_pLight);
}

void QLightsWidget::OnLightItemChanged(QListWidgetItem* pWidgetItem)
{
	QLightItem* pLightItem = dynamic_cast<QLightItem*>(m_LightList.currentItem());

	if (pLightItem)
		pLightItem->m_pLight->SetName(pWidgetItem->text());
}

void QLightsWidget::OnPresetNameChanged(const QString& Text)
{
	m_AddLight.setEnabled(Text.length() > 0);
}

void QLightsWidget::OnAddLight(void)
{
	if (m_LightName.text().isEmpty())
		return;

	QLight NewLight;
	NewLight.SetName(m_LightName.text());

	m_Lights.append(NewLight);

	UpdateLightList();
}

void QLightsWidget::OnRemoveLight(void)
{
	// Get current row
	const int CurrentRow = m_LightList.currentRow();

	if (CurrentRow < 0)
		return;

	m_Lights.removeAt(CurrentRow);

	UpdateLightList();
}

void QLightsWidget::OnLightPropertiesChanged(QLight* pLight)
{
	if (m_Lights.isEmpty())
		return;

	gpScene->m_Lights.m_NoLights = m_Lights.size();

	for (int i = 0; i < m_Lights.size(); i++)
	{
		QLight& Light = m_Lights[i];

		CLight NewLight;

		NewLight.m_Theta	= Light.GetTheta();
		NewLight.m_Phi		= Light.GetPhi();
		NewLight.m_Distance	= Light.GetDistance();
		NewLight.m_Width	= Light.GetWidth();
		NewLight.m_Height	= Light.GetHeight();
		NewLight.m_Color.r	= Light.GetColor().red() * Light.GetIntensity();
		NewLight.m_Color.g	= Light.GetColor().green() * Light.GetIntensity();
		NewLight.m_Color.b	= Light.GetColor().blue() * Light.GetIntensity();

		gpScene->m_Lights.m_Lights[i] = NewLight;
	}

	// Flag the lights as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(LightsDirty);

}