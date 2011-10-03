
// Precompiled headers
#include "Stable.h"

#include "NodePropertiesWidget.h"
#include "TransferFunction.h"
#include "NodeItem.h"
#include "RenderThread.h"

QNodePropertiesWidget::QNodePropertiesWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_IntensityLabel(),
	m_IntensitySlider(),
	m_IntensitySpinBox(),
	m_OpacityLabel(),
	m_OpacitySlider(),
	m_OpacitySpinBox(),
	m_Diffuse(),
	m_Specular(),
	m_GlossinessLabel(),
	m_GlossinessSlider(),
	m_GlossinessSpinner(),
	m_Emission()
{
	setTitle("Node Properties");
	setToolTip("Node Properties");
	setStatusTip("Node Properties");

	// Main layout
	m_MainLayout.setAlignment(Qt::AlignTop);

	setLayout(&m_MainLayout);

	setAlignment(Qt::AlignTop);

	// Position
	m_IntensityLabel.setFixedWidth(50);
	m_IntensityLabel.setText("Intensity");
	m_IntensityLabel.setStatusTip("Node Intensity");
	m_IntensityLabel.setToolTip("Node Intensity");
	m_MainLayout.addWidget(&m_IntensityLabel, 1, 0);

	m_IntensitySlider.setOrientation(Qt::Horizontal);
	m_IntensitySlider.setStatusTip("Node Intensity");
	m_IntensitySlider.setToolTip("Drag to change node intensity");
	m_MainLayout.addWidget(&m_IntensitySlider, 1, 1);
	
	m_IntensitySpinBox.setStatusTip("Node Position");
	m_IntensitySpinBox.setToolTip("Node Position");
	m_IntensitySpinBox.setSingleStep(1);
	m_MainLayout.addWidget(&m_IntensitySpinBox, 1, 2);

	QObject::connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), &m_IntensitySpinBox, SLOT(setValue(double)));
	QObject::connect(&m_IntensitySpinBox, SIGNAL(valueChanged(double)), &m_IntensitySlider, SLOT(setValue(double)));
	QObject::connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), this, SLOT(OnIntensityChanged(double)));

	// Opacity
	m_OpacityLabel.setText("Opacity");
	m_MainLayout.addWidget(&m_OpacityLabel, 2, 0);

	m_OpacitySlider.setOrientation(Qt::Horizontal);
	m_OpacitySlider.setStatusTip("Node Opacity");
	m_OpacitySlider.setToolTip("Node Opacity");
	m_OpacitySlider.setRange(0.0, 1.0);
	m_MainLayout.addWidget(&m_OpacitySlider, 2, 1);
	
	m_OpacitySpinBox.setStatusTip("Node Opacity");
	m_OpacitySpinBox.setToolTip("Node Opacity");
	m_OpacitySpinBox.setRange(0.0, 1.0);
	m_OpacitySpinBox.setDecimals(3);
	m_OpacitySpinBox.setSingleStep(0.01);

	m_MainLayout.addWidget(&m_OpacitySpinBox, 2, 2);
	
	QObject::connect(&m_OpacitySlider, SIGNAL(valueChanged(double)), &m_OpacitySpinBox, SLOT(setValue(double)));
	QObject::connect(&m_OpacitySpinBox, SIGNAL(valueChanged(double)), &m_OpacitySlider, SLOT(setValue(double)));
	QObject::connect(&m_OpacitySlider, SIGNAL(valueChanged(double)), this, SLOT(OnOpacityChanged(double)));

	// Diffuse
	m_MainLayout.addWidget(new QLabel("Diffuse"), 3, 0);
// 	m_Diffuse.setFixedWidth(50);
	m_MainLayout.addWidget(&m_Diffuse, 3, 1, 1, 2);

	QObject::connect(&m_Diffuse, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnDiffuseChanged(const QColor&)));

	// Specular
	m_MainLayout.addWidget(new QLabel("Specular"), 4, 0);
// 	m_Specular.setFixedWidth(50);
	m_MainLayout.addWidget(&m_Specular, 4, 1, 1, 2);

	QObject::connect(&m_Specular, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnSpecularChanged(const QColor&)));

	// Roughness
	m_GlossinessLabel.setText("Glossiness");
	m_MainLayout.addWidget(&m_GlossinessLabel, 5, 0);

	m_GlossinessSlider.setOrientation(Qt::Horizontal);
	m_GlossinessSlider.setStatusTip("Glossiness");
	m_GlossinessSlider.setToolTip("Glossiness");
	m_GlossinessSlider.setRange(0.0, 1.0);
	
	m_MainLayout.addWidget(&m_GlossinessSlider, 5, 1);

	m_GlossinessSpinner.setStatusTip("Glossiness");
	m_GlossinessSpinner.setToolTip("Glossiness");
	m_GlossinessSpinner.setRange(0.0, 1.0);
	m_GlossinessSpinner.setDecimals(3);

	m_MainLayout.addWidget(&m_GlossinessSpinner, 5, 2);

	QObject::connect(&m_GlossinessSlider, SIGNAL(valueChanged(double)), &m_GlossinessSpinner, SLOT(setValue(double)));
	QObject::connect(&m_GlossinessSpinner, SIGNAL(valueChanged(double)), &m_GlossinessSlider, SLOT(setValue(double)));
	QObject::connect(&m_GlossinessSlider, SIGNAL(valueChanged(double)), this, SLOT(OnGlossinessChanged(double)));

	// Emission
	m_MainLayout.addWidget(new QLabel("Emission"), 6, 0);
// 	m_Emission.setFixedWidth(50);
	m_MainLayout.addWidget(&m_Emission, 6, 1, 1, 2);

	QObject::connect(&m_Emission, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnEmissionChanged(const QColor&)));

	QObject::connect(&gTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));

	OnNodeSelectionChanged(NULL);
}

void QNodePropertiesWidget::OnNodeSelectionChanged(QNode* pNode)
{
	QObject::disconnect(this, SLOT(OnNodeIntensityChanged(QNode*)));
	QObject::disconnect(this, SLOT(OnNodeOpacityChanged(QNode*)));
	QObject::disconnect(this, SLOT(OnNodeDiffuseChanged(QNode*)));
	QObject::disconnect(this, SLOT(OnNodeSpecularChanged(QNode*)));
	QObject::disconnect(this, SLOT(OnNodeEmissionChanged(QNode*)));
	QObject::disconnect(this, SLOT(OnNodeGlossinessChanged(QNode*)));

	if (pNode)
	{
		QObject::connect(pNode, SIGNAL(IntensityChanged(QNode*)), this, SLOT(OnNodeIntensityChanged(QNode*)));
		QObject::connect(pNode, SIGNAL(OpacityChanged(QNode*)), this, SLOT(OnNodeOpacityChanged(QNode*)));
		QObject::connect(pNode, SIGNAL(DiffuseChanged(QNode*)), this, SLOT(OnNodeDiffuseChanged(QNode*)));
		QObject::connect(pNode, SIGNAL(SpecularChanged(QNode*)), this, SLOT(OnNodeSpecularChanged(QNode*)));
		QObject::connect(pNode, SIGNAL(EmissionChanged(QNode*)), this, SLOT(OnNodeEmissionChanged(QNode*)));
		QObject::connect(pNode, SIGNAL(RoughnessChanged(QNode*)), this, SLOT(OnNodeGlossinessChanged(QNode*)));
	}

	OnNodeIntensityChanged(pNode);
	OnNodeOpacityChanged(pNode);
	OnNodeDiffuseChanged(pNode);
	OnNodeSpecularChanged(pNode);
	OnNodeEmissionChanged(pNode);
	OnNodeGlossinessChanged(pNode);
}

void QNodePropertiesWidget::OnIntensityChanged(const double& Position)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetIntensity((Position - gScene.m_IntensityRange.GetMin()) / gScene.m_IntensityRange.GetLength());
}

void QNodePropertiesWidget::OnOpacityChanged(const double& Opacity)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetOpacity(Opacity);
}

void QNodePropertiesWidget::OnDiffuseChanged(const QColor& Diffuse)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetDiffuse(Diffuse);
}

void QNodePropertiesWidget::OnSpecularChanged(const QColor& Specular)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetSpecular(Specular);
}

void QNodePropertiesWidget::OnEmissionChanged(const QColor& Emission)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetEmission(Emission);
}

void QNodePropertiesWidget::OnGlossinessChanged(const double& Glossiness)
{
	if (gTransferFunction.GetSelectedNode())
		gTransferFunction.GetSelectedNode()->SetGlossiness(Glossiness);
}

void QNodePropertiesWidget::OnNodeIntensityChanged(QNode* pNode)
{
	bool Enable = false;

	if (pNode)
	{
		const int NodeIndex = gTransferFunction.GetNodes().indexOf(*pNode);

		Enable = NodeIndex != 0 && NodeIndex != gTransferFunction.GetNodes().size() - 1;

		m_IntensitySlider.setRange(gScene.m_IntensityRange.GetMin() + gScene.m_IntensityRange.GetLength() * pNode->GetMinX(), gScene.m_IntensityRange.GetMin() + gScene.m_IntensityRange.GetLength() * pNode->GetMaxX());
		m_IntensitySpinBox.setRange(gScene.m_IntensityRange.GetMin() + gScene.m_IntensityRange.GetLength() * pNode->GetMinX(), gScene.m_IntensityRange.GetMin() + gScene.m_IntensityRange.GetLength() * pNode->GetMaxX());

		m_IntensitySlider.setValue(gScene.m_IntensityRange.GetMin() + gScene.m_IntensityRange.GetLength() * pNode->GetIntensity(), true);
		m_IntensitySpinBox.setValue(gScene.m_IntensityRange.GetMin() + gScene.m_IntensityRange.GetLength() * pNode->GetIntensity(), true);
	}

	m_IntensityLabel.setEnabled(Enable);
	m_IntensitySlider.setEnabled(Enable);
	m_IntensitySpinBox.setEnabled(Enable);
}

void QNodePropertiesWidget::OnNodeOpacityChanged(QNode* pNode)
{
	const bool Enable = pNode != NULL;

	if (pNode)
	{
		m_OpacitySlider.setRange(pNode->GetMinY(), pNode->GetMaxY());
		m_OpacitySpinBox.setRange(pNode->GetMinY(), pNode->GetMaxY());

		m_OpacitySlider.setValue((double)pNode->GetOpacity(), true);
		m_OpacitySpinBox.setValue((double)pNode->GetOpacity(), true);
	}

	m_OpacityLabel.setEnabled(Enable);
	m_OpacitySlider.setEnabled(Enable);
	m_OpacitySpinBox.setEnabled(Enable);
}

void QNodePropertiesWidget::OnNodeDiffuseChanged(QNode* pNode)
{
	const bool Enable = pNode != NULL;

	if (pNode)
	{
		m_Diffuse.SetColor(pNode->GetDiffuse(), true);
	}

	m_Diffuse.setEnabled(Enable);
}

void QNodePropertiesWidget::OnNodeSpecularChanged(QNode* pNode)
{
	const bool Enable = pNode != NULL;

	if (pNode)
	{
		m_Specular.SetColor(pNode->GetSpecular(), true);
	}

	m_Specular.setEnabled(Enable);
}

void QNodePropertiesWidget::OnNodeEmissionChanged(QNode* pNode)
{
	const bool Enable = pNode != NULL;

	if (pNode)
	{
		m_Emission.SetColor(pNode->GetEmission(), true);
	}

	m_Emission.setEnabled(Enable);
}

void QNodePropertiesWidget::OnNodeGlossinessChanged(QNode* pNode)
{
	const bool Enable = pNode != NULL;

	if (pNode)
	{
		m_GlossinessSlider.setValue((double)pNode->GetGlossiness(), true);
	}

	m_GlossinessLabel.setEnabled(Enable);
	m_GlossinessSlider.setEnabled(Enable);
}