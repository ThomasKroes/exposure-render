#pragma once

class QNode;

class QNodePropertiesWidget : public QGroupBox
{
    Q_OBJECT

public:
	QNodePropertiesWidget(QWidget* pParent = NULL);

private slots:
	void OnNodeSelectionChanged(QNode* pNode);
	void OnIntensityChanged(const double& Position);
	void OnOpacityChanged(const double& Opacity);
	void OnDiffuseChanged(const QColor& Diffuse);
	void OnSpecularChanged(const QColor& Specular);
	void OnEmissionChanged(const QColor& Emission);
	void OnGlossinessChanged(const double& Roughness);
	void OnNodeIntensityChanged(QNode* pNode);
	void OnNodeOpacityChanged(QNode* pNode);
	void OnNodeDiffuseChanged(QNode* pNode);
	void OnNodeSpecularChanged(QNode* pNode);
	void OnNodeEmissionChanged(QNode* pNode);
	void OnNodeGlossinessChanged(QNode* pNode);

private:
	QGridLayout		m_MainLayout;
	QLabel			m_IntensityLabel;
	QDoubleSlider	m_IntensitySlider;
	QDoubleSpinner	m_IntensitySpinBox;
	QLabel			m_OpacityLabel;
	QDoubleSlider	m_OpacitySlider;
	QDoubleSpinner	m_OpacitySpinBox;
	QColorSelector	m_Diffuse;
	QColorSelector	m_Specular;
	QLabel			m_GlossinessLabel;
	QDoubleSlider	m_GlossinessSlider;
	QDoubleSpinner	m_GlossinessSpinner;
	QColorSelector	m_Emission;
};