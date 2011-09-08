#pragma once

#include <QtGui>

#include "Controls.h"

class QTransferFunction;
class QNode;

class QNodePropertiesWidget : public QGroupBox
{
    Q_OBJECT

public:
	QNodePropertiesWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(10, 10); }

private slots:
	void OnNodeSelectionChanged(QNode* pNode);
	void OnNodeSelectionChanged(const int& Index);
	void OnPreviousNode(void);
	void OnNextNode(void);
	void OnDeleteNode(void);
	void OnIntensityChanged(const double& Position);
	void OnOpacityChanged(const double& Opacity);
	void OnDiffuseChanged(const QColor& Diffuse);
	void OnSpecularChanged(const QColor& Specular);
	void OnEmissionChanged(const QColor& Emission);
	void OnRoughnessChanged(const double& Roughness);
	void OnNodeIntensityChanged(QNode* pNode);
	void OnNodeOpacityChanged(QNode* pNode);
	void OnNodeDiffuseChanged(QNode* pNode);
	void OnNodeSpecularChanged(QNode* pNode);
	void OnNodeEmissionChanged(QNode* pNode);
	void OnNodeRoughnessChanged(QNode* pNode);

private:
	void SetupSelectionUI(void);

private:
	QGridLayout				m_MainLayout;
	QLabel					m_SelectionLabel;
	QGridLayout				m_SelectionLayout;
	QComboBox				m_NodeSelection;
	QPushButton				m_PreviousNode;
	QPushButton				m_NextNode;
	QPushButton				m_DeleteNode;
	QLabel					m_IntensityLabel;
	QDoubleSlider			m_IntensitySlider;
	QDoubleSpinner			m_IntensitySpinBox;
	QLabel					m_OpacityLabel;
	QDoubleSlider			m_OpacitySlider;
	QDoubleSpinner			m_OpacitySpinBox;
	QColorPushButton		m_Diffuse;
	QColorPushButton		m_Specular;
	QColorPushButton		m_Emission;
	QLabel					m_RoughnessLabel;
	QDoubleSlider			m_RoughnessSlider;
	QDoubleSpinner			m_RoughnessSpinBox;
};