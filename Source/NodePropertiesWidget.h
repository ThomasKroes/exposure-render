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
	void OnDiffuseColorChanged(const QColor& DiffuseColor);
	void OnSpecularColorChanged(const QColor& Color);
	void OnRoughnessChanged(const double& Roughness);
	void OnNodeIntensityChanged(QNode* pNode);
	void OnNodeOpacityChanged(QNode* pNode);
	void OnNodeDiffuseColorChanged(QNode* pNode);
	void OnNodeSpecularColorChanged(QNode* pNode);
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
	QColorPushButton		m_DiffuseColor;
	QColorPushButton		m_SpecularColor;
	QLabel					m_RoughnessLabel;
	QDoubleSlider			m_RoughnessSlider;
	QDoubleSpinner			m_RoughnessSpinBox;
};