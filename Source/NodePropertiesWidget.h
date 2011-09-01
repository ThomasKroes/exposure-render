#pragma once

#include <QtGui>

#include "ColorButtonWidget.h"

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
	void OnIntensityChanged(const int& Position);
	void OnOpacityChanged(const int& Opacity);
	void OnColorChanged(const QColor& Color);
	void OnNodeIntensityChanged(QNode* pNode);
	void OnNodeOpacityChanged(QNode* pNode);
	void OnNodeColorChanged(QNode* pNode);

private:
	void SetupSelectionUI(void);
	void SetupIntensityUI(void);
	void SetupOpacityUI(void);
	void SetupColorUI(void);

private:
	QGridLayout				m_MainLayout;
	QLabel					m_SelectionLabel;
	QGridLayout				m_SelectionLayout;
	QComboBox				m_NodeSelection;
	QPushButton				m_PreviousNode;
	QPushButton				m_NextNode;
	QPushButton				m_DeleteNode;
	QLabel					m_IntensityLabel;
	QSlider					m_IntensitySlider;
	QSpinBox				m_IntensitySpinBox;
	QLabel					m_OpacityLabel;
	QSlider					m_OpacitySlider;
	QSpinBox				m_OpacitySpinBox;
	QColorPushButton		m_KdColor;
	QColorPushButton		m_KsColor;
	QColorPushButton		m_KtColor;
	QLabel					m_RoughnessLabel;
	QSlider					m_RoughnessSlider;
	QSpinBox				m_RoughnessSpinBox;
};