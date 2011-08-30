#pragma once

#include <QtGui>

class QTransferFunction;
class QNode;
class QColorSelectorWidget;

class QNodePropertiesWidget : public QGroupBox
{
    Q_OBJECT

public:
	QNodePropertiesWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(150, 300); }

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
	QNode*					m_pLastSelectedNode;
	QGridLayout*			m_pMainLayout;
	QLabel*					m_pSelectionLabel;
	QGridLayout*			m_pSelectionLayout;
	QComboBox*				m_pNodeSelectionComboBox;
	QPushButton*			m_pPreviousNodePushButton;
	QPushButton*			m_pNextNodePushButton;
	QPushButton*			m_pDeleteNodePushButton;
	QLabel*					m_pIntensityLabel;
	QSlider*				m_pIntensitySlider;
	QSpinBox*				m_pIntensitySpinBox;
	QLabel*					m_pOpacityLabel;
	QSlider*				m_pOpacitySlider;
	QSpinBox*				m_pOpacitySpinBox;
	QColorSelectorWidget*	m_pColorSelector;
	QLabel*					m_pRoughnessLabel;
	QSlider*				m_pRoughnessSlider;
	QSpinBox*				m_pRoughnessSpinBox;
};