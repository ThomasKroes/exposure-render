#pragma once

#include <QtGui>

class QTransferFunction;
class QNode;
class QColorSelectorWidget;

class QNodePropertiesWidget : public QGroupBox
{
    Q_OBJECT

public:
	QNodePropertiesWidget(QWidget* pParent);

private slots:
	void OnNodeSelectionChanged(QNode* pNode);
	void OnNodeSelectionChanged(const int& Index);
	void OnPreviousNode(void);
	void OnNextNode(void);
	void OnDeleteNode(void);
	void OnPositionChanged(const int& Position);
	void OnOpacityChanged(const int& Opacity);
	void OnColorChanged(const QColor& Color);
	void OnNodePositionChanged(QNode* pNode);
	void OnNodeOpacityChanged(QNode* pNode);
	void OnNodeColorChanged(QNode* pNode);
	void OnNodeAdd(QNode* pNode);
	void OnNodeRemove(QNode* pNode);
	void OnNodeRemoved(QNode* pNode);

private:
	QNode*					m_pLastSelectedNode;
	QGridLayout*			m_pMainLayout;
	QLabel*					m_pSelectionLabel;
	QGridLayout*			m_pSelectionLayout;
	QComboBox*				m_pNodeSelectionComboBox;
	QPushButton*			m_pPreviousNodePushButton;
	QPushButton*			m_pNextNodePushButton;
	QPushButton*			m_pDeleteNodePushButton;
	QLabel*					m_pPositionLabel;
	QSlider*				m_pPositionSlider;
	QSpinBox*				m_pPositionSpinBox;
	QLabel*					m_pOpacityLabel;
	QSlider*				m_pOpacitySlider;
	QSpinBox*				m_pOpacitySpinBox;
	QColorSelectorWidget*	m_pColorSelector;
	QLabel*					m_pRoughnessLabel;
	QSlider*				m_pRoughnessSlider;
	QSpinBox*				m_pRoughnessSpinBox;
};