#pragma once

#include <QtGui>

#include "TransferFunction.h"

class QTransferFunction;
class QTransferFunctionView;
class QGradientView;
class QNodeItem;

class CNodePropertiesWidget : public QWidget
{
    Q_OBJECT

public:
	CNodePropertiesWidget(QWidget* pParent, QTransferFunction* pTransferFunction);

private slots:
	void OnNodeSelectionChanged(QNode* pNode);
	void OnNodeSelectionChanged(const int& Index);
	void OnPreviousNode(void);
	void OnNextNode(void);
	void OnPositionChanged(const int& Position);
	void OnOpacityChanged(const int& Opacity);
	void OnColorChanged(const QColor& Color);
	void OnNodePositionChanged(QNode* pNode);
	void OnNodeOpacityChanged(QNode* pNode);
	void OnNodeColorChanged(QNode* pNode);
	void OnNodeAdd(QNode* pNode);
	void OnNodeRemove(QNode* pNode);
	void OnTransferFunctionChanged(void);

private:
	QTransferFunction*		m_pTransferFunction;
	QNode*					m_pLastSelectedNode;
	QGridLayout*			m_pMainLayout;
	QLabel*					m_pSelectionLabel;
	QGridLayout*			m_pSelectionLayout;
	QComboBox*				m_pNodeSelectionComboBox;
	QPushButton*			m_pPreviousNodePushButton;
	QPushButton*			m_pNextNodePushButton;
	QLabel*					m_pPositionLabel;
	QSlider*				m_pPositionSlider;
	QSpinBox*				m_pPositionSpinBox;
	QLabel*					m_pOpacityLabel;
	QSlider*				m_pOpacitySlider;
	QSpinBox*				m_pOpacitySpinBox;
	QLabel*					m_pColorLabel;
	QComboBox*				m_pColorComboBox;
	QLabel*					m_pRoughnessLabel;
	QSlider*				m_pRoughnessSlider;
	QSpinBox*				m_pRoughnessSpinBox;
};

class CTransferFunctionWidget : public QGroupBox
{
    Q_OBJECT

public:
    CTransferFunctionWidget(QWidget* pParent = NULL);

protected:
	QGridLayout*				m_pMainLayout;
	QTransferFunction*			m_pTransferFunction;
	QTransferFunctionView*		m_pTransferFunctionView;
	QGradientView*				m_pGradientView;
	CNodePropertiesWidget*		m_pNodePropertiesWidget;
};