#pragma once

#include "TransferFunctionView.h"
#include "GradientRamp.h"

class QTransferFunctionWidget : public QGroupBox
{
    Q_OBJECT

public:
    QTransferFunctionWidget(QWidget* pParent = NULL);

public slots:
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnUpdateGradients(void);

protected:
	QGridLayout			m_MainLayout;
	QGridLayout			m_TopLeftLayout;
	QGridLayout			m_TopLayout;
	QGridLayout			m_TopRightLayout;
	QGridLayout			m_LeftLayout;
	QGridLayout			m_MiddleLayout;
	QGridLayout			m_RightLayout;
	QGridLayout			m_BottomLeftLayout;
	QGridLayout			m_BottomLayout;
	QGridLayout			m_BottomRightLayout;
	QTFView				m_Canvas;
	QGradientRamp		m_GradientRampDiffuse;
	QGradientRamp		m_GradientRampSpecular;
	QGradientRamp		m_GradientRampEmission;
};