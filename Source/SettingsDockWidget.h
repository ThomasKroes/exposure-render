#pragma once

#include <QtGui>

class CTracerSettingsWidget : public QGroupBox
{
    Q_OBJECT

public:
    CTracerSettingsWidget(QWidget* pParent = NULL);

private slots:
	void SetNoBounces(const int& NoBounces);
	void SetPhase(const int& Phase);

private:
	QGridLayout*	m_pGridLayout;
	QComboBox*		m_pRenderTypeComboBox;
	QLabel*			m_pNoBouncesLabel;
	QSlider*		m_pNoBouncesSlider;
	QSpinBox*		m_pNoBouncesSpinBox;
	QLabel*			m_pPhaseLabel;
	QSlider*		m_pPhaseSlider;
	QSpinBox*		m_pPhaseSpinBox;
};

class CKernelSettingsWidget : public QGroupBox
{
    Q_OBJECT

public:
    CKernelSettingsWidget(QWidget* pParent = NULL);

private slots:
	void SetKernelWidth(const int& KernelWidth);
	void SetKernelHeight(const int& KernelHeight);
	void LockKernelHeight(const int& Lock);

private:
	QGridLayout*	m_pGridLayout;
	QLabel*			m_pKernelWidthLabel;
	QSlider*		m_pKernelWidthSlider;
	QSpinBox*		m_pKernelWidthSpinBox;
	QLabel*			m_pKernelHeightLabel;
	QSlider*		m_pKernelHeightSlider;
	QSpinBox*		m_pKernelHeightSpinBox;
	QCheckBox*		m_pLockKernelHeightCheckBox;
};

class CSettingsWidget : public QWidget
{
    Q_OBJECT

public:
    CSettingsWidget(QWidget* pParent = NULL);

private:
	QVBoxLayout*				m_pMainLayout;
	CTracerSettingsWidget*		m_pTracerSettingsWidget;
	CKernelSettingsWidget*		m_pKernelSettingsWidget;
};

class CSettingsDockWidget : public QDockWidget
{
	Q_OBJECT

public:
    CSettingsDockWidget(QWidget* pParent = NULL);

private:
	
	CSettingsWidget*	m_pSettingsWidget;
};