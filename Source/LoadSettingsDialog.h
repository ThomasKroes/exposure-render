#pragma once

#include <QtGui>

#include "Controls.h"

class QLoadSettingsDialog : public QDialog
{
	Q_OBJECT

public:
    QLoadSettingsDialog(QWidget* pParent = NULL);

	virtual void accept();

public:
	bool GetResample(void);
	float GetResampleX(void);
	float GetResampleY(void);
	float GetResampleZ(void);

	void Reset(void);
	void SetToolTips(void);

private slots:
	void LockY(const int& State);
	void LockZ(const int& State);
	void SetResample(const int& Resample);
	void SetResampleX(const double& ResampleX);
	void SetResampleY(const double& ResampleY);
	void SetResampleZ(const double& ResampleZ);
	void Accept(void);
	void Reject(void);
	void Clicked(QAbstractButton* pButton);

private:
	QGridLayout			m_MainLayout;
	QGroupBox			m_ResampleGroupBox;
	QGridLayout			m_ResampleLayout;
	QLabel				m_ResampleXLabel;
	QDoubleSlider		m_ResampleXSlider;
	QDoubleSpinner		m_ResampleXSpinBox;
	QLabel				m_ResampleYLabel;
	QDoubleSlider		m_ResampleYSlider;
	QDoubleSpinner		m_ResampleYSpinBox;
	QCheckBox			m_LockYCheckBox;
	QLabel				m_ResampleZLabel;
	QDoubleSlider		m_ResampleZSlider;
	QDoubleSpinner		m_ResampleZSpinBox;
	QCheckBox			m_LockZCheckBox;
	QDialogButtonBox	m_DialogButtons;

	bool				m_Resample;
	float				m_ResampleX;
	float				m_ResampleY;
	float				m_ResampleZ;
};