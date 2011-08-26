#pragma once

#include <QtGui>

class CLoadSettingsDialog : public QDialog
{
	Q_OBJECT

public:
    CLoadSettingsDialog(QWidget* pParent = NULL);

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
	void SetResampleX(const int& ResampleX);
	void SetResampleY(const int& ResampleY);
	void SetResampleZ(const int& ResampleZ);
	void Accept(void);
	void Reject(void);
	void Clicked(QAbstractButton* pButton);

private:
	QVBoxLayout*		m_pMainLayout;
	QGroupBox*			m_pResampleGroupBox;
	QGridLayout*		m_pResampleLayout;
	QLabel*				m_pResampleXLabel;
	QSlider*			m_pResampleXSlider;
	QSpinBox*			m_pResampleXSpinBox;
	QLabel*				m_pResampleYLabel;
	QSlider*			m_pResampleYSlider;
	QSpinBox*			m_pResampleYSpinBox;
	QCheckBox*			m_pLockYCheckBox;
	QLabel*				m_pResampleZLabel;
	QSlider*			m_pResampleZSlider;
	QSpinBox*			m_pResampleZSpinBox;
	QCheckBox*			m_pLockZCheckBox;
	QDialogButtonBox*	m_pDialogButtons;

	bool				m_Resample;
	float				m_ResampleX;
	float				m_ResampleY;
	float				m_ResampleZ;
};