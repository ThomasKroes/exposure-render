#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "TransferFunction.h"


class QTransferFunctionPresetsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QTransferFunctionPresetsWidget(QWidget* pParent = NULL);
	~QTransferFunctionPresetsWidget(void);

private:
	void LoadPresetsFromFile(void);
	void SavePresetsToFile(void);

protected:
	QGridLayout					m_GridLayout;
	QLineEdit					m_PresetName;
	QPushButton					m_LoadPresets;
	QPushButton					m_SavePreset;
	QPushButton					m_SavePresets;
	QPushButton					m_RemovePreset;
	QListWidget					m_PresetList;
	QList<QTransferFunction>	m_TransferFunctions;
	QStandardItemModel			m_Model;
	QDomDocument				m_DOM;
};