#pragma once

#include <QtGui>

#include "PresetsWidget.h"

class QTransferFunctionPresetsWidget : public QPresetsWidget
{
    Q_OBJECT

public:
    QTransferFunctionPresetsWidget(QWidget* pParent = NULL);
	
	void LoadPresets(QDomElement& Root);
	void SavePresets(QDomDocument& DomDoc, QDomElement& Root);
	void LoadPreset(QPresetXML* pPreset);
	void SavePreset(const QString& Name);

protected slots:

protected:
};