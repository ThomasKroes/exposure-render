
#include "LightingPresetsWidget.h"
#include "RenderThread.h"

QLightingPresetsWidget::QLightingPresetsWidget(QWidget* pParent) :
	QPresetsWidget("LightingPresets.xml", pParent)
{
	// Title, status and tooltip
	setTitle("Presets");
	setToolTip("Lighting Presets");
	setStatusTip("Lighting Presets");
}

void QLightingPresetsWidget::LoadPresets(QDomElement& Root)
{
	QPresetsWidget::LoadPresets(Root);

	/*
	QDomNodeList Presets = DomRoot.elementsByTagName("Preset");

	for (int i = 0; i < Presets.count(); i++)
	{
		QDomNode TransferFunctionNode = Presets.item(i);

		// Create new transfer function
		QTransferFunction* pTransferFunction = new QTransferFunction();

		// Append the transfer function
		m_TransferFunctions.append(pTransferFunction);

		// Load the preset into it
		m_TransferFunctions.back()->ReadXML(TransferFunctionNode.toElement());
	}
	*/
}

void QLightingPresetsWidget::SavePresets(QDomDocument& DomDoc, QDomElement& Root)
{
	QPresetsWidget::SavePresets(DomDoc, Root);

	/*
	QDomElement Presets = DOM.createElement("Presets");

	for (int i = 0; i < m_Presets.size(); i++)
	{
		m_Presets[i]->WriteXML(DOM, Presets);
	}

	DOM.appendChild(Presets);
	*/
}