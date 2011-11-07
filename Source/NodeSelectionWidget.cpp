/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "NodeSelectionWidget.h"
#include "TransferFunction.h"

QNodeSelectionWidget::QNodeSelectionWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_NodeSelection(),
	m_FirstNode(),
	m_PreviousNode(),
	m_NextNode(),
	m_LastNode(),
	m_DeleteNode()
{
	setTitle("Node Selection");
	setToolTip("Node Selection");
	setStatusTip("Node Selection");

	m_MainLayout.setAlignment(Qt::AlignTop);

	setLayout(&m_MainLayout);

	setAlignment(Qt::AlignTop);

	m_NodeSelection.setStatusTip("Node selection");
	m_NodeSelection.setToolTip("Node selection");
	m_MainLayout.addWidget(&m_NodeSelection, 0, 0);

	QObject::connect(&m_NodeSelection, SIGNAL(currentIndexChanged(int)), this, SLOT(OnNodeSelectionChanged(int)));

	// First node
	m_FirstNode.setIcon(GetIcon("control-stop-180"));
	m_FirstNode.setStatusTip("Select first node");
	m_FirstNode.setToolTip("Select first node");
	m_FirstNode.setFixedWidth(22);
	m_FirstNode.setFixedHeight(22);
	m_FirstNode.updateGeometry();
	m_MainLayout.addWidget(&m_FirstNode, 0, 2);

	QObject::connect(&m_FirstNode, SIGNAL(pressed()), this, SLOT(OnFirstNode()));

	// Previous node
	m_PreviousNode.setIcon(GetIcon("control-180"));
	m_PreviousNode.setStatusTip("Select previous node"); 
	m_PreviousNode.setToolTip("Select previous node");
	m_PreviousNode.setFixedWidth(22);
	m_PreviousNode.setFixedHeight(22);
	m_PreviousNode.updateGeometry();
	m_MainLayout.addWidget(&m_PreviousNode, 0, 3);

	QObject::connect(&m_PreviousNode, SIGNAL(pressed()), this, SLOT(OnPreviousNode()));

	// Next node
	m_NextNode.setIcon(GetIcon("control"));
	m_NextNode.setStatusTip("Select next node");
	m_NextNode.setToolTip("Select next node");
	m_NextNode.setFixedWidth(20);
	m_NextNode.setFixedHeight(20);
	m_MainLayout.addWidget(&m_NextNode, 0, 4);
	
	QObject::connect(&m_NextNode, SIGNAL(pressed()), this, SLOT(OnNextNode()));

	// Last node
	m_LastNode.setIcon(GetIcon("control-stop"));
	m_LastNode.setStatusTip("Select Last node");
	m_LastNode.setToolTip("Select Last node");
	m_LastNode.setFixedWidth(22);
	m_LastNode.setFixedHeight(22);
	m_LastNode.updateGeometry();
	m_MainLayout.addWidget(&m_LastNode, 0, 5);

	QObject::connect(&m_LastNode, SIGNAL(pressed()), this, SLOT(OnLastNode()));

	// Delete node
	m_DeleteNode.setIcon(GetIcon("cross"));
	m_DeleteNode.setStatusTip("Delete selected node");
	m_DeleteNode.setToolTip("Delete selected node");
	m_DeleteNode.setFixedWidth(20);
	m_DeleteNode.setFixedHeight(20);
	m_MainLayout.addWidget(&m_DeleteNode, 0, 6);
	
	QObject::connect(&m_DeleteNode, SIGNAL(pressed()), this, SLOT(OnDeleteNode()));

	QObject::connect(&gTransferFunction, SIGNAL(SelectionChanged(QNode*)), this, SLOT(OnNodeSelectionChanged(QNode*)));

	SetupSelectionUI();

	OnNodeSelectionChanged(NULL);
}

void QNodeSelectionWidget::OnNodeSelectionChanged(QNode* pNode)
{
	SetupSelectionUI();
}

void QNodeSelectionWidget::OnNodeSelectionChanged(const int& Index)
{
	gTransferFunction.SetSelectedNode(Index);
	SetupSelectionUI();
}

void QNodeSelectionWidget::OnFirstNode(void)
{
	gTransferFunction.SelectFirstNode();
}

void QNodeSelectionWidget::OnPreviousNode(void)
{
	gTransferFunction.SelectPreviousNode();
}

void QNodeSelectionWidget::OnNextNode(void)
{
	gTransferFunction.SelectNextNode();
}

void QNodeSelectionWidget::OnLastNode(void)
{
	gTransferFunction.SelectLastNode();
}

void QNodeSelectionWidget::OnDeleteNode(void)
{
	if (!gTransferFunction.GetSelectedNode())
		return;

	gTransferFunction.RemoveNode(gTransferFunction.GetSelectedNode());
}

void QNodeSelectionWidget::SetupSelectionUI(void)
{
	QNode* pNode = gTransferFunction.GetSelectedNode();

	if (pNode)
	{
		// Obtain current node index
		const int NodeIndex = gTransferFunction.GetNodes().indexOf(*gTransferFunction.GetSelectedNode());

		// Prevent circular dependency
		m_NodeSelection.blockSignals(true);

		m_NodeSelection.clear();

		for (int i = 0; i < gTransferFunction.GetNodes().size(); i++)
			m_NodeSelection.addItem("Node " + QString::number(i + 1));

		// Reflect node selection change in node selection combo box
		m_NodeSelection.setCurrentIndex(NodeIndex);

		m_NodeSelection.blockSignals(false);

		// Decide whether to enable/disable UI items
		const bool EnablePrevious	= NodeIndex > 0;
		const bool EnableNext		= NodeIndex < gTransferFunction.GetNodes().size() - 1;
		const bool EnableDelete		= gTransferFunction.GetSelectedNode() ? (NodeIndex != 0 && NodeIndex != gTransferFunction.GetNodes().size() - 1) : false;

		// Enable/disable buttons
		m_PreviousNode.setEnabled(EnablePrevious);
		m_NextNode.setEnabled(EnableNext);
		m_DeleteNode.setEnabled(EnableDelete);

		// Create tooltip strings
		QString PreviousToolTip = EnablePrevious ? "Select node " + QString::number(NodeIndex) : "No previous node";
		QString NextToolTip		= EnableNext ? "Select node " + QString::number(NodeIndex + 2) : "No next node";

		// Update push button tooltips
		m_PreviousNode.setStatusTip(PreviousToolTip);
		m_PreviousNode.setToolTip(PreviousToolTip);
		m_NextNode.setStatusTip(NextToolTip);
		m_NextNode.setToolTip(NextToolTip);
	}
}