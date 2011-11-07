/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Logger.h"

class QTimeTableWidgetItem : public QTableWidgetItem
{
public:
	QTimeTableWidgetItem(void);

	virtual QSize sizeHint() const;
};

class QTableItemMessage : public QTableWidgetItem
{
public:
	QTableItemMessage(const QString& Message, const QLogger::MessageType& MessageType);
};

class QTableItemProgress : public QTableWidgetItem
{
public:
	QTableItemProgress(const QString& Event, const float& Progress);
};

class QLogWidget : public QTableWidget
{
    Q_OBJECT

public:
    QLogWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const;

	void SetLogger(QLogger* pLogger);

protected:
	void contextMenuEvent(QContextMenuEvent* pContextMenuEvent);

public slots:
	void OnLog(const QString& Message, const int& Type);
	void OnLog(const QString& Message, const QString& Icon);
	void OnLogProgress(const QString& Event, const float& Progress);
	void OnClear(void);
	void OnClearAll(void);

private:
	QLogger*	m_pLogger;
};