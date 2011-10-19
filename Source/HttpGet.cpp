/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "HttpGet.h"

QHttpGet::QHttpGet(QObject* pParent /*= NULL*/) :
	QObject(pParent)
{
    connect(&m_Http, SIGNAL(done(bool)), this, SLOT(HttpDone(bool)));
}

bool QHttpGet::GetFile(const QUrl& Url, const QString& FilePath)
{
    if (!Url.isValid())
	{
        Log("Error: Invalid URL", "globe");
        return false;
    }

    if (Url.scheme() != "http")
	{
        Log("Error: URL must start with 'http:'", "globe");
        return false;
    }

    if (Url.path().isEmpty())
	{
        Log("Error: URL has no path", "globe");
        return false;
    }

    m_File.setFileName(FilePath);

    if (!m_File.open(QIODevice::WriteOnly))
	{
        Log("Error: Cannot write m_File ", "globe");
        return false;
    }

    m_Http.setHost(Url.host(), Url.port(80));
    m_Http.get(Url.path(), &m_File);
    m_Http.close();

    return true;
}

void QHttpGet::HttpDone(bool Error)
{
    if (Error)
		Log("Error: " + QString(m_Http.errorString()), QLogger::Critical);

    m_File.close();
    emit done();
}