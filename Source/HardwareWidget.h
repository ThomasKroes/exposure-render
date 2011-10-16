/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

class QCudaDevice
{
public:
	QCudaDevice(void){};

	QCudaDevice& QCudaDevice::operator=(const QCudaDevice& Other)
	{
		m_ID				= Other.m_ID;
		m_Name				= Other.m_Name;
		m_Capability		= Other.m_Capability;
		m_GlobalMemory		= Other.m_GlobalMemory;
		m_MemoryClockRate	= Other.m_MemoryClockRate;
		m_NoMultiProcessors	= Other.m_NoMultiProcessors;
		m_GpuClockSpeed		= Other.m_GpuClockSpeed;
		m_RegistersPerBlock	= Other.m_RegistersPerBlock;

		return *this;
	}

	int			m_ID;
	QString		m_Name;
	QString		m_Capability;
	QString		m_GlobalMemory;
	QString		m_MemoryClockRate;
	QString		m_NoMultiProcessors;
	QString		m_GpuClockSpeed;
	QString		m_RegistersPerBlock;
};

typedef QList<QCudaDevice> CudaDevices;

class QCudaDevicesModel : public QAbstractTableModel
{
     Q_OBJECT

 public:
     QCudaDevicesModel(QObject* pParent = NULL);
     QCudaDevicesModel(CudaDevices Devices, QObject* pParent = NULL);

     int rowCount(const QModelIndex &parent) const;
     int columnCount(const QModelIndex &parent) const;
     QVariant data(const QModelIndex &index, int role) const;
     QVariant headerData(int section, Qt::Orientation orientation, int role) const;
     Qt::ItemFlags flags(const QModelIndex &index) const;
     bool setData(const QModelIndex &index, const QVariant &value, int role=Qt::EditRole);
     bool insertRows(int position, int rows, const QModelIndex &index=QModelIndex());
     bool removeRows(int position, int rows, const QModelIndex &index=QModelIndex());

     CudaDevices getList();

	 void EnumerateDevices(void);
	 void AddDevice(const QCudaDevice& Device);
	 bool GetDevice(const int& DeviceID, QCudaDevice& CudaDevice);

 private:
     CudaDevices listOfPairs;
 };

class QHardwareWidget : public QGroupBox
{
    Q_OBJECT

public:
    QHardwareWidget(QWidget* pParent = NULL);

private slots:
	void OnOptimalDevice(void);
	void OnSelection(const QModelIndex& ModelIndex);
	void OnCudaDeviceChanged(void);

private:
	QGridLayout			m_MainLayout;
	QCudaDevicesModel	m_Model;
	QTableView			m_Devices;
	QPushButton			m_OptimalDevice;
};