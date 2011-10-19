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

#include "HardwareWidget.h"
#include "RenderThread.h"

QCudaDevicesModel::QCudaDevicesModel(QObject *parent) :
	QAbstractTableModel(parent)
{
}

QCudaDevicesModel::QCudaDevicesModel(CudaDevices Devices, QObject *parent) :
	QAbstractTableModel(parent),
	listOfPairs(Devices)
{
}

int QCudaDevicesModel::rowCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent);
	return listOfPairs.size();
}

int QCudaDevicesModel::columnCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent);
	return 6;
}

QVariant QCudaDevicesModel::data(const QModelIndex &index, int role) const
{
	if (!index.isValid())
		return QVariant();

	if (index.row() >= listOfPairs.size() || index.row() < 0)
		return QVariant();

	if (role == Qt::DisplayRole)
	{
		QCudaDevice CudaDevice = listOfPairs.at(index.row());
		
		if (index.column() == 0)
			return QVariant(CudaDevice.m_ID);
		else if (index.column() == 1)
			return CudaDevice.m_Name;
		else if (index.column() == 2)
			return CudaDevice.m_Capability;
		else if (index.column() == 3)
			return CudaDevice.m_GlobalMemory;
		else if (index.column() == 4)
			return CudaDevice.m_NoMultiProcessors;
		else if (index.column() == 5)
			return CudaDevice.m_GpuClockSpeed;
	}

	return QVariant();
}

QVariant QCudaDevicesModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role != Qt::DisplayRole)
		return QVariant();

	if (orientation == Qt::Horizontal)
	{
		switch (section)
		{
			case 0:
				return tr("ID");

			case 1:
				return tr("Model");

			case 2:
				return tr("Capability");

			case 3:
				return tr("Global Memory");

			case 4:
				return tr("No. Multi Processors");
			
			case 5:
				return tr("CPU Clock Speed");

			default:
				return QVariant();
		}
	}

	return QVariant();
}

 bool QCudaDevicesModel::insertRows(int position, int rows, const QModelIndex &index)
 {
     Q_UNUSED(index);
     beginInsertRows(QModelIndex(), position, position+rows-1);
     endInsertRows();

     return true;
 }

 bool QCudaDevicesModel::removeRows(int position, int rows, const QModelIndex &index)
 {
     Q_UNUSED(index);
     beginRemoveRows(QModelIndex(), position, position+rows-1);

     for (int row=0; row < rows; ++row) {
         listOfPairs.removeAt(position);
     }

     endRemoveRows();
     return true;
 }

 bool QCudaDevicesModel::setData(const QModelIndex &index, const QVariant &value, int role)
 {
	return false;
 }

 Qt::ItemFlags QCudaDevicesModel::flags(const QModelIndex &index) const
 {
     if (!index.isValid())
         return Qt::ItemIsEnabled;

     return QAbstractTableModel::flags(index) | Qt::ItemIsEditable;
 }

CudaDevices QCudaDevicesModel::getList()
 {
     return listOfPairs;
 }

void QCudaDevicesModel::EnumerateDevices(void)
{
	int NoDevices = 0;

	HandleCudaError(cudaGetDeviceCount(&NoDevices), "no. Cuda capable devices");

	for (int DeviceID = 0; DeviceID < NoDevices; DeviceID++)
	{
		QCudaDevice CudaDevice;

		cudaDeviceProp DeviceProperties;

		HandleCudaError(cudaGetDeviceProperties(&DeviceProperties, DeviceID));

		CudaDevice.m_ID					= DeviceID;
		CudaDevice.m_Name				= QString::fromAscii(DeviceProperties.name);
		CudaDevice.m_Capability			= QString::number(DeviceProperties.major) + "." + QString::number(DeviceProperties.minor);
		CudaDevice.m_GlobalMemory		= QString::number((float)DeviceProperties.totalGlobalMem / powf(1024.0f, 2.0f)) + "MB";
		CudaDevice.m_NoMultiProcessors	= QString::number(DeviceProperties.multiProcessorCount);
		CudaDevice.m_GpuClockSpeed		= QString::number(DeviceProperties.clockRate * 1e-6f, 'f', 2) + "GHz";

		AddDevice(CudaDevice);
	}
}

void QCudaDevicesModel::AddDevice(const QCudaDevice& Device)
{
	beginInsertRows(QModelIndex(), 0, 0);

	listOfPairs.append(Device);

	endInsertRows();
}

bool QCudaDevicesModel::GetDevice(const int& DeviceID, QCudaDevice& CudaDevice)
{
	for (int i = 0; i < listOfPairs.size(); i++)
	{
		if (listOfPairs[i].m_ID == DeviceID)
		{
			CudaDevice = listOfPairs[i];
			return true;
		}
	}

	return false;
}

QHardwareWidget::QHardwareWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_Devices(),
	m_OptimalDevice()
{
	setTitle("Hardware Selection");
	setStatusTip("Hardware Selection");
	setToolTip("Hardware Selection");

	m_MainLayout.setColumnMinimumWidth(0, 75);
	setLayout(&m_MainLayout);

	int DriverVersion = 0, RuntimeVersion = 0; 

	cudaDriverGetVersion(&DriverVersion);
	cudaRuntimeGetVersion(&RuntimeVersion);

	QString DriverVersionString		= QString::number(DriverVersion / 1000) + "." + QString::number(DriverVersion % 100);
	QString RuntimeVersionString	= QString::number(RuntimeVersion / 1000) + "." + QString::number(RuntimeVersion % 100);

	gStatus.SetStatisticChanged("Graphics Card", "CUDA Driver Version", DriverVersionString);
	gStatus.SetStatisticChanged("Graphics Card", "CUDA Runtime Version", RuntimeVersionString);

	QString VersionInfo;

	VersionInfo += "CUDA Driver Version: " + DriverVersionString;
	VersionInfo += ", CUDA Runtime Version: " + RuntimeVersionString;

	m_MainLayout.addWidget(new QLabel(VersionInfo), 0, 0, 1, 2);
	
	m_MainLayout.addWidget(&m_Devices, 1, 0, 1, 2);

	m_Model.EnumerateDevices();

	m_Devices.horizontalHeader()->setResizeMode(QHeaderView::ResizeToContents);
	m_Devices.horizontalHeader()->setStretchLastSection(true); 
	m_Devices.horizontalHeader()->setDefaultSectionSize(1);
	m_Devices.horizontalHeader()->setDefaultAlignment(Qt::AlignLeft);
	m_Devices.horizontalHeader()->setHighlightSections(false);
	m_Devices.verticalHeader()->setVisible(false);
	m_Devices.verticalHeader()->setDefaultSectionSize(20);
	m_Devices.setSelectionMode(QAbstractItemView::SingleSelection);
	m_Devices.setSelectionBehavior(QAbstractItemView::SelectRows);
	m_Devices.setFixedHeight(75);

	m_Devices.setModel(&m_Model);

	m_OptimalDevice.setText("Optimal Device");
	m_OptimalDevice.setToolTip("Optimal Device");
	m_OptimalDevice.setStatusTip("Choose the most optimal device for rendering");
	m_OptimalDevice.setFixedWidth(90);
	m_OptimalDevice.setVisible(m_Model.rowCount(QModelIndex()) > 1);

	m_MainLayout.addWidget(&m_OptimalDevice);
	
	QObject::connect(&m_OptimalDevice, SIGNAL(clicked()), this, SLOT(OnOptimalDevice()));
	QObject::connect(&m_Devices, SIGNAL(clicked(const QModelIndex&)), this, SLOT(OnSelection(const QModelIndex&)));

	OnOptimalDevice();
}

void QHardwareWidget::OnOptimalDevice(void)
{
	const int MaxGigaFlopsDeviceID = GetMaxGigaFlopsDeviceID();

	m_Devices.selectRow(MaxGigaFlopsDeviceID);
	m_Devices.setFocus();

	OnCudaDeviceChanged();
}

void QHardwareWidget::OnSelection(const QModelIndex& ModelIndex)
{
	OnCudaDeviceChanged();
}

void QHardwareWidget::OnCudaDeviceChanged(void)
{
	QCudaDevice CudaDevice;

	if (m_Model.GetDevice(m_Devices.currentIndex().row(), CudaDevice))
	{
		gCurrentDeviceID = CudaDevice.m_ID;
		Log("Cuda device changed to " + CudaDevice.m_Name, "graphic-card");
	}
}