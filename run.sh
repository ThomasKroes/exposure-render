#!/bin/bash

NC='\033[0m'
BLACK='\033[0;30m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'

echo_green()
{
  echo -e "${GREEN}$1${NC}"
}

echo_red()
{
  echo -e "${RED}$1${NC}"
}

echo_yellow()
{
  echo -e "${YELLOW}$1${NC}"
}

DetermineCpuCount()
{
  if [ -x "$(command -v nproc)" ]; then
    # Linux
    CPU_AVAIL=`nproc --all`
  elif [ -x "$(command -v sysctl)" ]; then
    # Mac
    CPU_AVAIL=`sysctl -n hw.ncpu`
  fi
  echo "Detected ${CPU_AVAIL} core(s) on this machine";
}

CheckResultAbortIfFailed()
{
    if [ $1 -eq 0 ]
    then
      echo "INFO: Step finished successfully"
    else
      echo_red "ERROR: An error occurred. aborting..."
      exit 1
    fi
}

GetCurrentDirectory() {
    SOURCE="${BASH_SOURCE[0]}"

    # resolve $SOURCE until the file is no longer a symlink
    while [ -h "$SOURCE" ]; do
      DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
      SOURCE="$(readlink "$SOURCE")"
      # if $SOURCE was a relative symlink, we need to resolve it relative to
      # the path where the symlink file was located
      [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
    done
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
}

DetermineCpuCount
GetCurrentDirectory

#######################################################################
## parameters to collect from the user
## 1. Target OS (Ubuntu, Yocto, Windows)
## 2. Build Type (Debug, Release)
#######################################################################
PROJECT_BUILD_DIR="${DIR}/build"
PROJECT_BUILD_OS="ubuntu"
PROJECT_BUILD_TYPE="release"

usage() { echo "Usage: $0 [-b <debug/release> (default=${PROJECT_BUILD_TYPE}) ] [-t <ubuntu/yocto/windows>] (default=${PROJECT_BUILD_OS})" 1>&2; exit 1; }

while getopts ":b:t:" o; do
    case "${o}" in
        b)
            PROJECT_BUILD_TYPE="${OPTARG}"
            ;;
        t)
            PROJECT_BUILD_OS="${OPTARG}"
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

# check build type
if [ "${PROJECT_BUILD_TYPE}" != "release" ] && [ "${PROJECT_BUILD_TYPE}" != "debug" ]
then
    echo_red "ERROR: Invalid build type"
    exit 1
fi

# check target os
if [ "${PROJECT_BUILD_OS}" != "ubuntu" ] && [ "${PROJECT_BUILD_OS}" != "yocto" ] && [ "${PROJECT_BUILD_OS}" != "windows" ]
then
    echo_red "ERROR: Invalid target PROJECT_BUILD_OS"
    exit 1
fi

PROJECT_BUILD_DIR="${DIR}/build-${PROJECT_BUILD_OS}-${PROJECT_BUILD_TYPE}"
echo "PROJECT_BUILD_OS=[${PROJECT_BUILD_OS}]"
echo "PROJECT_BUILD_TYPE=[${PROJECT_BUILD_TYPE}]"
echo "PROJECT_BUILD_DIR=[${PROJECT_BUILD_DIR}]"

echo "INFO: Re-create the build directory"
rm -rf ${PROJECT_BUILD_DIR}
if [ -L "${DIR}/build" ]; then
    unlink "${DIR}/build"
fi

mkdir ${PROJECT_BUILD_DIR}

#######################################################################
# use the cross toolchain in case we are building for yocto
#######################################################################
if [ "${MARS_BUILD_OS}" == "yocto" ]
then
    YOCTO_CROSS_ENV="/local_tools_linux/linux-distros/linux-for-mvp/environment-setup-corei7-64-poky-linux"
    if [ ! -f ${YOCTO_CROSS_ENV} ]; then
        echo_red "ERROR: Yocto cross toolchain is not found: path = [${YOCTO_CROSS_ENV}]"
        exit 1
    fi
    source ${YOCTO_CROSS_ENV}
fi

# build
if [ "${PROJECT_BUILD_OS}" == "ubuntu" ] || [ "${PROJECT_BUILD_OS}" == "yocto" ]
then
    pushd ${PROJECT_BUILD_DIR}
      echo_green "build for [${PROJECT_BUILD_OS}]"
      cmake -DCMAKE_BUILD_TYPE=${PROJECT_BUILD_TYPE} -DCUDA_TOOLKIT_ROOT_DIR="/devsystem/dependencies/ubuntu16/cuda/cuda-8.0/" ../Source
      make -j $CPU_AVAIL
    popd
elif [ "${PROJECT_BUILD_OS}" == "windows" ]
then
    pushd ${PROJECT_BUILD_DIR}    

    echo_green "build for [${PROJECT_BUILD_OS}]"
    cmake -DCMAKE_BUILD_TYPE=${PROJECT_BUILD_TYPE} -G "Visual Studio 14 2015 Win64" ..
    CheckResultAbortIfFailed $?

    echo_green "Add MSBuild to path"
    export PATH="C:\Program Files (x86)\MSBuild\14.0\Bin":$PATH
    MSBuild.exe *.sln /maxcpucount:${CPU_CORES_COUNT}
    CheckResultAbortIfFailed $?

    popd
else
    echo_red "ERROR: unsupported PROJECT_BUILD_OS ${PROJECT_BUILD_OS}"
    exit 1
fi

