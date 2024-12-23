#!/bin/bash

set -e # exit upon failure

if [ "$EUID" -ne 0 ]; then
	echo "Please run this script as root or with sudo."
	exit 1
fi

# Initialize flags
psiflow=false
gpaw=false
cp2k=false
build_sif=false
#mpi=mpich

# Parse command line options
while [[ $# -gt 0 ]]; do
	case "$1" in
	--gpaw)
		gpaw=true
		shift # Shift to next argument
		;;
	--cp2k)
		cp2k=true
		shift
		;;
	--psiflow)
		psiflow=true
		shift
		;;
	--build_sif)
		build_sif=true
		shift
		;;
	*)
		echo "Unknown option: $1"
		exit 1
		;;
	esac
done

PSIFLOW_VERSION="v4.0.0"
CCTOOLS_VERSION=7.14.0
PLUMED_VERSION=2.9.0
GPU_LIBRARIES=("rocm6.2" "cu118")

# build model
if [ "$psiflow" = "true" ]; then
	for GPU_LIBRARY in "${GPU_LIBRARIES[@]}"; do
		TAG="psiflow:${PSIFLOW_VERSION}_${GPU_LIBRARY}"
		docker build \
			--build-arg GPU_LIBRARY=${GPU_LIBRARY} \
			--build-arg PARSL_VERSION=$PARSL_VERSION \
			--build-arg PSIFLOW_VERSION=$PSIFLOW_VERSION \
			--build-arg CCTOOLS_VERSION=$CCTOOLS_VERSION \
			--build-arg PLUMED_VERSION=$PLUMED_VERSION \
			--build-arg DATE=$(date +%s) \
			-t ghcr.io/molmod/$TAG \
			-f Dockerfile . # test
		if [ "$build_sif" = "true" ]; then
			export TMPDIR=$(pwd)/tmp
			mkdir -p $TMPDIR
			apptainer build -F $TAG.sif docker-daemon:ghcr.io/molmod/$TAG
			apptainer push $TAG.sif oras://ghcr.io/molmod/$TAG
			rm $TAG.sif
			rm -rf $TMPDIR
		fi
	done
fi

if [ "$cp2k" = "true" ]; then
	TAG="cp2k:2024.1"
	docker build \
		-t ghcr.io/molmod/$TAG \
		-f Dockerfile.cp2k .
	if [ "$build_sif" = "true" ]; then
		apptainer build -F $TAG.sif docker-daemon:ghcr.io/molmod/$TAG
		apptainer push $TAG.sif oras://ghcr.io/molmod/$TAG
		rm $TAG.sif
	fi
fi

if [ "$gpaw" = "true" ]; then
	TAG="gpaw:24.1"
	sudo docker build \
		--build-arg PSIFLOW_VERSION=$PSIFLOW_VERSION \
		-t ghcr.io/molmod/$TAG \
		-f Dockerfile.gpaw .
	if [ "$build_sif" = "true" ]; then
		apptainer build -F $TAG.sif docker-daemon:ghcr.io/molmod/$TAG
		apptainer push $TAG.sif oras://ghcr.io/molmod/$TAG
		rm $TAG.sif
	fi
fi
