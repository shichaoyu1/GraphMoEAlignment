"""Compatibility import for Idea4.5 paired data."""

from glioma.data.idea4_5 import (
    AvailabilityDataset,
    INFORMATION_CONDITIONS,
    UTSWAvailabilityDataset,
    load_availability_npz,
    make_availability_splits,
    make_utsw_availability_splits,
    partition_training,
)


__all__ = [
    "AvailabilityDataset", "UTSWAvailabilityDataset", "INFORMATION_CONDITIONS",
    "load_availability_npz", "make_availability_splits", "make_utsw_availability_splits",
    "partition_training",
]
