"""Pytest configuration and compatibility shims."""

import pandas.core.dtypes.common as pd_dtypes_common

# Provide compatibility shim for third-party libraries (e.g. evidently)
# that reference infer_dtype_from_object removed in newer pandas releases.
if not hasattr(pd_dtypes_common, "infer_dtype_from_object"):
    from pandas.api.types import pandas_dtype

    def _infer_dtype_from_object(dtype):
        try:
            d = pandas_dtype(dtype)
            return getattr(d, "type", object)
        except Exception:
            return object

    pd_dtypes_common.infer_dtype_from_object = _infer_dtype_from_object
