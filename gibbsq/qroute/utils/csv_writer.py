"""
Universal CSV data writer for GibbsQ experiments.

This module provides a robust, type-safe CSV writer that serves as the
single data-output mechanism for all z2 thesis experiments. It enforces
schema validation, atomic writes, and automatic metadata logging.

Design principles:
    - Experiments produce DATA (CSV), never figures.
    - Figures are generated separately from CSV artifacts.
    - Every CSV gets a companion .meta.json sidecar.
    - Numpy types are coerced transparently.
    - Writes are atomic (temp file + rename) to prevent corruption.

Usage example::

    from gibbsq.qroute.utils.csv_writer import ExperimentCSVWriter, Column

    writer = ExperimentCSVWriter(
        experiment_name="boundary_equilibrium_verification",
        output_dir="outputs/data",
        columns=[
            Column("system_id", str, "Benchmark system identifier"),
            Column("K_star", float, "Scalar equilibrium parameter"),
            Column("max_discrepancy", float, "Max absolute error"),
        ],
        metadata={"hypothesis": "H2"},
    )
    writer.write_row({"system_id": "bench10", "K_star": 2.96e-4, "max_discrepancy": 1e-14})
    writer.finalize()

References:
    - z2/13_thesis_evidence_requirements.md (output criteria)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "Column",
    "ExperimentCSVWriter",
]


@dataclass(frozen=True)
class Column:
    """Schema definition for a single CSV column.

    Parameters
    ----------
    name : str
        Column name.  Must be a valid Python identifier and unique within
        the writer's column list.
    dtype : type
        Expected Python type (``str``, ``int``, ``float``, ``bool``).
        Used for validation and coercion of numpy scalars.
    description : str
        Human-readable description for documentation and metadata sidecar.
    """

    name: str
    dtype: type
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Column name must be a non-empty string.")
        if not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Column name '{self.name}' contains invalid characters.  "
                f"Use only alphanumeric characters, underscores, and hyphens."
            )
        _ALLOWED_TYPES = (str, int, float, bool)
        if self.dtype not in _ALLOWED_TYPES:
            raise ValueError(
                f"Column '{self.name}' has unsupported dtype {self.dtype.__name__}.  "
                f"Allowed types: {', '.join(t.__name__ for t in _ALLOWED_TYPES)}."
            )


def _coerce_value(value: Any, col: Column) -> Any:
    """Coerce a value to the column's declared type.

    Handles numpy scalar types transparently so that experiment code does
    not need to manually call ``.item()`` or cast.

    Parameters
    ----------
    value : Any
        Raw value from the experiment.
    col : Column
        Column schema used for type validation and coercion.

    Returns
    -------
    Any
        Python-native value suitable for CSV serialization.

    Raises
    ------
    TypeError
        If the value cannot be coerced to the declared type.
    ValueError
        If a float value is NaN or Inf and the column does not allow it.
    """
    if value is None:
        return ""

    # Numpy scalar → Python native
    if isinstance(value, np.generic):
        value = value.item()

    # Numpy array → JSON string for storage
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())

    # Bool check must come before int because bool is a subtype of int
    if col.dtype is bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in ("true", "1", "yes"):
                return True
            if lower in ("false", "0", "no"):
                return False
        raise TypeError(
            f"Column '{col.name}' expects bool, got {type(value).__name__}: {value!r}"
        )

    if col.dtype is int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not value.is_integer():
                raise TypeError(
                    f"Column '{col.name}' expects int, got non-integer float: {value}"
                )
            return int(value)
        raise TypeError(
            f"Column '{col.name}' expects int, got {type(value).__name__}: {value!r}"
        )

    if col.dtype is float:
        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError(
            f"Column '{col.name}' expects float, got {type(value).__name__}: {value!r}"
        )

    if col.dtype is str:
        return str(value)

    return value


def _generate_run_id() -> str:
    """Generate a timestamp-based run identifier."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class ExperimentCSVWriter:
    """Thread-safe, atomic CSV writer for experiment data.

    This writer is the canonical data-output mechanism for all z2
    experiments.  It enforces a declared column schema, coerces numpy
    types, writes atomically, and produces a metadata sidecar.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (used in filenames).
    output_dir : str or Path
        Root directory for CSV output files.
    columns : list of Column
        Ordered column schema.  Every ``write_row`` call must supply
        values for exactly these columns.
    metadata : dict, optional
        Arbitrary metadata to include in the sidecar JSON file.
    run_id : str, optional
        Explicit run identifier.  If not provided, a timestamp-based
        ID is generated automatically.
    include_timestamp_column : bool
        If True (default), a ``_timestamp`` column is prepended
        automatically to every row.
    include_run_id_column : bool
        If True (default), a ``_run_id`` column is prepended
        automatically to every row.

    Attributes
    ----------
    csv_path : Path
        Absolute path to the output CSV file.
    meta_path : Path
        Absolute path to the companion metadata JSON file.
    row_count : int
        Number of data rows written so far.
    """

    experiment_name: str
    output_dir: str | Path
    columns: list[Column]
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    include_timestamp_column: bool = True
    include_run_id_column: bool = True

    # Internal state (not part of the constructor signature)
    _csv_path: Path = field(init=False, repr=False)
    _meta_path: Path = field(init=False, repr=False)
    _temp_path: Path = field(init=False, repr=False)
    _row_count: int = field(init=False, default=0, repr=False)
    _writer: csv.writer = field(init=False, repr=False)  # type: ignore[type-arg]
    _file_handle: io.TextIOWrapper = field(init=False, repr=False)
    _finalized: bool = field(init=False, default=False, repr=False)
    _header: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must be a non-empty string.")

        # Validate column uniqueness
        col_names = [c.name for c in self.columns]
        if len(col_names) != len(set(col_names)):
            duplicates = [n for n in col_names if col_names.count(n) > 1]
            raise ValueError(
                f"Duplicate column names detected: {sorted(set(duplicates))}"
            )

        if not self.run_id:
            self.run_id = _generate_run_id()

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.experiment_name}_{self.run_id}.csv"
        self._csv_path = output_dir / filename
        self._meta_path = output_dir / f"{self.experiment_name}_{self.run_id}.meta.json"

        # Build header with optional system columns
        self._header = []
        if self.include_run_id_column:
            self._header.append("_run_id")
        if self.include_timestamp_column:
            self._header.append("_timestamp")
        self._header.extend(c.name for c in self.columns)

        # Open temp file for atomic writes
        fd, temp_name = tempfile.mkstemp(
            suffix=".csv.tmp",
            prefix=f"{self.experiment_name}_",
            dir=str(output_dir),
        )
        os.close(fd)
        self._temp_path = Path(temp_name)
        self._file_handle = open(self._temp_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file_handle, quoting=csv.QUOTE_MINIMAL)
        self._writer.writerow(self._header)
        self._file_handle.flush()

        log.info(
            "ExperimentCSVWriter initialized: experiment=%s, run_id=%s, columns=%d, path=%s",
            self.experiment_name,
            self.run_id,
            len(self.columns),
            self._csv_path,
        )

    @property
    def csv_path(self) -> Path:
        """Absolute path to the output CSV file."""
        return self._csv_path

    @property
    def meta_path(self) -> Path:
        """Absolute path to the companion metadata JSON file."""
        return self._meta_path

    @property
    def row_count(self) -> int:
        """Number of data rows written so far."""
        return self._row_count

    def write_row(self, data: dict[str, Any]) -> None:
        """Write a single data row to the CSV file.

        Parameters
        ----------
        data : dict
            Mapping of column names to values.  Must contain exactly the
            columns declared in the schema (system columns like
            ``_run_id`` and ``_timestamp`` are added automatically).

        Raises
        ------
        RuntimeError
            If the writer has already been finalized.
        KeyError
            If a declared column is missing from ``data``.
        TypeError
            If a value cannot be coerced to the declared column type.
        """
        if self._finalized:
            raise RuntimeError(
                f"Cannot write to finalized writer for experiment '{self.experiment_name}'."
            )

        # Validate all declared columns are present
        declared_names = {c.name for c in self.columns}
        provided_names = set(data.keys())
        missing = declared_names - provided_names
        if missing:
            raise KeyError(
                f"Missing columns in write_row: {sorted(missing)}.  "
                f"Expected: {sorted(declared_names)}."
            )
        extra = provided_names - declared_names
        if extra:
            log.warning(
                "Extra columns ignored in write_row for '%s': %s",
                self.experiment_name,
                sorted(extra),
            )

        # Build row
        row: list[Any] = []
        if self.include_run_id_column:
            row.append(self.run_id)
        if self.include_timestamp_column:
            row.append(datetime.now(tz=timezone.utc).isoformat())

        col_map = {c.name: c for c in self.columns}
        for col_name in (c.name for c in self.columns):
            col = col_map[col_name]
            raw_value = data[col_name]
            coerced = _coerce_value(raw_value, col)
            row.append(coerced)

        self._writer.writerow(row)
        self._file_handle.flush()
        self._row_count += 1

    def write_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        """Write multiple data rows to the CSV file.

        Parameters
        ----------
        rows : sequence of dict
            Each dict maps column names to values, same as ``write_row``.
        """
        for row_data in rows:
            self.write_row(row_data)

    def finalize(self) -> Path:
        """Finalize the CSV writer.

        Flushes all pending data, atomically renames the temp file to the
        final CSV path, and writes the metadata sidecar JSON.

        Returns
        -------
        Path
            Absolute path to the finalized CSV file.

        Raises
        ------
        RuntimeError
            If the writer has already been finalized.
        """
        if self._finalized:
            raise RuntimeError(
                f"Writer for experiment '{self.experiment_name}' has already been finalized."
            )

        self._file_handle.flush()
        self._file_handle.close()

        # Atomic rename: temp → final
        shutil.move(str(self._temp_path), str(self._csv_path))

        # Write metadata sidecar
        sidecar = {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "csv_file": self._csv_path.name,
            "row_count": self._row_count,
            "columns": [asdict(c) for c in self.columns],
            "column_dtypes": {c.name: c.dtype.__name__ for c in self.columns},
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "metadata": self.metadata,
        }
        # Serialize Column dtype fields (type objects are not JSON-serializable)
        for col_entry in sidecar["columns"]:
            col_entry["dtype"] = col_entry["dtype"].__name__

        self._meta_path.write_text(
            json.dumps(sidecar, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        self._finalized = True
        log.info(
            "ExperimentCSVWriter finalized: experiment=%s, rows=%d, path=%s",
            self.experiment_name,
            self._row_count,
            self._csv_path,
        )
        return self._csv_path

    def __enter__(self) -> ExperimentCSVWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self._finalized:
            if exc_type is None:
                self.finalize()
            else:
                # On exception, clean up the temp file
                self._file_handle.close()
                if self._temp_path.exists():
                    self._temp_path.unlink()
                log.warning(
                    "ExperimentCSVWriter aborted due to exception: experiment=%s",
                    self.experiment_name,
                )

    def __del__(self) -> None:
        if not self._finalized and hasattr(self, "_file_handle"):
            try:
                if not self._file_handle.closed:
                    self._file_handle.close()
                if hasattr(self, "_temp_path") and self._temp_path.exists():
                    self._temp_path.unlink()
            except Exception:
                pass
