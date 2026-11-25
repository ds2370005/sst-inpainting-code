python3 - <<'PY'
from pathlib import Path
import tarfile
import numpy as np
import eccodes as ec

root = Path("/datasets/jmagpv/jpn")
archive = next(iter(sorted(root.glob("*.tar.gz"))), None)
if archive is None:
    raise SystemExit("Archive not found")

target = "_Lz1-1000_Psbs_FD01_grib2.bin"

def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            raise RuntimeError("Unexpected end of GRIB message")
        data.extend(block)
    return bytes(data)

def get(gid, key):
    try:
        return ec.codes_get(gid, key)
    except Exception:
        return "unavailable"

print("Archive:", archive, flush=True)
print("Searching for Psbs FD01...", flush=True)

with tarfile.open(archive, "r|gz") as tar:
    for member in tar:
        if not member.isfile() or not member.name.endswith(target):
            continue

        print("File:", member.name, flush=True)
        with tar.extractfile(member) as stream:
            header = read_exact(stream, 16)
            if header[:4] != b"GRIB" or header[7] != 2:
                raise RuntimeError("Not a GRIB2 message")

            size = int.from_bytes(header[8:16], "big")
            if not 20 <= size <= 512 * 1024 * 1024:
                raise RuntimeError(f"Unexpected message size: {size}")

            message = header + read_exact(stream, size - 16)

        gid = ec.codes_new_from_message(message)
        try:
            print("\n--- Metadata ---")
            keys = [
                "centre", "tablesVersion", "localTablesVersion",
                "discipline", "parameterCategory", "parameterNumber",
                "shortName", "name", "units",
                "typeOfLevel", "level",
                "typeOfFirstFixedSurface",
                "scaleFactorOfFirstFixedSurface",
                "scaledValueOfFirstFixedSurface",
                "dataDate", "dataTime",
                "stepType", "stepUnits", "startStep", "endStep",
                "validityDate", "validityTime",
                "gridType", "Ni", "Nj",
                "iDirectionIncrementInDegrees",
                "jDirectionIncrementInDegrees",
                "iScansNegatively", "jScansPositively",
                "bitmapPresent", "numberOfMissing",
            ]
            for key in keys:
                print(f"{key}: {get(gid, key)}")

            identity = tuple(
                ec.codes_get_long(gid, key)
                for key in (
                    "discipline", "parameterCategory", "parameterNumber"
                )
            )
            if identity != (10, 4, 15):
                raise RuntimeError(f"Unexpected parameter: {identity}")

            values = np.asarray(ec.codes_get_values(gid))
            lat = np.asarray(ec.codes_get_array(gid, "latitudes"))
            lon = np.asarray(ec.codes_get_array(gid, "longitudes"))

            valid = np.isfinite(values)
            if ec.codes_get_long(gid, "bitmapPresent"):
                bitmap = np.asarray(ec.codes_get_array(gid, "bitmap"))
                valid &= bitmap.astype(bool)

            missing = ec.codes_get_double(gid, "missingValue")
            valid &= values != missing

            print("\n--- Grid / missing values ---")
            print("Latitude range:", float(lat.min()), float(lat.max()))
            print("Longitude range:", float(lon.min()), float(lon.max()))
            print("Total points:", values.size)
            print("Valid points:", int(valid.sum()))
            print("Missing points:", int((~valid).sum()))
            print(f"Missing fraction: {(~valid).mean():.2%}")

            if not valid.any():
                raise RuntimeError("No valid temperature values")

            # JMA Psbs (10, 4, 15) uses kelvin.
            kelvin = values[valid]
            celsius = kelvin - 273.15

            print("\n--- Temperature ---")
            print(f"K min / max: {kelvin.min():.3f} / {kelvin.max():.3f}")
            print(f"C min / max: {celsius.min():.3f} / {celsius.max():.3f}")
            print(f"C mean: {celsius.mean():.3f}")
            print("C percentiles [0, 1, 50, 99, 100]:")
            print(np.percentile(celsius, [0, 1, 50, 99, 100]))

            print("\n--- First 5 valid points ---")
            for i in np.flatnonzero(valid)[:5]:
                print(
                    f"lat={lat[i]:.6f}, lon={lon[i]:.6f}, "
                    f"temperature_C={values[i] - 273.15:.3f}"
                )
        finally:
            ec.codes_release(gid)
        break
    else:
        raise SystemExit("Target Psbs FD01 file not found")
PY