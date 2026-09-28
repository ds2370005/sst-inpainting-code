#!/usr/bin/env python3
"""Download hourly P-Tree L3C SST files using only the standard library.

Credentials: PTREE_FTP_USER / PTREE_FTP_PASSWORD or --user / --password.
Destination: PTREE_SST_OUTPUT or --output (default: ./data).
Example: python3 download_himawari_sst.py --output /srv/data/himawari

This is a finite batch covering all days in the configured months. Rerunning
skips existing nonempty final files; interrupted .tmp files restart from zero.
Run only one instance per output directory. FTP sends credentials unencrypted.
"""

import argparse
import calendar
from datetime import datetime
import ftplib
import logging
import math
import os
from pathlib import Path
import re
import sys
import time


DEFAULT_MONTHS = ("202505", "202506", "202604", "202605", "202606")
BASE_PATH = "/pub/himawari/L3/SST/v201_nc4_normal_std"
FILE_PATTERN = re.compile(
    r"^(\d{14})-JAXA-L3C_GHRSST-SSTskin-H\d{2}_AHI-"
    r"v[\d.]+-v[\d.]+-fv[\d.]+\.nc$"
)
LOG = logging.getLogger("himawari_sst")


class LocalWriteError(Exception):
    """A local disk error must not be retried as a network failure."""


class FTPClient:
    def __init__(self, args):
        self.args = args
        self.ftp = None

    def close(self):
        if self.ftp is not None:
            try:
                self.ftp.close()
            except OSError:
                pass
            self.ftp = None

    def connect(self):
        if self.ftp is None:
            ftp = ftplib.FTP(timeout=self.args.timeout)
            try:
                ftp.connect(self.args.host, self.args.port)
                ftp.login(self.args.user, self.args.password)
                ftp.set_pasv(True)
            except BaseException:
                ftp.close()
                raise
            self.ftp = ftp
        return self.ftp

    def retry(self, operation, label):
        # --retries counts retries after the initial attempt.
        for attempt in range(self.args.retries + 1):
            try:
                return operation(self.connect())
            except ftplib.all_errors as exc:
                self.close()
                if attempt == self.args.retries:
                    raise
                delay = min(self.args.retry_delay * 2 ** attempt, 60.0)
                LOG.warning("%s: %s; retry %d/%d in %.1fs", label,
                            type(exc).__name__, attempt + 1,
                            self.args.retries, delay)
                time.sleep(delay)

    def list_files(self, directory, day):
        def listing(ftp):
            ftp.cwd(directory)
            return ftp.nlst()

        entries = self.retry(listing, directory)
        names = set()
        for entry in entries:
            name = entry.rsplit("/", 1)[-1]
            match = FILE_PATTERN.fullmatch(name)
            if not match or not match[1].startswith(day):
                continue
            try:
                stamp = datetime.strptime(match[1], "%Y%m%d%H%M%S")
            except ValueError:
                continue
            if stamp.minute == 0 and stamp.second == 0:
                names.add(name)
        return sorted(names)

    def download(self, directory, name, destination):
        temporary = destination.with_name(destination.name + ".tmp")

        def transfer(ftp):
            ftp.cwd(directory)
            ftp.voidcmd("TYPE I")
            try:
                expected_size = ftp.size(name)
            except ftplib.error_perm as exc:
                if str(exc)[:3] not in {"500", "502", "504"}:
                    raise
                expected_size = None  # SIZE is optional on FTP servers.
            try:
                output = temporary.open("wb")
            except OSError as exc:
                raise LocalWriteError(str(exc)) from exc
            try:
                with output:
                    def write(block):
                        try:
                            output.write(block)
                        except OSError as exc:
                            raise LocalWriteError(str(exc)) from exc

                    ftp.retrbinary("RETR " + name, write, blocksize=256 * 1024)
                    try:
                        output.flush()
                        os.fsync(output.fileno())
                        actual_size = output.tell()
                    except OSError as exc:
                        raise LocalWriteError(str(exc)) from exc
            except OSError as exc:
                # Socket errors from retrbinary must reach the retry handler.
                if not output.closed:
                    output.close()
                raise
            if actual_size == 0 or (expected_size is not None
                                    and actual_size != expected_size):
                raise ftplib.error_temp("Incomplete transfer: size mismatch")
            try:
                os.replace(temporary, destination)
            except OSError as exc:
                raise LocalWriteError(str(exc)) from exc

        self.retry(transfer, name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", default=os.environ.get("PTREE_FTP_USER"))
    parser.add_argument("--password", default=os.environ.get("PTREE_FTP_PASSWORD"))
    parser.add_argument("--output", type=Path,
                        default=Path(os.environ.get("PTREE_SST_OUTPUT", "./data")))
    parser.add_argument("--host", default="ftp.ptree.jaxa.jp")
    parser.add_argument("--port", type=int, default=21)
    parser.add_argument("--base-path", default=BASE_PATH)
    parser.add_argument("--months", nargs="+", default=DEFAULT_MONTHS,
                        metavar="YYYYMM")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, choices=range(3, 6), default=4,
                        help="retries after the initial attempt (3–5; default: 4)")
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="pause after each attempted file download")
    args = parser.parse_args(argv)
    if not args.user or not args.password:
        parser.error("Set PTREE_FTP_USER and PTREE_FTP_PASSWORD or --user/--password")
    if any("\r" in value or "\n" in value
           for value in (args.user, args.password, args.base_path)):
        parser.error("Credentials and base path must not contain newlines")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if any(not math.isfinite(value) or value <= 0
           for value in (args.timeout, args.retry_delay, args.sleep)):
        parser.error("--timeout, --retry-delay and --sleep must be finite and positive")
    for month in args.months:
        try:
            if not re.fullmatch(r"\d{6}", month):
                raise ValueError
            datetime.strptime(month, "%Y%m")
        except ValueError:
            parser.error("Invalid YYYYMM month: " + month)
    args.months = sorted(set(args.months))
    args.output = args.output.expanduser()
    return args


def run(args):
    client = FTPClient(args)
    downloaded = skipped = failed = failed_days = 0
    try:
        for month in args.months:
            days = calendar.monthrange(int(month[:4]), int(month[4:]))[1]
            for day_number in range(1, days + 1):
                day = f"{day_number:02d}"
                label = f"{month}/{day}"
                directory = f"{args.base_path.rstrip('/')}/{label}"
                local = args.output / month / day
                local.mkdir(parents=True, exist_ok=True)
                try:
                    names = client.list_files(directory, month + day)
                except ftplib.all_errors as exc:
                    LOG.error("[%s] Listing failed (%s)", label, type(exc).__name__)
                    failed_days += 1
                    continue
                daily_downloaded = daily_skipped = daily_failed = 0
                if not names:
                    LOG.warning("[%s] No matching hourly SST files", label)
                    failed_days += 1
                elif len({name[:10] for name in names}) < 24:
                    LOG.warning("[%s] Fewer than 24 hourly observations listed", label)
                for name in names:
                    destination = local / name
                    if destination.is_file() and destination.stat().st_size > 0:
                        daily_skipped += 1
                    else:
                        try:
                            client.download(directory, name, destination)
                            daily_downloaded += 1
                        except ftplib.all_errors as exc:
                            LOG.error("[%s] Failed: %s (%s)", label, name,
                                      type(exc).__name__)
                            daily_failed += 1
                        finally:
                            time.sleep(args.sleep)
                    LOG.info("[%s] Total: %d, Downloaded: %d, Skipped: %d, Failed: %d",
                             label, len(names), daily_downloaded, daily_skipped, daily_failed)
                if not names:
                    LOG.info("[%s] Total: 0, Downloaded: 0, Skipped: 0", label)
                downloaded += daily_downloaded
                skipped += daily_skipped
                failed += daily_failed
    finally:
        client.close()
    LOG.info("Finished: Downloaded: %d, Skipped: %d, Failed: %d, Failed days: %d",
             downloaded, skipped, failed, failed_days)
    return 1 if failed or failed_days else 0


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        LOG.warning("Interrupted. Rerun to skip completed files and retry .tmp files.")
        return 130
    except (OSError, LocalWriteError) as exc:
        LOG.error("Local filesystem error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
