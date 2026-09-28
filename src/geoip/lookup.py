"""
Offline GeoIP/ASN enrichment.

This NEVER calls any online IP lookup API. If a local MaxMind-style .mmdb database
(e.g. GeoLite2-Country.mmdb / GeoLite2-ASN.mmdb) is found under data/geoip/, and the
optional `geoip2` package is installed, it is used to enrich/cross-check records.
Otherwise, the module gracefully falls back to the dataset's own geo_country / asn
fields, which is the default behavior for the TRACE-X synthetic dataset (it has no
raw IP-to-geo database attached).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class OfflineGeoIPResolver:
    def __init__(self, geoip_dir: Path, country_db_filename: str, asn_db_filename: str):
        self.geoip_dir = Path(geoip_dir)
        self.country_db_path = self.geoip_dir / country_db_filename
        self.asn_db_path = self.geoip_dir / asn_db_filename
        self.country_reader = None
        self.asn_reader = None
        self.available = False
        self._try_load()

    def _try_load(self):
        if not (self.country_db_path.exists() or self.asn_db_path.exists()):
            return
        try:
            import geoip2.database  # optional dependency

            if self.country_db_path.exists():
                self.country_reader = geoip2.database.Reader(str(self.country_db_path))
            if self.asn_db_path.exists():
                self.asn_reader = geoip2.database.Reader(str(self.asn_db_path))
            self.available = self.country_reader is not None or self.asn_reader is not None
        except Exception:
            # geoip2 not installed, or DB unreadable/corrupt -> fall back gracefully.
            self.available = False

    def lookup_country(self, ip: str) -> Optional[str]:
        if not self.available or self.country_reader is None:
            return None
        try:
            resp = self.country_reader.country(ip)
            return resp.country.iso_code
        except Exception:
            return None

    def lookup_asn(self, ip: str) -> Optional[str]:
        if not self.available or self.asn_reader is None:
            return None
        try:
            resp = self.asn_reader.asn(ip)
            return f"AS{resp.autonomous_system_number}"
        except Exception:
            return None

    def close(self):
        if self.country_reader:
            self.country_reader.close()
        if self.asn_reader:
            self.asn_reader.close()


def enrich_geo_asn(df: pd.DataFrame, resolver: OfflineGeoIPResolver) -> pd.DataFrame:
    """Fill/cross-check geo_country and asn using the offline DB where available;
    otherwise leave the dataset's own fields untouched."""
    out = df.copy()
    if not resolver.available:
        return out  # graceful fallback: dataset fields are already present

    # Look each distinct IP up once (same result as a per-row lookup, far fewer DB reads).
    unique_ips = pd.unique(out["src_ip"])
    country_by_ip = {ip: resolver.lookup_country(ip) for ip in unique_ips}
    asn_by_ip = {ip: resolver.lookup_asn(ip) for ip in unique_ips}

    out["geo_country"] = [country_by_ip[ip] or orig for ip, orig in zip(out["src_ip"], out["geo_country"])]
    out["asn"] = [asn_by_ip[ip] or orig for ip, orig in zip(out["src_ip"], out["asn"])]
    return out
