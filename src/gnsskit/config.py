"""Typed configuration of RTKLIB rnx2rtkp: `RtkConfig` and the GNSS system masks."""
# ruff: noqa: RUF009  (`_opt(...)` in dataclass defaults is intentional)

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

# Bit values of the `pos1-navsys` mask; the mask is the sum over the chosen systems.
NAVSYS_BITS = {"G": 1, "S": 2, "R": 4, "E": 8, "J": 16, "C": 32, "I": 64}


def navsys_mask(systems: str) -> int:
    """RTKLIB navsys bitmask from system letters, e.g. 'GEC' -> 41."""
    try:
        return sum(NAVSYS_BITS[s] for s in set(systems.upper()))
    except KeyError as e:
        raise ValueError(
            f"unknown GNSS system {e}; use letters from {''.join(NAVSYS_BITS)}"
        ) from None


# Allowed values of the enum-like options (checked in RtkConfig.__post_init__)
PosMode = Literal["single", "dgps", "kinematic", "static", "static-start", "fixed"]
Frequency = Literal["l1", "l1+l2", "l1+l2+l5", "l1+l2+l5+l6"]
SolType = Literal["forward", "backward", "combined", "combined-nophasereset"]
IonoOpt = Literal[
    "off", "brdc", "sbas", "dual-freq", "est-stec", "ionex-tec", "qzs-brdc"
]
TropOpt = Literal["off", "saas", "sbas", "est-ztd", "est-ztdgrad"]
SatEph = Literal["brdc", "precise", "brdc+sbas", "brdc+ssrapc", "brdc+ssrcom"]
ArMode = Literal["off", "continuous", "instantaneous", "fix-and-hold"]
GloArMode = Literal["off", "on", "autocal", "fix-and-hold"]
PosType = Literal["llh", "xyz", "single", "posfile", "rinexhead", "rtcm"]
SolFormat = Literal["llh", "xyz", "enu", "nmea"]
TimeSys = Literal["gpst", "utc", "jst"]
OutStat = Literal["off", "state", "residual"]
TimeForm = Literal["tow", "hms"]
DegForm = Literal["deg", "dms"]
HeightType = Literal["ellipsoidal", "geodetic"]
GeoidModel = Literal["internal", "egm96", "egm08_2.5", "egm08_1", "gsi2000"]

# Tuple-valued options expand to three RTKLIB keys (ant1 = rover, ant2 = base)
Triple = tuple[float, float, float]
_ROVER_POS = ("ant1-pos1", "ant1-pos2", "ant1-pos3")
_BASE_POS = ("ant2-pos1", "ant2-pos2", "ant2-pos3")
_ROVER_ANTDEL = ("ant1-antdele", "ant1-antdeln", "ant1-antdelu")
_BASE_ANTDEL = ("ant2-antdele", "ant2-antdeln", "ant2-antdelu")


def _opt(key: str | tuple[str, ...], default: object = None) -> Any:
    """Config field mapped to one RTKLIB key, or to several for tuple values."""
    return field(default=default, metadata={"key": key})


def _literal_values(hint: Any) -> tuple[str, ...]:
    """Allowed strings of a (possibly Optional) Literal type hint; empty otherwise."""
    if get_origin(hint) is Literal:
        return get_args(hint)
    return sum((_literal_values(a) for a in get_args(hint)), ())


def _render(value: object) -> str:
    """Value as written in a .conf file (bools become on/off)."""
    return "on" if value is True else "off" if value is False else str(value)


@dataclass(frozen=True)
class RtkConfig:
    """Typed rnx2rtkp options. `None` leaves the option at RTKLIB's default (not written).

    Defaults are a static, dual-frequency, short-baseline setup (GPS+Galileo). Make
    variants with `.updated(elmask=10)`; options without a field go in `extra` as raw
    RTKLIB keys (e.g. `extra={"stats-eratio1": "300"}`) and override the fields.
    """

    # --- processing (pos1) ---
    posmode: PosMode | None = _opt("pos1-posmode", "static")
    frequency: Frequency | None = _opt("pos1-frequency", "l1+l2")
    soltype: SolType | None = _opt("pos1-soltype", "forward")
    elmask: float | None = _opt("pos1-elmask", 15.0)  # deg
    dynamics: bool | None = _opt("pos1-dynamics", False)  # receiver dynamics model
    ionoopt: IonoOpt | None = _opt("pos1-ionoopt", "brdc")
    tropopt: TropOpt | None = _opt("pos1-tropopt", "saas")
    sateph: SatEph | None = _opt("pos1-sateph", "brdc")
    sat_pcv: bool | None = _opt("pos1-posopt1", True)  # satellite antenna PCO/PCV
    rcv_pcv: bool | None = _opt("pos1-posopt2", True)  # receiver antenna PCO/PCV
    phase_windup: bool | None = _opt("pos1-posopt3")
    exclude_eclipse: bool | None = _opt("pos1-posopt4")  # drop eclipsing satellites
    exclsats: str | None = _opt("pos1-exclsats")  # e.g. "G04 +E11" (+ = force include)
    snrmask_r: bool | None = _opt("pos1-snrmask_r")  # rover SNR mask
    snrmask_b: bool | None = _opt("pos1-snrmask_b")  # base SNR mask
    # System letters (see NAVSYS_BITS) or an integer bitmask
    navsys: str | int | None = _opt("pos1-navsys", "GE")

    # --- ambiguity resolution and filter (pos2) ---
    armode: ArMode | None = _opt("pos2-armode", "continuous")
    gloarmode: GloArMode | None = _opt("pos2-gloarmode", "off")
    bdsarmode: bool | None = _opt("pos2-bdsarmode")
    arthres: float | None = _opt("pos2-arthres", 3.0)  # ratio-test threshold
    arelmask: float | None = _opt("pos2-arelmask", 15.0)  # deg, min elevation for AR
    arlockcnt: int | None = _opt("pos2-arlockcnt")
    arminfix: int | None = _opt("pos2-arminfix")  # consecutive fixes before hold
    minfixsats: int | None = _opt("pos2-minfixsats", 4)
    minholdsats: int | None = _opt("pos2-minholdsats")
    mindropsats: int | None = _opt("pos2-mindropsats")
    slipthres: float | None = _opt("pos2-slipthres")  # m, cycle-slip threshold
    maxage: float | None = _opt("pos2-maxage")  # s, max base-rover time difference
    syncsol: bool | None = _opt("pos2-syncsol")
    timeinterp: bool | None = _opt("misc-timeinterp")  # interpolate base obs to rover epochs

    # --- observation and process noise (stats) ---
    eratio1: float | None = _opt("stats-eratio1")  # code/phase error ratio, L1
    eratio2: float | None = _opt("stats-eratio2")  # code/phase error ratio, L2
    errphase: float | None = _opt("stats-errphase")  # m, phase error, constant term
    errphaseel: float | None = _opt("stats-errphaseel")  # m, elevation-dependent term
    errphasebl: float | None = _opt("stats-errphasebl")  # m/10km, baseline term
    prnbias: float | None = _opt("stats-prnbias")  # ambiguity process noise
    prniono: float | None = _opt("stats-prniono")
    prntrop: float | None = _opt("stats-prntrop")

    # --- antennas and positions ---
    rover_anttype: str | None = _opt("ant1-anttype", "*")  # "*" = from RINEX header
    base_anttype: str | None = _opt("ant2-anttype", "*")
    rover_postype: PosType | None = _opt("ant1-postype")
    rover_pos: Triple | None = _opt(_ROVER_POS)
    base_postype: PosType | None = _opt("ant2-postype", "rinexhead")
    base_pos: Triple | None = _opt(_BASE_POS)
    # Marker -> ARP offset (E, N, U) in m; "*" antenna types read it from the RINEX header
    rover_antdel: Triple | None = _opt(_ROVER_ANTDEL)
    base_antdel: Triple | None = _opt(_BASE_ANTDEL)
    satantfile: str | None = _opt("file-satantfile")  # ANTEX files
    rcvantfile: str | None = _opt("file-rcvantfile")

    # --- output ---
    solformat: SolFormat | None = _opt("out-solformat", "xyz")
    outhead: bool | None = _opt("out-outhead", True)
    outopt: bool | None = _opt("out-outopt")  # also print the options in the header
    timesys: TimeSys | None = _opt("out-timesys", "gpst")
    timeform: TimeForm | None = _opt("out-timeform")
    timendec: int | None = _opt("out-timendec")  # decimals of the time
    degform: DegForm | None = _opt("out-degform")
    height: HeightType | None = _opt("out-height")
    geoid: GeoidModel | None = _opt("out-geoid")
    outstat: OutStat | None = _opt("out-outstat")

    # --- anything else, as raw RTKLIB keys ---
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for f in fields(self):
            allowed = _literal_values(hints[f.name])
            value = getattr(self, f.name)
            if allowed and value is not None and value not in allowed:
                raise ValueError(f"{f.name}={value!r}: expected one of {allowed}")

    # --- construction ---

    @classmethod
    def empty(cls) -> RtkConfig:
        """Config with every field unset: only what is added to it is written."""
        return cls(**{f.name: None for f in fields(cls) if f.name != "extra"})

    @classmethod
    def kinematic(cls) -> RtkConfig:
        """Defaults for a moving rover: kinematic mode, forward+backward combined."""
        return cls(posmode="kinematic", soltype="combined", timeinterp=True)

    @classmethod
    def from_file(cls, path: Path | str, defaults: RtkConfig | None = None) -> RtkConfig:
        """Load an RTKLIB .conf file into `extra`, on top of `defaults` (`RtkConfig()` if None).

        Pass `RtkConfig.empty()` to get only the keys of the file.
        """
        extra = {}
        for line in Path(path).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                extra[key.strip()] = value.strip()
        base = cls() if defaults is None else defaults
        return replace(base, extra={**base.extra, **extra})

    def updated(self, **changes: Any) -> RtkConfig:
        """Copy with some fields changed."""
        return replace(self, **changes)

    def with_options(self, options: Mapping[str, object]) -> RtkConfig:
        """Copy with raw RTKLIB keys set (they override the fields)."""
        extra = {**self.extra, **{k: _render(v) for k, v in options.items()}}
        return replace(self, extra=extra)

    def with_base_llh(
        self, lat: float, lon: float, h: float, antenna_h: float | None = None
    ) -> RtkConfig:
        """Copy with the base at lat/lon [deg], ellipsoidal height [m].

        `antenna_h` [m] is the vertical offset marker -> ARP; it has no effect with
        `base_anttype="*"`, where RTKLIB reads it from the RINEX header.
        """
        cfg = replace(self, base_postype="llh", base_pos=(lat, lon, h))
        return cfg if antenna_h is None else replace(cfg, base_antdel=(0.0, 0.0, antenna_h))

    @property
    def has_base_position(self) -> bool:
        return self.base_pos is not None or all(
            k in self.extra for k in ("ant2-pos1", "ant2-pos2", "ant2-pos3")
        )

    # --- output ---

    @property
    def options(self) -> dict[str, str]:
        """RTKLIB `key -> value` strings: set fields in declaration order, then `extra`."""
        opts: dict[str, str] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "extra" or value is None:
                continue
            key = f.metadata["key"]
            if isinstance(key, tuple):
                opts.update({k: _render(v) for k, v in zip(key, value, strict=True)})
            elif key == "pos1-navsys" and isinstance(value, str):
                opts[key] = str(navsys_mask(value))
            else:
                opts[key] = _render(value)
        return {**opts, **self.extra}

    def to_text(self) -> str:
        return "".join(f"{k:<19}={v}\n" for k, v in self.options.items())

    def write(self, path: Path | str) -> Path:
        """Write the .conf file (creating parent directories) and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_text())
        return path
