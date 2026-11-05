"""Precise Point Positioning (PPP) with RTKLIB. Placeholder: nothing is implemented yet.

Planned scope:
    - single-receiver static and kinematic solutions with rnx2rtkp (`posmode` "ppp-static",
      "ppp-kine" or "ppp-fixed"), reusing `gnsskit.rtklib` for configuration, running and
      reading the .pos output;
    - helpers for the precise products PPP needs: SP3 orbits, CLK clocks, ANTEX antenna
      models (satellite and receiver), and optionally DCB/OSB biases and ocean loading;
    - a PPP preset of `RtkConfig` (dual-frequency iono-free, `sateph="precise"`,
      `tropopt="est-ztd"`).

To do before the first function:
    - add "ppp-kine", "ppp-static" and "ppp-fixed" to `PosMode` in `gnsskit.rtklib.config`.
"""

__all__: list[str] = []
