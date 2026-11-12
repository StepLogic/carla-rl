
### ViNT-native topomap vs ViNT mapless, paired (same episodes)
| Town | Band | n | SR@1m Topomap | SR@1m Mapless | Topomap-only / Mapless-only | p (McNemar, Holm) | SR@2m Topomap | SR@2m Mapless | 2 m: Topomap-only / Mapless-only | p 2 m (Holm) | SPL Topomap | SPL Mapless |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Town02 | easy | 100 | 0.180 | 0.290 | 7 / 18 | 0.2597 | 0.370 | 0.460 | 14 / 23 | 0.5632 | 0.180 | 0.290 |
| Town02 | medium | 100 | 0.150 | 0.070 | 12 / 4 | 0.3841 | 0.340 | 0.160 | 27 / 9 | 0.0197 | 0.150 | 0.070 |
| Town02 | hard | 100 | 0.030 | 0.000 | 3 / 0 | 1.0000 | 0.160 | 0.000 | 16 / 0 | 0.0002 | 0.030 | 0.000 |
| Town03 | easy | 100 | 0.390 | 0.340 | 15 / 10 | 1.0000 | 0.590 | 0.550 | 13 / 9 | 0.9463 | 0.389 | 0.340 |
| Town03 | medium | 100 | 0.130 | 0.150 | 10 / 12 | 1.0000 | 0.240 | 0.190 | 18 / 13 | 0.9463 | 0.130 | 0.150 |
| Town03 | hard | 100 | 0.030 | 0.020 | 2 / 1 | 1.0000 | 0.090 | 0.020 | 8 / 1 | 0.1562 | 0.030 | 0.020 |
| Town02 | pooled | 300 | 0.120 | 0.120 | 22 / 22 | 1.0000 (raw) | 0.290 | 0.207 | 57 / 32 | 0.0105 (raw) | 0.120 | 0.120 |
| Town03 | pooled | 300 | 0.183 | 0.170 | 27 / 23 | 0.6718 (raw) | 0.307 | 0.253 | 39 / 23 | 0.0559 (raw) | 0.183 | 0.170 |
| All | pooled | 600 | 0.152 | 0.145 | 49 / 45 | 0.7572 (raw) | 0.298 | 0.230 | 96 / 55 | 0.0011 (raw) | 0.151 | 0.145 |

### Policy comparison: ViNT mapless vs c0 mapless, paired (same episodes)
| Town | Band | n | SR@1m ViNT | SR@1m c0 | ViNT-only / c0-only | p (McNemar, Holm) | SR@2m ViNT | SR@2m c0 |
|---|---|---|---|---|---|---|---|---|
| Town02 | easy | 100 | 0.290 | 0.730 | 3 / 47 | 1.483e-10 | 0.460 | 0.760 |
| Town02 | medium | 100 | 0.070 | 0.680 | 2 / 63 | 6.98e-16 | 0.160 | 0.690 |
| Town02 | hard | 100 | 0.000 | 0.450 | 0 / 45 | 2.842e-13 | 0.000 | 0.460 |
| Town03 | easy | 100 | 0.340 | 0.670 | 5 / 38 | 2.5e-07 | 0.550 | 0.690 |
| Town03 | medium | 100 | 0.150 | 0.570 | 4 / 46 | 8.924e-10 | 0.190 | 0.590 |
| Town03 | hard | 100 | 0.020 | 0.390 | 1 / 38 | 4.366e-10 | 0.020 | 0.400 |
| All | pooled | 600 | 0.145 | 0.582 | 15 / 277 | 1.348e-63 (raw) | 0.230 | 0.598 |

### Topomap arm: fallbacks and PRIVILEGED diagnostics (reporting only)
| Town | Band | n | No route | Same node | Start node err med (m) | Start <=8 m | Goal node err med (m) | Goal <=8 m | Final leg reached | Routed fails | Ended >5 m off route | Navigator-vs-true route gap med (nodes) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Town02 | easy | 100 | 0 | 0 | 2.3 | 0.96 | 1.3 | 0.91 | 0.58 | 82 | 37 | 0.0 |
| Town02 | medium | 100 | 0 | 0 | 2.5 | 0.94 | 0.9 | 0.94 | 0.43 | 85 | 27 | 0.0 |
| Town02 | hard | 100 | 0 | 0 | 1.3 | 0.99 | 0.7 | 1.00 | 0.20 | 97 | 17 | 1.0 |
| Town03 | easy | 100 | 0 | 0 | 1.2 | 0.99 | 0.8 | 0.98 | 0.81 | 61 | 28 | 0.0 |
| Town03 | medium | 100 | 0 | 0 | 1.2 | 0.97 | 0.7 | 0.94 | 0.40 | 87 | 22 | 1.0 |
| Town03 | hard | 100 | 0 | 0 | 1.3 | 0.96 | 0.7 | 0.93 | 0.17 | 97 | 14 | 1.0 |

### Speed and end mix
| Town | Band | Arm | km/h | Passed within 2 m (any speed) | Reached | Crossing | Off road | Collision | Clock/other |
|---|---|---|---|---|---|---|---|---|---|
| Town02 | easy | Topomap | 11.7 | 0.57 | 18 | 56 | 15 | 11 | 0 |
| Town02 | easy | Mapless | 9.7 | 0.47 | 29 | 51 | 18 | 2 | 0 |
| Town02 | medium | Topomap | 12.3 | 0.43 | 15 | 52 | 23 | 10 | 0 |
| Town02 | medium | Mapless | 11.0 | 0.16 | 7 | 65 | 19 | 9 | 0 |
| Town02 | hard | Topomap | 12.5 | 0.19 | 3 | 62 | 30 | 5 | 0 |
| Town02 | hard | Mapless | 11.4 | 0.00 | 0 | 66 | 33 | 1 | 0 |
| Town03 | easy | Topomap | 10.0 | 0.59 | 39 | 39 | 21 | 1 | 0 |
| Town03 | easy | Mapless | 8.3 | 0.56 | 34 | 41 | 25 | 0 | 0 |
| Town03 | medium | Topomap | 11.8 | 0.24 | 13 | 61 | 26 | 0 | 0 |
| Town03 | medium | Mapless | 10.4 | 0.20 | 15 | 61 | 23 | 1 | 0 |
| Town03 | hard | Topomap | 12.4 | 0.11 | 3 | 80 | 17 | 0 | 0 |
| Town03 | hard | Mapless | 11.5 | 0.02 | 2 | 64 | 31 | 3 | 0 |

### Getting near vs arriving (diagnostic)
| Town | Band | Arm | Passed within 2 m | Of those, arrived at 1 m | Speed at closest, median (km/h) | Near-pass miss, median (m) |
|---|---|---|---|---|---|---|
| Town02 | easy | Topomap | 57 | 18 | 8.8 | 1.35 |
| Town02 | easy | Mapless | 47 | 29 | 5.8 | 1.41 |
| Town02 | medium | Topomap | 43 | 15 | 9.4 | 1.43 |
| Town02 | medium | Mapless | 16 | 7 | 5.8 | 1.42 |
| Town02 | hard | Topomap | 19 | 3 | 6.4 | 1.37 |
| Town02 | hard | Mapless | 0 | 0 | nan | nan |
| Town03 | easy | Topomap | 59 | 39 | 5.8 | 1.57 |
| Town03 | easy | Mapless | 56 | 34 | 5.8 | 1.48 |
| Town03 | medium | Topomap | 24 | 13 | 6.9 | 1.43 |
| Town03 | medium | Mapless | 20 | 15 | 5.8 | 1.72 |
| Town03 | hard | Topomap | 11 | 3 | 5.8 | 1.59 |
| Town03 | hard | Mapless | 2 | 2 | 5.8 | nan |
