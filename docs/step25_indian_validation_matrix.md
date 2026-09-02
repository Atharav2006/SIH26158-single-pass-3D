# Indian Validation Suite — Capability Matrix

## Required Capabilities vs Candidate Datasets

### BUILDINGS

| Capability | UASG Delhi | UASG Roorkee | UASG Slum | SkyEye | IIIT-H Infra | Evidence |
|---|---|---|---|---|---|---|
| 2–4 storey buildings | LIKELY | LIKELY | LIKELY | LIKELY | LIKELY | Dense urban UAV imagery at 100m altitude |
| Balconies | LIKELY | LIKELY | UNKNOWN | UNKNOWN | UNKNOWN | — |
| Windows | LIKELY | LIKELY | UNKNOWN | UNKNOWN | CONFIRMED | IIIT-H campus window dataset |
| Doors | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | — |
| Rooftops | LIKELY | LIKELY | LIKELY | LIKELY | UNKNOWN | Nadir UAV captures rooftops |
| Water tanks | LIKELY | LIKELY | UNKNOWN | UNKNOWN | UNKNOWN | Common on Indian rooftops |
| AC outdoor units | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | — |
| External pipes | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | — |
| Facade geometry | LIKELY | LIKELY | UNKNOWN | UNKNOWN | UNKNOWN | — |

### ROADS

| Capability | UASG Delhi | SkyEye | Evidence |
|---|---|---|---|
| Straight road | LIKELY | CONFIRMED | SkyEye captures named roads |
| Intersection | UNKNOWN | CONFIRMED | 4 Ahmedabad intersections |
| T-junction | UNKNOWN | LIKELY | — |
| Road markings | UNKNOWN | LIKELY | — |
| Speed breaker | UNKNOWN | UNKNOWN | — |

### DYNAMIC OBJECTS

| Capability | SkyEye | MUAAD | Evidence |
|---|---|---|---|
| Moving cars | CONFIRMED | UNKNOWN | SkyEye MOT annotations |
| Pedestrians | CONFIRMED | LIKELY | SkyEye annotations |
| Moving motorcycles | CONFIRMED | UNKNOWN | SkyEye annotations |
| Auto-rickshaws | LIKELY | UNKNOWN | Ahmedabad traffic |

### VEGETATION & TERRAIN

| Capability | Gujarat Forest | Nagaland Landslide | Evidence |
|---|---|---|---|
| Large trees | CONFIRMED | LIKELY | "Dense Forest" label |
| Irregular vegetation | CONFIRMED | LIKELY | — |
| Landslide terrain | NOT_SUPPORTED | CONFIRMED | "Landslide Area" label |
