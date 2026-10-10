# zod-fssl

Does federated self-supervised aggregation (FedAvg, FedEMA, L-DAWA) rank the same way on real
driving data split by country (Zenseact Open Dataset) as on CIFAR-style synthetic splits?

Status: scaffold only. Nothing has been trained and no results exist yet.

## Plan
1. Centralized SSL baseline (ResNet-18, SimSiam/BYOL-style) on a ZOD subset at reduced resolution.
2. Country-level clients vs Dirichlet clients on the same data.
3. FedAvg vs FedEMA vs L-DAWA, several seeds, communication cost per round.
4. Frozen-encoder evaluation at 1%, 5%, 10% labels (traffic signs, road condition, ego-road).
5. Optional: sequence / camera-LiDAR positives, distillation to a different architecture.

## Running on Modal
```
modal run modal_app.py::smoke       # GPU + imports
modal secret create zod-download ZOD_URL=<link from zod.zenseact.com>
modal run modal_download.py         # ZOD mini frames -> volume 'zod-data'
```
Heavy compute runs on Modal; the volume `zod-data` holds the dataset.

## Open checks
- Metadata fields used for the split (`country_code`, `time_of_day`, `scraped_weather`) come from the devkit README; verify on the downloaded data.
- Number of countries in the mini subset decides how many clients are possible.
