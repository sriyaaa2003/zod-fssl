# Literature review: federated self-supervised learning and cross-architecture knowledge transfer on ZOD

Date: 2026-10-10. Status of evidence: I read abstracts and search summaries, not full papers. "I found no work on X" means my searches did not surface it, not that none exists. The RISE job pages were gated (HTTP 410 or a CAPTCHA), so the RISE scope comes from the text Sriya pasted and the public DREAM project page.

## 1. The project in one paragraph

A fleet of vehicles, grouped by country, learns a shared driving representation from unlabeled Zenseact Open Dataset (ZOD) data without sharing raw data (Act 1: federated self-supervised learning, FSSL). A later vehicle generation with a different model or sensor set then inherits that knowledge by distillation (Act 2: federated knowledge transfer). The study adds one question to the established setup: do ZOD's own structure (temporal neighbours in Sequences, camera-LiDAR agreement) make better self-supervision signals than generic image augmentations when clients are non-IID?

## 2. Context: where the thesis sits

- **DREAM** (Distributed, Robust and Efficient AI for Autonomous Vehicles) runs Sep 2025 to Aug 2027, coordinated by RISE with Zenseact, Scaleout and AI Sweden. Its stated pillars are self-supervised learning to cut annotation cost, knowledge distillation when sensors and hardware change, and communication-efficient aggregation. [RISE project page](https://www.ri.se/en/project/dream-distributed-robust-and-efficient-ai-for-autonomous-vehicles)
- **ZOD**: 100k curated camera frames (8 MP, 120 degree field of view), 1,473 20-second sequences, 29 multi-minute drives; cameras, three LiDARs, radar, 100 Hz GNSS/IMU; 14 European countries over two years; CC BY-SA 4.0; annotations for 2D/3D boxes, lanes and road paint, ego-road segmentation, 156 traffic-sign classes and road-surface condition. [ZOD site](https://zod.zenseact.com), [ZOD paper](https://ar5iv.labs.arxiv.org/html/2305.02008)
- **Previous student work on the same problem**: a 2023 Chalmers thesis, [Towards Federated Fleet Learning: Leveraging Unannotated Data](https://odr.chalmers.se/items/f84f18e6-61b8-4df3-a3c6-44564bb8c5ce) (Bellander and Ghafir), used semi-supervised and imitation learning in federated settings for ego-road segmentation and trajectory prediction. Its result was performance on par with or slightly below conventional training, and the authors argued that self-supervised learning in this setting is largely unexplored. This is the clearest statement of the gap, and it is also proof that RISE has seen the baseline before. Its full text (a PDF over 10 MB) was not read.
- RISE's earlier posting describes the aim as demonstrating the efficacy of combining self-supervised and supervised learning on ZOD under various federated scenarios (from search snippets of a gated page).

## 3. Federated self-supervised learning (FSSL)

| Work | What it contributes | Use in this project |
|---|---|---|
| [FedU / Collaborative Unsupervised Visual Representation Learning](https://arxiv.org/pdf/2108.06492) | Federated BYOL-style training with online and target networks | Baseline family |
| [FedEMA / Divergence-aware FedSSL](https://arxiv.org/pdf/2204.04385) | Divergence-aware EMA update of local models; reported 3-4% linear-eval gain over earlier FSSL under non-IID data | Primary FSSL baseline |
| [FedU2](https://arxiv.org/pdf/2403.16398) | Uniformity regulariser plus a unified aggregator against representation collapse under non-IID data (CIFAR-scale evaluation) | Second baseline or ablation |
| [FedMAE](https://ar5iv.labs.arxiv.org/html/2303.11339) | Masked autoencoder variant, cheaper per client | Optional low-compute alternative |
| [Non-IID and heterogeneity review](https://arxiv.org/pdf/2401.00809) | Taxonomy of remedies | Related-work background |

Takeaways: the FSSL literature is mostly evaluated on CIFAR-scale classification with synthetic Dirichlet splits. Non-IID in this project would be real (country, road type, time of day), on 8 MP driving images, which is rarely studied. A positive finding that methods tuned on CIFAR do not transfer would be a legitimate result.

## 4. Federated learning for driving perception

- [Semi-supervised federated object detection under data heterogeneity](https://arxiv.org/pdf/2310.17097), [FL with heterogeneous data handling for vehicular object detection](https://arxiv.org/html/2405.01108v1) and [FedPylot](https://arxiv.org/html/2406.03611v1) (real-time detectors in federated vehicle settings, with resource profiling) show that federated detection works but is mainly supervised or semi-supervised, with communication and compute as constraints.
- Patent filings on FL of self-supervised networks in driving systems exist (found in search results); they are not scientific evidence but indicate industrial interest.

Gap: little public work combines FSSL with ZOD-scale multimodal driving data, and the one thesis I found reports a modest result.

## 5. ZOD-specific self-supervision signals (the project's added question)

- **Temporal positives.** Time-contrastive learning treats nearby frames as positives ([Time-Contrastive Networks](https://arxiv.org/pdf/1704.06888)); [pixel-level contrastive learning on driving videos with optical flow](https://d-itlab.c.titech.ac.jp/wp-content/uploads/2023/11/Takahashi_Pixel-Level_Contrastive_Learning_of_Driving_Videos_With_Optical_Flow_CVPRW_2023_paper.pdf) improves dense downstream tasks. ZOD Sequences (20 s) are the natural source of such pairs. Caveat: ZOD Frames are curated single frames with about 2 s of context, so temporal pairs may be limited to Sequences.
- **Camera-LiDAR positives.** [SLidR](https://arxiv.org/pdf/2203.16258) distills image features into a LiDAR network with superpixels; [LargeAD](https://arxiv.org/pdf/2501.04005) extends this with foundation-model superpixels and temporal consistency. Both are centralized. I found no federated version.
- Combining these inside a federated, non-IID setting was not found. This is the proposed contribution, to be tested, not assumed.

## 6. Cross-architecture and cross-sensor knowledge transfer

- [FedDF](https://arxiv.org/pdf/2006.07242): ensemble distillation on unlabeled data fuses client models of different sizes or structures, with fewer rounds than FedAvg on CIFAR, ImageNet and NLP tasks. FedMD exchanges predictions on a shared set (I only have a secondary description). [Distillation in serverless FL for heterogeneous clients](https://arxiv.org/pdf/2402.07295) is a recent application.
- [BEVDistill](https://arxiv.org/pdf/2211.09386) and [DistillBEV](https://arxiv.org/pdf/2309.15109): a LiDAR-based teacher guides a camera-only BEV student, motivated by fleets where data-collection cars have LiDAR and production cars do not. They are centralized and use detection pipelines on nuScenes-style data.
- Gap: I found no federated version of LiDAR-to-camera distillation, and none on ZOD. FedDF-style transfer for object detection under a platform change is also thinly covered in what I found.

## 7. Privacy (optional extension, links to AI Sweden privacy topics)

- [Do SSL models have deja vu?](https://arxiv.org/pdf/2304.13850) shows SSL encoders can memorise training images, up to a reported 33% deja-vu rate for VICReg, and that standard representation-quality metrics miss it. This motivates a short memorisation check on the federated encoder and the distilled student.

## 8. Gap statement (to be validated)

1. Real, country-level non-IID FSSL on ZOD-scale multimodal driving data, evaluated with label-fraction curves, is rarely reported publicly.
2. Using ZOD's temporal and camera-LiDAR structure as positives inside FSSL was not found.
3. Cross-sensor distillation (LiDAR+camera teacher, camera-only student) inside a federated setting was not found.
4. Whether distillation transfers or removes memorisation of client data was not found.

## 9. Research questions and testable hypotheses

- RQ1: Under country-level non-IID, does FSSL pre-training beat federated supervised-only training at 1%, 5% and 10% labels on traffic-sign classification, road-condition classification and ego-road segmentation? (Baselines: local-only, FedAvg supervised, centralized SSL as the upper bound.)
- RQ2: Do sequence and camera-LiDAR positives beat standard augmentations at equal communication budget? Report negative results plainly.
- RQ3: Can a different-architecture or camera-only student recover most of the teacher's downstream accuracy, in how many rounds and bytes, and with what forgetting of earlier countries?
- RQ4 (optional): Does the distilled student show less or more memorisation than the teacher?

## 10. Reading order

1. DREAM page and ZOD paper (context and data).
2. The Chalmers 2023 thesis (what was tried, what failed).
3. FedEMA, FedU, then FedU2 (FSSL baselines).
4. SLidR (cross-modal positives), then TCN and the driving-video contrastive paper.
5. FedDF, then BEVDistill and DistillBEV.
6. Deja vu (optional).

## 11. How to position the project per target

| Target | Honest fit | What to say |
|---|---|---|
| RISE DREAM theses (deadline 2026-12-15) | Strong | A small-scale study of exactly their two topics on their data, with one added question. It prepares for, and does not replace, the thesis |
| AI Sweden, Continuous Active Federated Learning (2026-12-27) | Strong | Same consortium and same ZOD data; covers federated learning, self-supervision and forgetting; the single-pass active-selection part is not covered unless added |
| Volvo | Moderate | Fleet federated learning and privacy at the edge; I did not verify a currently open Volvo listing |
| Ericsson | Weak to moderate | The vision work is off its core; the earlier time-series projects (nfcast, tailwatch) fit better |

## 12. Risks and open checks

- Check on day one that the ZOD devkit exposes country and time-of-day metadata, a mini subset, and sequence access; I did not confirm any of these.
- Compute: this machine has no NVIDIA GPU; use Colab, Kaggle or university GPUs and a small image subset at reduced resolution.
- The project is a replication plus one question, not a novel framework. Present it that way.
- RISE requires on-site work in Kista at least three days a week.
- Add nothing to `cv.md` until it has been run.
