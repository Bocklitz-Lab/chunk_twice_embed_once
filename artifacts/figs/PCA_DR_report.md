# Dimensionality Reduction Report
- Generated: 2025-10-14T14:54:25
- Method: **PCA**
- PCA explained variance ratio: PC1=0.8867, PC2=0.1040, sum=0.9907
- k_clusters: 4
- GEOM weights: w_ndcg=1.0, w_recall=1.0, w_time=0.0
- time_transform: inverse, reference_seconds=1.0, reference_qps=100.0
- Top-N by GEOM@10: 16
- Recall transform: neglog1m
- NDCG transform: neglog1m
- #rows (after dropna): 84 | #models: 41
- Metrics used: ndcg_at_1, ndcg_at_10, ndcg_at_100, map_at_1, map_at_10, map_at_100, recall_at_1, recall_at_10, recall_at_100, precision_at_1, precision_at_10, precision_at_100, mrr_at_1, mrr_at_10, mrr_at_100

## Left subplot: DR scatter & KMeans clusters
| cluster | n_models | DR1_centroid | DR2_centroid | mean(metric_cols) | hull_area | hull_verts |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13 | 1.1037 | 0.3449 | 0.4576 | 2.1817 | 6 |
| 1 | 8 | -5.1581 | 0.1514 | 0.1656 | 1.4175 | 6 |
| 2 ⭐ | 17 | 3.0336 | -0.2361 | 0.5503 | 0.9373 | 6 |
| 3 | 3 | -8.2184 | -0.5604 | 0.0235 | 0.0762 | 3 |

### Per-model DR coordinates & cluster assignment
| model | category | cluster | DR1 | DR2 | DR1_plot | DR2_plot | marker | color_hex |
|---|---|---:|---:|---:|---:|:--:|:--:|
| DC_ChemBERTa | biomedical_chemical | 3 | -8.5845 | -0.8161 | -8.5845 | -0.8161 | o | #9467bd |
| MS_BiomedBERT_a | biomedical_chemical | 1 | -4.9227 | 0.4245 | -4.9528 | 0.3800 | s | #9467bd |
| MS_BiomedBERT_af | biomedical_chemical | 1 | -4.5391 | 0.3605 | -4.5391 | 0.3605 | D | #9467bd |
| Rec_ChemBERT | biomedical_chemical | 1 | -5.3955 | 0.6099 | -5.3955 | 0.6099 | ^ | #9467bd |
| MS_MPNetB | foundation_nlp | 1 | -6.0875 | -0.4764 | -6.0875 | -0.4764 | o | #7f7f7f |
| MS_MiniLM384 | foundation_nlp | 3 | -8.4510 | -0.5840 | -8.4510 | -0.5840 | s | #7f7f7f |
| Nomic_BERT2048 | foundation_nlp | 3 | -7.6196 | -0.2810 | -7.6196 | -0.2810 | D | #7f7f7f |
| HKU_InstructXL | instruction_tuned | 2 | 2.8231 | -0.2323 | 2.7775 | -0.3513 | o | #e377c2 |
| Intf_mE5B | multilingual | 2 | 2.6512 | -0.1478 | 2.6345 | -0.1451 | o | #2ca02c |
| Intf_mE5L | multilingual | 0 | 1.9842 | 0.4513 | 2.0077 | 0.5204 | s | #2ca02c |
| Intf_mE5S | multilingual | 2 | 3.3228 | -0.3158 | 3.3361 | -0.3387 | D | #2ca02c |
| Sent_paraMPNetB_v2 | multilingual | 0 | -0.1886 | 1.2017 | -0.1886 | 1.2017 | ^ | #2ca02c |
| BAAI_bgeB | retrieval | 2 | 2.3316 | -0.2020 | 2.3626 | -0.2619 | o | #1f77b4 |
| BAAI_bgeB_v1.5 | retrieval | 2 | 3.9566 | -0.9481 | 3.9566 | -0.9481 | s | #1f77b4 |
| BAAI_bgeL | retrieval | 0 | -0.1291 | -0.3901 | -0.1291 | -0.3901 | D | #1f77b4 |
| BAAI_bgeL_v1.5 | retrieval | 2 | 3.1185 | 0.0912 | 3.1093 | 0.1093 | ^ | #1f77b4 |
| BAAI_bgeM3 | retrieval | 2 | 3.2155 | -0.1307 | 3.2288 | -0.1112 | v | #1f77b4 |
| BAAI_bgeS | retrieval | 0 | 0.4912 | 0.5783 | 0.4912 | 0.5783 | < | #1f77b4 |
| BAAI_bgeS_v1.5 | retrieval | 2 | 2.3848 | 0.2792 | 2.3848 | 0.2792 | > | #1f77b4 |
| FB_contriever | retrieval | 0 | 0.3800 | -0.4291 | 0.3800 | -0.4291 | p | #1f77b4 |
| FB_contrieverMS | retrieval | 0 | 1.8000 | 0.4253 | 1.7711 | 0.4266 | P | #1f77b4 |
| Intf_e5B | retrieval | 2 | 2.8632 | -0.1618 | 2.9290 | -0.1514 | * | #1f77b4 |
| Intf_e5B_v2 | retrieval | 0 | 1.9786 | 0.3412 | 1.9841 | 0.2707 | X | #1f77b4 |
| Intf_e5L | retrieval | 2 | 3.3962 | -0.5996 | 3.3185 | -0.5889 | H | #1f77b4 |
| Intf_e5L_v2 | retrieval | 2 | 3.4596 | -0.6492 | 3.5391 | -0.7083 | 8 | #1f77b4 |
| Intf_e5S | retrieval | 2 | 3.1153 | -0.2881 | 3.0854 | -0.3476 | d | #1f77b4 |
| Intf_e5S_v2 | retrieval | 2 | 2.4507 | 0.0129 | 2.4634 | 0.0383 | h | #1f77b4 |
| Sent_MQA_MPNetB | retrieval | 0 | 1.3097 | 0.4331 | 1.3246 | 0.4349 | 1 | #1f77b4 |
| Sent_gtrT5_B | retrieval | 0 | 1.2793 | 0.2059 | 1.2957 | 0.1856 | 2 | #1f77b4 |
| Sent_gtrT5_L | retrieval | 0 | 1.1042 | 0.7071 | 1.1042 | 0.7071 | 3 | #1f77b4 |
| Sent_gtrT5_XL | retrieval | 0 | 1.4892 | 0.5763 | 1.5154 | 0.5977 | 4 | #1f77b4 |
| AI_SciBERT | scientific | 1 | -5.2427 | -0.3101 | -5.2427 | -0.3101 | o | #d62728 |
| AI_Specter | scientific | 1 | -4.8424 | 0.5433 | -4.8124 | 0.5878 | s | #d62728 |
| IITD_MatSci | scientific | 1 | -6.2961 | 0.0115 | -6.2961 | 0.0115 | D | #d62728 |
| Sent_BERTnli | semantic_similarity | 1 | -3.9387 | 0.0480 | -3.9387 | 0.0480 | o | #ff7f0e |
| Sent_MiniLM12_v2 | semantic_similarity | 0 | 1.6997 | 0.0399 | 1.6997 | 0.0399 | s | #ff7f0e |
| Sent_MiniLM6_v2 | semantic_similarity | 2 | 2.2846 | -0.0969 | 2.2286 | -0.0498 | D | #ff7f0e |
| Sent_allMPNetB_v2 | semantic_similarity | 0 | 1.1498 | 0.3430 | 1.0923 | 0.3402 | ^ | #ff7f0e |
| Nomic_text_v1 | universal_encoders | 2 | 3.8979 | -0.3744 | 3.8979 | -0.3744 | o | #8c564b |
| Nomic_text_v1.5 | universal_encoders | 2 | 3.4414 | -0.1940 | 3.4917 | -0.1420 | s | #8c564b |
| Nomic_text_v2 | universal_encoders | 2 | 2.8585 | -0.0565 | 2.8278 | 0.0780 | D | #8c564b |

## Right subplot: Recall@10 vs NDCG@10 & GEOM@10
- Best cluster by mean(metric_cols): **2**
- Intersection (Best ∩ Top-16 GEOM@10): 16 models
  - BAAI_bgeB_v1.5, BAAI_bgeL_v1.5, BAAI_bgeM3, BAAI_bgeS_v1.5, HKU_InstructXL, Intf_e5B, Intf_e5L, Intf_e5L_v2, Intf_e5S, Intf_e5S_v2, Intf_mE5B, Intf_mE5S, Nomic_text_v1, Nomic_text_v1.5, Nomic_text_v2, Sent_MiniLM6_v2
- Only in Best cluster: 1 models
  - BAAI_bgeB
- Only in Top-16 GEOM@10: 0 models
  - (none)

### Top 16 models by GEOM@10 (w=[1.0,1.0,0.0])
| rank | model | category | GEOM@10 | recall@10 | ndcg@10 | time_score |
|---:|---|---|---:|---:|---:|---:|
| 1 | Nomic_text_v1.5 | universal_encoders | 0.7909 | 0.8920 | 0.7013 | 0.0142 |
| 2 | Nomic_text_v1 | universal_encoders | 0.7908 | 0.8827 | 0.7085 | 0.0141 |
| 3 | Intf_e5L | retrieval | 0.7749 | 0.8549 | 0.7023 | 0.0068 |
| 4 | Intf_mE5S | multilingual | 0.7733 | 0.8673 | 0.6895 | 0.0387 |
| 5 | BAAI_bgeL_v1.5 | retrieval | 0.7674 | 0.8765 | 0.6718 | 0.0068 |
| 6 | BAAI_bgeM3 | retrieval | 0.7593 | 0.8457 | 0.6818 | 0.0065 |
| 7 | Intf_e5B | retrieval | 0.7535 | 0.8395 | 0.6762 | 0.0193 |
| 8 | Intf_e5S | retrieval | 0.7516 | 0.8395 | 0.6729 | 0.0403 |
| 9 | BAAI_bgeB_v1.5 | retrieval | 0.7504 | 0.8117 | 0.6936 | 0.0192 |
| 10 | Nomic_text_v2 | universal_encoders | 0.7379 | 0.8210 | 0.6632 | 0.0087 |
| 11 | HKU_InstructXL | instruction_tuned | 0.7354 | 0.8210 | 0.6588 | 0.0017 |
| 12 | Intf_e5L_v2 | retrieval | 0.7322 | 0.7901 | 0.6786 | 0.0068 |
| 13 | Intf_mE5B | multilingual | 0.7246 | 0.7994 | 0.6568 | 0.0184 |
| 14 | BAAI_bgeS_v1.5 | retrieval | 0.7157 | 0.8179 | 0.6263 | 0.0399 |
| 15 | Sent_MiniLM6_v2 | semantic_similarity | 0.6971 | 0.7932 | 0.6127 | 0.0625 |
| 16 | Intf_e5S_v2 | retrieval | 0.6970 | 0.7809 | 0.6222 | 0.0406 |

