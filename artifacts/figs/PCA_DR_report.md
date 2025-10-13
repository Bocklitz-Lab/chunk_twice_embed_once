# Dimensionality Reduction Report
- Generated: 2025-10-09T09:40:04
- Method: **PCA**
- PCA explained variance ratio: PC1=0.8850, PC2=0.1054, sum=0.9903
- k_clusters: 4
- β for Fβ@10: 2.0
- Top-N for Fβ@10: 16
- Recall transform: neglog1m
- NDCG transform: neglog1m
- #rows (after dropna): 88 | #models: 43
- Metrics used: ndcg_at_1, ndcg_at_10, ndcg_at_100, map_at_1, map_at_10, map_at_100, recall_at_1, recall_at_10, recall_at_100, precision_at_1, precision_at_10, precision_at_100, mrr_at_1, mrr_at_10, mrr_at_100

## Left subplot: DR scatter & KMeans clusters
| cluster | n_models | DR1_centroid | DR2_centroid | mean(metric_cols) | hull_area | hull_verts |
|---:|---:|---:|---:|---:|---:|---:|
| 0 ⭐ | 19 | 2.8965 | -0.2120 | 0.5470 | 1.0644 | 6 |
| 1 | 5 | -7.6304 | -0.4332 | 0.0614 | 0.9516 | 5 |
| 2 | 13 | 1.0080 | 0.3454 | 0.4576 | 2.2600 | 6 |
| 3 | 6 | -4.9975 | 0.2839 | 0.1814 | 0.7560 | 4 |

### Per-model DR coordinates & cluster assignment
| model | category | cluster | DR1 | DR2 | DR1_plot | DR2_plot | marker | color_hex |
|---|---|---:|---:|---:|---:|---:|:--:|:--:|
| DC_ChemBERTa | biomedical_chemical | 1 | -8.8251 | -0.8211 | -8.8251 | -0.8211 | o | #9467bd |
| MS_BiomedBERT_a | biomedical_chemical | 3 | -5.1083 | 0.4310 | -5.1377 | 0.3863 | s | #9467bd |
| MS_BiomedBERT_af | biomedical_chemical | 3 | -4.7187 | 0.3629 | -4.7187 | 0.3629 | D | #9467bd |
| Rec_ChemBERT | biomedical_chemical | 3 | -5.5881 | 0.6183 | -5.5881 | 0.6183 | ^ | #9467bd |
| MS_MPNetB | foundation_nlp | 1 | -6.2906 | -0.4810 | -6.2906 | -0.4810 | o | #7f7f7f |
| MS_MiniLM384 | foundation_nlp | 1 | -8.6894 | -0.5881 | -8.6894 | -0.5881 | s | #7f7f7f |
| Nomic_BERT2048 | foundation_nlp | 1 | -7.8454 | -0.2832 | -7.8454 | -0.2832 | D | #7f7f7f |
| HKU_InstructXL | instruction_tuned | 0 | 2.7528 | -0.2386 | 2.7107 | -0.3622 | o | #e377c2 |
| Intf_mE5B | multilingual | 0 | 2.5781 | -0.1500 | 2.5654 | -0.1530 | o | #2ca02c |
| Intf_mE5L | multilingual | 2 | 1.9017 | 0.4527 | 1.9262 | 0.5249 | s | #2ca02c |
| Intf_mE5S | multilingual | 0 | 3.2599 | -0.3233 | 3.2727 | -0.3443 | D | #2ca02c |
| Sent_paraMPNetB_v2 | multilingual | 2 | -0.3038 | 1.2177 | -0.3038 | 1.2177 | ^ | #2ca02c |
| BAAI_bgeB | retrieval | 0 | 2.2553 | -0.2191 | 2.2922 | -0.2721 | o | #1f77b4 |
| BAAI_bgeB_v1.5 | retrieval | 0 | 3.9042 | -0.9741 | 3.9042 | -0.9741 | s | #1f77b4 |
| BAAI_bgeL | retrieval | 2 | -0.2419 | -0.4107 | -0.2419 | -0.4107 | D | #1f77b4 |
| BAAI_bgeL_v1.5 | retrieval | 0 | 3.0527 | 0.0880 | 3.0418 | 0.1094 | ^ | #1f77b4 |
| BAAI_bgeM3 | retrieval | 0 | 3.1511 | -0.1360 | 3.1633 | -0.1143 | v | #1f77b4 |
| BAAI_bgeS | retrieval | 2 | 0.3863 | 0.5830 | 0.3863 | 0.5830 | < | #1f77b4 |
| BAAI_bgeS_v1.5 | retrieval | 0 | 2.3081 | 0.2792 | 2.2861 | 0.2772 | > | #1f77b4 |
| FB_contriever | retrieval | 2 | 0.2747 | -0.4501 | 0.2747 | -0.4501 | p | #1f77b4 |
| FB_contrieverMS | retrieval | 2 | 1.7143 | 0.4305 | 1.6849 | 0.4322 | P | #1f77b4 |
| Intf_e5B | retrieval | 0 | 2.7931 | -0.1630 | 2.8594 | -0.1555 | * | #1f77b4 |
| Intf_e5B_v2 | retrieval | 2 | 1.8955 | 0.3455 | 1.9004 | 0.2716 | X | #1f77b4 |
| Intf_e5L | retrieval | 0 | 3.3340 | -0.6082 | 3.2598 | -0.5985 | H | #1f77b4 |
| Intf_e5L_v2 | retrieval | 0 | 3.3993 | -0.6665 | 3.4768 | -0.7317 | 8 | #1f77b4 |
| Intf_e5S | retrieval | 0 | 3.0494 | -0.2960 | 3.0183 | -0.3544 | d | #1f77b4 |
| Intf_e5S_v2 | retrieval | 0 | 2.3752 | 0.0069 | 2.4198 | 0.0558 | h | #1f77b4 |
| Jina_v2B | retrieval | 0 | 2.5177 | 0.2983 | 2.5396 | 0.3003 | 1 | #1f77b4 |
| Jina_v2S | retrieval | 0 | 2.0828 | -0.1769 | 2.0071 | -0.2206 | 2 | #1f77b4 |
| Sent_MQA_MPNetB | retrieval | 2 | 1.2169 | 0.4361 | 1.2332 | 0.4376 | 3 | #1f77b4 |
| Sent_gtrT5_B | retrieval | 2 | 1.1864 | 0.2029 | 1.2025 | 0.1833 | 4 | #1f77b4 |
| Sent_gtrT5_L | retrieval | 2 | 1.0081 | 0.7168 | 1.0081 | 0.7168 | x | #1f77b4 |
| Sent_gtrT5_XL | retrieval | 2 | 1.3987 | 0.5856 | 1.4235 | 0.6067 | + | #1f77b4 |
| AI_SciBERT | scientific | 3 | -5.4333 | -0.3118 | -5.4333 | -0.3118 | o | #d62728 |
| AI_Specter | scientific | 3 | -5.0272 | 0.5543 | -4.9978 | 0.5990 | s | #d62728 |
| IITD_MatSci | scientific | 1 | -6.5016 | 0.0074 | -6.5016 | 0.0074 | D | #d62728 |
| Sent_BERTnli | semantic_similarity | 3 | -4.1096 | 0.0483 | -4.1096 | 0.0483 | o | #ff7f0e |
| Sent_MiniLM12_v2 | semantic_similarity | 2 | 1.6128 | 0.0361 | 1.6128 | 0.0361 | s | #ff7f0e |
| Sent_MiniLM6_v2 | semantic_similarity | 0 | 2.2066 | -0.1048 | 2.1823 | -0.0359 | D | #ff7f0e |
| Sent_allMPNetB_v2 | semantic_similarity | 2 | 1.0547 | 0.3447 | 0.9974 | 0.3416 | ^ | #ff7f0e |
| Nomic_text_v1 | universal_encoders | 0 | 3.8440 | -0.3869 | 3.8440 | -0.3869 | o | #8c564b |
| Nomic_text_v1.5 | universal_encoders | 0 | 3.3800 | -0.1974 | 3.4311 | -0.1450 | s | #8c564b |
| Nomic_text_v2 | universal_encoders | 0 | 2.7886 | -0.0593 | 2.7582 | 0.0781 | D | #8c564b |

## Right subplot: Recall@10 vs NDCG@10 & Fβ@10
- Best cluster by mean(metric_cols): **0**
- Intersection (Best ∩ Top-16 Fβ@10): 16 models
  - BAAI_bgeB_v1.5, BAAI_bgeL_v1.5, BAAI_bgeM3, BAAI_bgeS_v1.5, HKU_InstructXL, Intf_e5B, Intf_e5L, Intf_e5L_v2, Intf_e5S, Intf_mE5B, Intf_mE5S, Jina_v2B, Nomic_text_v1, Nomic_text_v1.5, Nomic_text_v2, Sent_MiniLM6_v2
- Only in Best cluster: 3 models
  - BAAI_bgeB, Intf_e5S_v2, Jina_v2S
- Only in Top-16 Fβ@10: 0 models
  - (none)

### Top 16 models by Fβ@10 (β=2.0)
| rank | model | category | Fβ@10 | recall@10 | ndcg@10 |
|---:|---|---|---:|---:|---:|
| 1 | Nomic_text_v1.5 | universal_encoders | 0.8460 | 0.8920 | 0.7013 |
| 2 | Nomic_text_v1 | universal_encoders | 0.8413 | 0.8827 | 0.7085 |
| 3 | BAAI_bgeL_v1.5 | retrieval | 0.8262 | 0.8765 | 0.6718 |
| 4 | Intf_mE5S | multilingual | 0.8248 | 0.8673 | 0.6895 |
| 5 | Intf_e5L | retrieval | 0.8193 | 0.8549 | 0.7023 |
| 6 | BAAI_bgeM3 | retrieval | 0.8069 | 0.8457 | 0.6818 |
| 7 | Intf_e5B | retrieval | 0.8008 | 0.8395 | 0.6762 |
| 8 | Intf_e5S | retrieval | 0.7999 | 0.8395 | 0.6729 |
| 9 | Jina_v2B | retrieval | 0.7997 | 0.8426 | 0.6645 |
| 10 | BAAI_bgeB_v1.5 | retrieval | 0.7850 | 0.8117 | 0.6936 |
| 11 | Nomic_text_v2 | universal_encoders | 0.7837 | 0.8210 | 0.6632 |
| 12 | HKU_InstructXL | instruction_tuned | 0.7825 | 0.8210 | 0.6588 |
| 13 | BAAI_bgeS_v1.5 | retrieval | 0.7707 | 0.8179 | 0.6263 |
| 14 | Intf_mE5B | multilingual | 0.7661 | 0.7994 | 0.6568 |
| 15 | Intf_e5L_v2 | retrieval | 0.7650 | 0.7901 | 0.6786 |
| 16 | Sent_MiniLM6_v2 | semantic_similarity | 0.7491 | 0.7932 | 0.6127 |

### Per-model: recall/ndcg/Fβ and transformed coords
| model | category | cluster | recall@10 | ndcg@10 | Fβ@10 | x=ndcg_t_plot | y=recall_t_plot |
|---|---|---:|---:|---:|---:|---:|---:|
| DC_ChemBERTa | biomedical_chemical | 1 | 0.0278 | 0.0084 | 0.0190 | -0.0309 | 0.0295 |
| MS_BiomedBERT_a | biomedical_chemical | 3 | 0.2870 | 0.2061 | 0.2661 | 0.2167 | 0.3265 |
| MS_BiomedBERT_af | biomedical_chemical | 3 | 0.3241 | 0.2198 | 0.2960 | 0.2539 | 0.3938 |
| Rec_ChemBERT | biomedical_chemical | 3 | 0.2500 | 0.1747 | 0.2302 | 0.1546 | 0.2812 |
| MS_MPNetB | foundation_nlp | 1 | 0.1852 | 0.1442 | 0.1752 | 0.1599 | 0.2044 |
| MS_MiniLM384 | foundation_nlp | 1 | 0.0278 | 0.0088 | 0.0194 | 0.0452 | 0.0184 |
| Nomic_BERT2048 | foundation_nlp | 1 | 0.0833 | 0.0495 | 0.0733 | 0.0530 | 0.0949 |
| HKU_InstructXL | instruction_tuned | 0 | 0.8210 | 0.6588 | 0.7825 | 1.0482 | 1.7233 |
| Intf_mE5B | multilingual | 0 | 0.7994 | 0.6568 | 0.7661 | 1.0695 | 1.6064 |
| Intf_mE5L | multilingual | 2 | 0.7839 | 0.6042 | 0.7399 | 0.9249 | 1.5342 |
| Intf_mE5S | multilingual | 0 | 0.8673 | 0.6895 | 0.8248 | 1.1696 | 2.0196 |
| Sent_paraMPNetB_v2 | multilingual | 2 | 0.6667 | 0.4839 | 0.6199 | 0.6615 | 1.0986 |
| BAAI_bgeB | retrieval | 0 | 0.7006 | 0.5909 | 0.6755 | 0.8960 | 1.2058 |
| BAAI_bgeB_v1.5 | retrieval | 0 | 0.8117 | 0.6936 | 0.7850 | 1.1831 | 1.6698 |
| BAAI_bgeL | retrieval | 2 | 0.5278 | 0.4447 | 0.5088 | 0.5883 | 0.7503 |
| BAAI_bgeL_v1.5 | retrieval | 0 | 0.8765 | 0.6718 | 0.8262 | 1.1140 | 2.0918 |
| BAAI_bgeM3 | retrieval | 0 | 0.8457 | 0.6818 | 0.8069 | 1.1507 | 1.8878 |
| BAAI_bgeS | retrieval | 2 | 0.7006 | 0.5197 | 0.6550 | 0.7334 | 1.2060 |
| BAAI_bgeS_v1.5 | retrieval | 0 | 0.8179 | 0.6263 | 0.7707 | 0.9747 | 1.7005 |
| FB_contriever | retrieval | 2 | 0.5648 | 0.4743 | 0.5441 | 0.6431 | 0.8320 |
| FB_contrieverMS | retrieval | 2 | 0.7654 | 0.6023 | 0.7261 | 0.9051 | 1.4598 |
| Intf_e5B | retrieval | 0 | 0.8395 | 0.6762 | 0.8008 | 1.1690 | 1.8131 |
| Intf_e5B_v2 | retrieval | 2 | 0.7901 | 0.6140 | 0.7472 | 0.9924 | 1.5709 |
| Intf_e5L | retrieval | 0 | 0.8549 | 0.7023 | 0.8193 | 1.2136 | 1.9321 |
| Intf_e5L_v2 | retrieval | 0 | 0.7901 | 0.6786 | 0.7650 | 1.1350 | 1.5612 |
| Intf_e5S | retrieval | 0 | 0.8395 | 0.6729 | 0.7999 | 1.0931 | 1.8008 |
| Intf_e5S_v2 | retrieval | 0 | 0.7809 | 0.6222 | 0.7430 | 0.9944 | 1.4932 |
| Jina_v2B | retrieval | 0 | 0.8426 | 0.6645 | 0.7997 | 1.0674 | 1.8733 |
| Jina_v2S | retrieval | 0 | 0.7623 | 0.6122 | 0.7267 | 0.9709 | 1.4200 |
| Sent_MQA_MPNetB | retrieval | 2 | 0.7469 | 0.5634 | 0.7012 | 0.8119 | 1.3669 |
| Sent_gtrT5_B | retrieval | 2 | 0.7191 | 0.5522 | 0.6781 | 0.7987 | 1.2855 |
| Sent_gtrT5_L | retrieval | 2 | 0.7654 | 0.5610 | 0.7134 | 0.8214 | 1.4466 |
| Sent_gtrT5_XL | retrieval | 2 | 0.7809 | 0.5894 | 0.7332 | 0.8496 | 1.5182 |
| AI_SciBERT | scientific | 3 | 0.2500 | 0.1980 | 0.2375 | 0.2466 | 0.2557 |
| AI_Specter | scientific | 3 | 0.3426 | 0.2211 | 0.3087 | 0.2427 | 0.4699 |
| IITD_MatSci | scientific | 1 | 0.1389 | 0.1092 | 0.1317 | 0.1120 | 0.1443 |
| Sent_BERTnli | semantic_similarity | 3 | 0.3333 | 0.2609 | 0.3158 | 0.3292 | 0.4094 |
| Sent_MiniLM12_v2 | semantic_similarity | 2 | 0.7531 | 0.5843 | 0.7119 | 0.8866 | 1.3852 |
| Sent_MiniLM6_v2 | semantic_similarity | 0 | 0.7932 | 0.6127 | 0.7491 | 0.9325 | 1.6204 |
| Sent_allMPNetB_v2 | semantic_similarity | 2 | 0.7068 | 0.5582 | 0.6711 | 0.8194 | 1.2115 |
| Nomic_text_v1 | universal_encoders | 0 | 0.8827 | 0.7085 | 0.8413 | 1.2327 | 2.1431 |
| Nomic_text_v1.5 | universal_encoders | 0 | 0.8920 | 0.7013 | 0.8460 | 1.2083 | 2.2254 |
| Nomic_text_v2 | universal_encoders | 0 | 0.8210 | 0.6632 | 0.7837 | 1.1250 | 1.7201 |

