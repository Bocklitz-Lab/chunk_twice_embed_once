# Per-task performance of all evaluated models

This table shows each model’s best configuration in terms of **F₂** score. Precision and recall at cutoff 10 are also reported to provide a breakdown of the F₂ components.

| Task                  | Model           | F₂     | Precision@10 | Recall@10 |
|------------------------|-----------------|--------|--------------|-----------|
| ChemHotpotQARetrieval  | BAAI_bgeL_v1.5  | 0.3373 | 0.0944       | 0.9444    |
| ChemHotpotQARetrieval  | BAAI_bgeS_v1.5  | 0.3373 | 0.0944       | 0.9444    |
| ChemHotpotQARetrieval  | Intf_mE5S       | 0.3373 | 0.0944       | 0.9444    |
| ChemHotpotQARetrieval  | Nomic_text_v1   | 0.3373 | 0.0944       | 0.9444    |
| ChemHotpotQARetrieval  | Nomic_text_v1.5 | 0.3373 | 0.0944       | 0.9444    |
| ChemHotpotQARetrieval  | BAAI_bgeB       | 0.3175 | 0.0889       | 0.8889    |
| ChemHotpotQARetrieval  | BAAI_bgeL       | 0.3175 | 0.0889       | 0.8889    |
| ChemHotpotQARetrieval  | BAAI_bgeM3      | 0.3175 | 0.0889       | 0.8889    |
| ChemHotpotQARetrieval  | BAAI_bgeS       | 0.3175 | 0.0889       | 0.8889    |
| ChemHotpotQARetrieval  | Intf_e5S        | 0.3175 | 0.0889       | 0.8889    |
| ChemNQRetrieval        | Nomic_text_v1.5 | 0.3999 | 0.1222       | 0.9259    |
| ChemNQRetrieval        | Intf_mE5B       | 0.3901 | 0.1185       | 0.9136    |
| ChemNQRetrieval        | Intf_mE5L       | 0.3901 | 0.1185       | 0.9136    |
| ChemNQRetrieval        | Intf_e5L_v2     | 0.3803 | 0.1148       | 0.9012    |
| ChemNQRetrieval        | Intf_e5B        | 0.3767 | 0.1148       | 0.8765    |
| ChemNQRetrieval        | Intf_e5B_v2     | 0.3767 | 0.1148       | 0.8765    |
| ChemNQRetrieval        | Intf_e5L        | 0.3767 | 0.1148       | 0.8765    |
| ChemNQRetrieval        | Nomic_text_v1   | 0.3758 | 0.1148       | 0.8704    |
| ChemNQRetrieval        | BAAI_bgeB_v1.5  | 0.3633 | 0.1111       | 0.8395    |
| ChemNQRetrieval        | Intf_e5S_v2     | 0.3633 | 0.1111       | 0.8395    |
| FSUChemRxivQest        | Intf_e5L_v2     | 0.3058 | 0.0980       | 0.6504    |
| FSUChemRxivQest        | Intf_e5B        | 0.3051 | 0.0976       | 0.6507    |
| FSUChemRxivQest        | Intf_e5L        | 0.3034 | 0.0970       | 0.6481    |
| FSUChemRxivQest        | Nomic_text_v1   | 0.3019 | 0.0966       | 0.6438    |
| FSUChemRxivQest        | Intf_mE5L       | 0.3010 | 0.0964       | 0.6414    |
| FSUChemRxivQest        | Intf_e5B_v2     | 0.2986 | 0.0956       | 0.6368    |
| FSUChemRxivQest        | Intf_mE5B       | 0.2974 | 0.0954       | 0.6320    |
| FSUChemRxivQest        | BAAI_bgeM3      | 0.2958 | 0.0945       | 0.6323    |
| FSUChemRxivQest        | Intf_e5S_v2     | 0.2932 | 0.0939       | 0.6244    |
| FSUChemRxivQest        | Intf_e5S        | 0.2925 | 0.0935       | 0.6253    |
