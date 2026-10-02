# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.2638  0.0203       0.2880  0.0107
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.2676  0.0281       0.2952  0.0073
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2656  0.0229       0.2961  0.0101
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.2743  0.0240       0.2936  0.0068
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2694  0.0225       0.2939  0.0078
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.2727  0.0246       0.2937  0.0062
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2703  0.0221       0.2946  0.0079
```
