# sorAI

sorAI predicts whether a plant sample is diploid or polyploid from targeted-capture sequencing data.

The model to use on a new sample is the random forest. It combines the polyploid probabilities from locus-specific convolutional networks trained on ferns sequenced with the GoFlag 408 probe set. On the held-out fern test set (34 samples) this forest has accuracy 0.765 and AUROC 0.870. The decision threshold, chosen on the training data only, is 0.48.

sorAI does not estimate a chromosome number, and it does not separate triploids, tetraploids, and higher polyploids. It was trained on ferns. Applying it to other clades or probe sets without retraining is not supported.

## Access

| What | Where |
| --- | --- |
| Code and random forest | this repository: [github.com/Fahimehgit/sorai](https://github.com/Fahimehgit/sorai) |
| Locus CNN checkpoints (6.8 GB) | Hugging Face: [huggingface.co/Fahimehr140/sorai](https://huggingface.co/Fahimehr140/sorai) |

```bash
# 1. Get the code (includes the random forest)
git clone https://github.com/Fahimehgit/sorai.git
cd sorai
pip install -e .

# 2. Get the locus CNN checkpoints (needed to score alignments)
pip install -U "huggingface_hub[cli]"
hf download Fahimehr140/sorai --local-dir sorai-cnns
```

After that, the forest is in `sorai/models/rf/` and the CNN files are in `sorai-cnns/checkpoints/`.

## Install notes

Python 3.9 or newer. The exported forest was saved with scikit-learn 1.6.1; use that version so the file loads cleanly. A conda environment is in `environment.yml`.

## Predict

If you already have per-locus probabilities, one row per recovered locus:

```bash
sorai predict --probabilities locus_probabilities.csv
```

The CSV needs a `prob_polyploid` column. The result is the forest probability, the call at threshold 0.48, and the number of loci used.

If you have tokenized locus alignments and the downloaded CNN checkpoints:

```bash
sorai predict --loci-dir sample_loci/ --cnn-dir sorai-cnns/checkpoints
```

Each file in `sample_loci/` is named `L<probe_id>.json` and contains `reference_tokens` and `auxiliary_tokens` (integers: A=1, C=2, G=3, T=4, gap or other=0). Sequences are truncated or padded to 154 positions. Up to 350 reads are kept, and missing rows are padded, giving the 351 × 154 input used in training. `models/probe_index_to_id.csv` maps each GoFlag locus id to the checkpoint `best_model_probe_3_<index>.pt`.

Loci that are missing from the sample are skipped. The forest was trained on whatever loci a sample actually recovered.

## Model weights

Included in this repository:

- `sorai/models/rf/rf_model.pkl`
- `sorai/models/rf/feature_columns.json`
- `sorai/models/rf/model_metadata.json`

Downloaded separately from Hugging Face ([Fahimehr140/sorai](https://huggingface.co/Fahimehr140/sorai)):

- `checkpoints/best_model_probe_3_<index>.pt` for 384 loci
- `probe_index_to_id.csv`

## Citation

Fahimeh Rahimi, Jessie A. Pelosi, J. Gordon Burleigh, and Juannan Zhou. sorAI: machine learning prediction of plant polyploidy from target capture data.

If you use this repository, please also cite the paper once it is available. Author ORCIDs are in `CITATION.cff`.

## License

MIT. See `LICENSE`.
