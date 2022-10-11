## Adversarial Stuff
- Adversarial training can happen in just one stage - neat!
- The main things is we need domain labels, but those can be auto-generated (maybe even in the train function)

## General library notes
- `forward_features`, from `mix_transformer.py` is not used anywhere. `dacs.py` instead uses `encoder_decoder.py`'s `get_features` paramter to get the final features (IIUC). We may need to modify `encoder_decoder.py` (and in turn other models) to get the specific features we need.

## USEFUL LATER
- There is a way to ignore train IDs - https://mmsegmentation.readthedocs.io/en/latest/tutorials/training_tricks.html#ignore-specified-label-index-in-loss-calculation

### Still need to figure out
- How to "train" on target dataset when we don't actually care about the output because we don't have any labels
    - Maybe it's as simple as we just only use -L_d and not L_y for target? That seems to be what Max is doing
