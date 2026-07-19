# Seam Carving for Content-Aware Image Resizing

An Octave implementation of the classic image resizing algorithm:

> Avidan, S., Shamir, A.
> *Seam carving for content-aware image resizing.*
> *ACM Transactions on Graphics* 26, 10 (2007).
> [https://doi.org/10.1145/1276377.1276390](https://doi.org/10.1145/1276377.1276390)

Requires the `image` (`pkg install -forge image`) and `video` (`pkg install -forge video`) packages.

```console
➜ octave functions/main.m
Usage: octave main.m <input> <output> <reduction direction {v|h}> <amount> <p for L_p or 0 for L_infinity> [--animations]
```

## Gallery

The minimum-energy vertical or horizontal seams (shown in red) are iteratively removed until the target image dimensions are reached.

| Image | Energy Map | Cumulative Energy Map |
| :---: | :---: | :---: |
|<img width="240" height="180" alt="lake_h200_linfty" src="https://github.com/user-attachments/assets/bf89e373-cae1-4fcd-9e34-3f3b6260ecdc" />|<img width="240" height="180" alt="lake_h200_linfty_energy" src="https://github.com/user-attachments/assets/eecf7ec3-26a2-40f6-8e9e-7a9985999686" />|<img width="240" height="180" alt="lake_h200_linfty_minEnergy" src="https://github.com/user-attachments/assets/417f5c57-9cc5-495f-92b8-aa24bdb1beff" />|
|<img width="240" height="180" alt="lake_v200_linfty" src="https://github.com/user-attachments/assets/79ad0f3e-f0ef-41d8-8894-5188b89ddd4b" />|<img width="240" height="180" alt="lake_v200_linfty_energy" src="https://github.com/user-attachments/assets/4be839f0-b5a7-4dee-b7c3-d0f2fc7471b7" />|<img width="240" height="180" alt="lake_v200_linfty_minEnergy" src="https://github.com/user-attachments/assets/cbe67919-5ad6-4f15-8f8e-f96de46c2001" />|

Examples of resized images:

| Original | Resized | Reduction |
| :---: | :---: | :---: |
| <img src="./gallery/lake.jpg" height="225"/> | <img src="./gallery/lake_h200_linfty.jpg" height="225"/> | -200 px width |
| <img src="./gallery/lake.jpg" width="300"/> | <img src="./gallery/lake_v200_linfty.jpg" width="300"/> | -200 px height |
| <img src="./gallery/ioannina_clock_tower.jpg" width="300"/> | <img src="./gallery/ioannina_clock_tower_v200_l1.jpg" width="300"/> | -200 px height |
| <img src="./gallery/ioannina_clock_tower.jpg" height="400"/> | <img src="./gallery/ioannina_clock_tower_h200_l1.jpg" height="400"/> | -200 px width |
| <img src="./gallery/layla.jpg" width="300"/> | <img src="./gallery/layla_v150_linfty.jpg" width="300"/> | -150 px height |
| <img src="./gallery/pixie.jpg" height="224"/> | <img src="./gallery/pixie_h100_linfty.jpg" height="224"/> | -100 px width |
| <img src="./gallery/bedroom.jpg" width="300"/> | <img src="./gallery/bedroom_v100_linfty.jpg" width="300"/> | -100 px height |
| <img src="./gallery/a_message_from_Captain_Obvious.jpg" height="200"/> | <img src="./gallery/a_message_from_Captain_Obvious_h200.jpg" height="200"/> | -200 px width |
| <img src="./gallery/forest_panorama_small.jpg" height="132"/> | <img src="./gallery/forest_panorama_small_h200_l1.jpg" height="132"/> | -200 px width |
| <img src="./gallery/sea_Shell_in_Leyte.jpg" width="300"/> | <img src="./gallery/sea_Shell_in_Leyte_v300_linfty.jpg" width="300"/> | -300 px height |
| <img src="./gallery/milky_way.jpg" height="420"/> | <img src="./gallery/milky_way_h200_l1.jpg" height="420"/> | -200 px width |