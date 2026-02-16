# Seam Carving for Content-Aware Image Resizing

An octave implementation of the classic image resizing algorithm:

> Avidan, S., Shamir, A.
> *Seam carving for content-aware image resizing.*
> *ACM Transactions on Graphics* 26, 10 (2007).
> [https://doi.org/10.1145/1276377.1276390](https://doi.org/10.1145/1276377.1276390)

Vertical/horizontal seams with the lowest energy are iteratively removed until the target width/height is reached.

Requires the `image` (`pkg install -forge image`) and `video` (`pkg install -forge video`) packages.

```console
➜ octave functions/main.m
Usage: octave main.m <input> <output> <reduction direction {v|h}> <amount> <p for L_p or 0 for L_infinity> [--animations]
```

## Gallery

<table>
  <tr>
    <th>Image</th>
    <th>Energy</th>
    <th>Minimum Energy Seams</th>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/ioannina_clock_tower_h200_l1.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/ioannina_clock_tower_h200_l1_energy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/ioannina_clock_tower_h200_l1_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/ioannina_clock_tower_v200_l1.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/ioannina_clock_tower_v200_l1_energy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/ioannina_clock_tower_v200_l1_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/lake_h200_linfty.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/lake_h200_linfty_energy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/lake_h200_linfty_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/lake_v200_linfty.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/lake_v200_linfty_energy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/lake_v200_linfty_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/layla_v150_linfty.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/layla_v150_linfty_energy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/layla_v150_linfty_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/pixie_h100_linfty.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/pixie_h100_linfty_energy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/pixie_h100_linfty_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/bedroom_v100_linfty.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/bedroom_v100_linfty_energy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/bedroom_v100_linfty_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/a_message_from_Captain_Obvious_h200.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/a_message_from_Captain_Obvious_h200_energy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/a_message_from_Captain_Obvious_h200_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/forest_panorama_small_h200_l1.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/forest_panorama_small_h200_l1_energy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/forest_panorama_small_h200_l1_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/sea_Shell_in_Leyte_v300_linfty.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/sea_Shell_in_Leyte_v300_linfty_energy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/sea_Shell_in_Leyte_v300_linfty_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/milky_way_h200_l1.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/milky_way_h200_l1_energy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/milky_way_h200_l1_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <tr>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/milky_way_v200_l1.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/milky_way_v200_l1_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
    <td>
      <video width="480" autoplay loop muted>
        <source src="./gallery/milky_way_v200_l1_minEnergy.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

</table>
