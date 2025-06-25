# AI Harmonizer

This is the implementation of our paper [AI Harmonizer: Expanding Vocal Expression with a Generative Neurosymbolic Music AI System](https://nime.org/proc/nime2025_84/index.html). It is based on the amazing project [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

> [!CAUTION]
> This repository uses by default an [Anticipatory Music Transformer](https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html) (AMT) finetuned on the [JSB Chorales](https://github.com/lancelotblanchard/JSB-Chorales-dataset-midi) dataset, which is accessible here: [https://huggingface.co/mitmedialab/jsbChorales-1000](https://huggingface.co/mitmedialab/jsbChorales-1000). As such, it is <span style="color: #ba473e">**heavily biased towards baroque music**</span>. If you would like to explore other genres, we invite you to finetune AMT on another four-part harmony dataset.

## How to Use

1. Make sure that you clone this repository along with its submodules:
```
git clone --recurse-submodules https://github.com/mitmedialab/ai-harmonizer-nime2025.git
```

2. Install voice models following the [instructions of the RVC project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md).

3. Run the `run.sh` script.
```
./run.sh
```

4. In the Gradio interface that opens up, select your voice model, load an audio file, and click "Convert!"

## Dependencies

This project is made possible thanks to:

- [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Basic Pitch](https://basicpitch.spotify.com)
- [Anticipatory Music Transformer](https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html)

## Citation

```
@article{nime2025_84,
  title = {AI Harmonizer: Expanding Vocal Expression with a Generative Neurosymbolic Music AI System},
  author = {Lancelot Blanchard and Cameron Holt and Joseph Paradiso},
  booktitle = {Proceedings of the International Conference on New Interfaces for Musical Expression},
  address = {Canberra, Australia},
  articleno = {84},
  doi = {10.5281/zenodo.15698966},
  editor = {Doga Cavdir and Florent Berthaut},
  issn = {2220-4806},
  month = {June},
  numpages = {4},
  pages = {578--581},
  track = {Paper},
  url = {http://nime.org/proceedings/2025/nime2025_84.pdf},
  year = {2025}
}
```