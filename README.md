# face-attribute-classifier

A Python project that uses CLIP-based models to classify human face attributes such as age, gender, and race from images.

## Disclaimer

This project is intended as an exploration of how machine learning models analyze and classify human facial attributes. The classifications produced by the models are not definitive or absolute judgments of any individual's identity, gender, race, or age. Human identity is complex and cannot be fully captured by automated systems. This tool should not be used for making personal, legal, or ethical decisions. It aims to provide insight into model behavior and performance, not to categorize or label people in a conclusive way.

## Features

- Classifies faces by age, gender, and race using a custom multitask CLIP vision model.
- Organizes classified images into folders based on predicted attributes.
- Uses a multitask vision transformer (CLIP-based) model fine-tuned for facial attribute classification, via Hugging Face.
- Logs detailed processing info including errors and statistics.

## Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/face-attribute-classifier.git
cd face-attribute-classifier
````

2. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Place images you want to classify into the `images/` folder at the project root.

2. Run the classifier module from the project root:

```bash
python3 -m src.main
```

3. Check the `results/` folder for classified images organized by gender, race, and age.

4. Logs are saved in the `logs/` folder.

## Project Structure

```
face-attribute-classifier/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── paths.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── multitask_clip_vision_model.py
├── images/
├── results/
├── logs/
├── venv/
├── requirements.txt
├── README.md
└── LICENSE
```

## Model Attribution

This project uses the following models and libraries:

* [syntheticbot/clip-face-attribute-classifier](https://huggingface.co/syntheticbot/clip-face-attribute-classifier)
  A multitask CLIP-based vision model for facial attribute classification.

* [Hugging Face Transformers](https://github.com/huggingface/transformers)
  For model loading and processor utilities.

* [OpenAI CLIP](https://github.com/openai/CLIP)
  The original vision-language backbone architecture underlying the model.

---

## Image Attribution

All demonstration images are sourced from [Unsplash](https://unsplash.com) and are licensed under the [Unsplash License](https://unsplash.com/license).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
