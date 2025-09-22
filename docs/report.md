# BIN23 Group 3 - MRI Brain Tumor classifier

- Group Member: Rubin Chempananickal James (D876)
- Supervisor: Dr. Lamya Abdullah
- Course: Projektpraktikum (SS 2025)
- University: Provadis Hochschule

## Project Overview
The aim of this "Projektpraktikum" project is to develop an MRI brain tumor classifier using deep learning techniques. The project focuses on creating a user-friendly application that can accurately classify brain MRI images into different tumor categories and provide visual explanations for its predictions.

## Software Development Methodology
The project followed a waterfall/agile hybrid model approach, as a one man development team (with assistance from a GitHub Copilot AI assistant). The development process was carried out with a clear end goal, but with flexibility to adapt to changes and improvements based on scope and requirements. 

Originally it was meant to be a pure waterfall model building itself atop a preexisting codebase, but that hit a snag real quick (more on that later). 

The project was divided into distinct phases, including requirements gathering, design, implementation, testing, and deployment. The development process involved iterative cycles of coding, testing, and refining the application to ensure its functionality and usability. 

The project did not require any sprints or standups, as it was a solo effort. Progress was steady with just four hiccups along the way (see docs/adrs in the repository). Hence it wasn't strictly waterfall, but it wasn't strictly agile either.

To keep track of big sweeping changes, ADRs (Architecture Decision Records) were used, documented in the [`docs/adrs`](https://github.com/chempananickal/Tumor_Classifier/blob/main/docs/adrs) directory in the repository.

## Key Requirements
1. A frontend application where the user can upload an MRI image and receive a classification result.
2. Some kind of backend UNET model that can classify the image into one of multiple tumor categories.
3. A Grad-CAM implementation to provide visual explanations for the model's predictions (i.e., highlight areas of the image that influenced the decision using a heatmap).

## Original Plan
The original plan was to build the project on top of an existing repository [(Hybrid Vision U-Net)](https://doi.org/10.1038/s41598-025-09833-y) that provided an advanced UNET model for brain tumor segmentation and classification. The idea was to leverage this existing model and just build a frontend application around it.

### The Snag
However, upon closer inspection, it was found that the existing repository was not well-documented and had several issues that made it difficult to use as a foundation for the project. Among other things:

- The repository didn't come with a pre-trained model, and training it from scratch would have been too time-consuming and computationally expensive for the scope of this project.
- The repository didn't have a requirements.txt or environment.yml file, making it hard to set up the development environment.
- The code was far from modular, requiring extensive rewrites.
- The repository lacked a license, creating legal uncertainty around its use and modification.

So, I decided to pivot to a simpler approach by post-training a small Image Classification model and writing the Grad-CAM code myself. The project scope was adjusted accordingly, focusing on slice-level classification of the [Figshare dataset](https://figshare.com/articles/figure/Brain_tumor_dataset/27102082) with Grad-CAM visualization instead of worrying about segmenting a 3D volume MRI scan.

For more details about the problems encountered and the reasons for the pivot, see [ADR000](https://github.com/chempananickal/Tumor_Classifier/blob/main/docs/adrs/ADR-000-move-away-from-hvu.md). This document was called ADR000 because I made these decisions before writing a single line of code.

## The Dataset
The dataset used for training and testing the model is the [Figshare brain tumor dataset](https://figshare.com/articles/figure/Brain_tumor_dataset/27102082) by Shiraz Afzal.

This dataset contains 2868 MRI images of brain tumors, categorized into four classes: Glioma, Meningioma, Pituitary tumors, and No Tumor. The images are in JPEG format and vary in size and quality. 

The distribution of classes in the dataset is as follows:
- Glioma: 826 images
- Meningioma: 822 images
- Pituitary: 825 images
- No Tumor: 395 images

## Revised Plan
Since I'm vaguely familiar with PyTorch (unlike TensorFlow), I decided to use it for the machine learning components. 
This also made it easy to package the weight files, as PyTorch has a simple API for saving and loading model weights (see save_checkpoint in [`scripts/train.py`](https://github.com/chempananickal/Tumor_Classifier/blob/main/scripts/train.py)).

After a cursory amount of research, I decided to use a DenseNet121 architecture for the classification model. [DenseNet121](https://arxiv.org/abs/1608.06993) is a well-known convolutional neural network architecture that has been shown to perform well on image classification tasks. The Grad-CAM code looks at the last dense block in the DenseNet121 architecture to generate the heatmaps. This also had the advantage of being a relatively small model, making it feasible to train on an underpowered laptop using only the CPU. The final [weights file](https://github.com/chempananickal/Tumor_Classifier/blob/main/models/weights/best.pt) is also just 80 MB, making it easy to share and deploy (I just packaged it with the repo).

More details about the decision to use DenseNet121 is documented in [ADR001](https://github.com/chempananickal/Tumor_Classifier/blob/main/docs/adrs/ADR-001-densenet-baseline.md).

The project was implemented using the following key components:
- **PyTorch**: For building and training the DenseNet121 model. You can find the model code in [`models/unet_densenet.py`](https://github.com/chempananickal/Tumor_Classifier/blob/main/models/unet_densenet.py).
- **Streamlit**: For creating the frontend application.
- **Grad-CAM**: For generating visual explanations for the model's predictions. The Grad-CAM implementation was done from scratch, and almost entirely written by GitHub Copilot.
- **Matplotlib**: For visualizing the Grad-CAM heatmaps.
- **OpenCV**: For image processing tasks, such as resizing and blending heatmaps with the original images.
- **PIL (Pillow)**: For image loading and basic manipulations.
- **Torchvision**: For data transformations and augmentations.
- **Numpy**: To handle array operations on the image (these are used in pre-processing because numpy arrays are substantially less computationally intensive than PyTorch tensors for simple image manipulations like background subtraction and corner masking).

### The Other Snag
At first, the model confidently classified every item in a different class than the one it was supposed to. After some debugging, I found that the class ordering in the dataset was different from the hardcoded class ordering in the code. This was a simple fix, but it took way too long to figure out. 
More details in [ADR002](https://github.com/chempananickal/Tumor_Classifier/blob/main/docs/adrs/ADR-002-class-order-persistence.md)

### One Last Hurdle
Occasionally, the model would get the classification extremely wrong. Thanks to Grad-CAM, I could see that the model was focusing on irrelevant parts of the image (like the corners or background) instead of the tumor region. To fix this, I added some simple preprocessing steps to flatten out the corners and subtract the background using a basic thresholding technique. This significantly improved the model's performance. Goes to show how useful Grad-CAM can be for debugging and improving model performance.
More details in [ADR003](https://github.com/chempananickal/Tumor_Classifier/blob/main/docs/adrs/ADR-003-preprocessing-strategy.md)

## Final Deliverables
The final deliverables of the project include:
1. A fully functional Streamlit application that allows users to upload MRI images and receive classification results.
2. A trained DenseNet121 model capable of classifying brain MRI images into different tumor categories.
3. A Grad-CAM implementation that provides visual explanations for the model's predictions.
4. Documentation detailing the development process, challenges faced, and lessons learned.

All the code, documentation, and resources related to the project are available in the [GitHub repository](https://github.com/chempananickal/Tumor_Classifier).

#### The App
After cloning the repository and creating a conda environment based on the provided `environment.yml` file, the app can be run locally using the command:
```
conda activate tumor
streamlit run app/main.py
```
The app provides a simple interface for uploading MRI images and displays the classification results along with the Grad-CAM heatmaps. It even gives a direct link to download the dataset where on can find the images to test the app with.

#### The Model
If you wish to retrain the model, there is a simple training script provided in `scripts/train.py`. Read the [README](https://github.com/chempananickal/Tumor_Classifier/blob/main/README.md) for more details on how to prepare the dataset and run training. The epochs, learning rate, and batch size can be adjusted via command line arguments if one wishes to fine-tune the model further.

## Limitations
The project has several limitations, which are documented in detail in the [`docs/limitations.md`](https://github.com/chempananickal/Tumor_Classifier/blob/main/docs/limitations.md).

## Lessons Learned
1. Just because it was in a well-known journal, doesn't mean the code is high quality. The concept behind the Hybrid Vision UNET is remarkable and the people who developed it are undoubtedly talented. However, the unfortunate lack of proper documentation made the results impossible to reproduce.

2. Explainable AI (XAI) techniques like Grad-CAM are incredibly useful for debugging and improving model performance. They provide insights into what the model is focusing on, allowing for targeted improvements. Wouldn't have been able to fix the corner/background issue without it.

3. Preprocessing can have a significant impact on model performance. Simple techniques like background subtraction and corner masking can help the model focus on relevant features.

4. Always verify class orderings and label mappings when working with datasets. Mismatches can lead to confusing results and wasted time.

5. Document decisions and changes using ADRs. This helps keep track of the rationale behind choices and makes it easier to revisit and understand them later.

6. GitHub Copilot Agent mode can be trusted with small coding tasks and with helping fix bugs and writing documentation, but it cannot handle complex tasks. Keeping a copilot-instructions.md file in the repository helps it avoid overreach and also enables transparency.

7. Time management is crucial. I grossly overestimated the complexity of this project while simultaneously underestimating my ability to procrastinate.

## Glossary
- **ADRs**: Architecture Decision Records, documents that capture important architectural decisions made during a project, along with their context and consequences.
- **Checkpoint**: A saved state of a machine learning model during training, allowing for resumption or evaluation at a later point. Here, the best checkpoint is saved based on validation accuracy and used for inference.
- **DenseNet121**: A pre-trained convolutional neural network architecture known for its efficiency and performance on image classification tasks.
- **Environment.yml**: A file used by Conda to create a virtual environment with specific packages and dependencies.
- **Grad-CAM**: Gradient-weighted Class Activation Mapping, a technique for visualizing the regions of an input image that were designated as important by a convolutional neural network for making a specific prediction.
- **ImageNet**: A large visual database designed for use in visual object recognition research, often used for pre-training deep learning models. The mean and standard deviation of ImageNet images are commonly used for normalizing input images in transfer learning.
- **MRI**: Magnetic Resonance Imaging, a medical imaging technique used to visualize detailed internal structures, particularly soft tissues like the brain.
- **Preprocessing**: Techniques applied to raw data to prepare it for use in a machine learning model, such as normalization, resizing, and augmentation.
- **PyTorch**: An open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.
- **Streamlit**: An open-source app framework for Machine Learning and Data Science teams to create web apps easily.
- **Tumor Types**: 

    The model classifies MRI images into four categories:
  - **Glioma**: A type of tumor that occurs in the brain and spinal cord, originating from glial cells.
  - **Meningioma**: A tumor that arises from the meninges, the membranes that surround the brain and spinal cord.
  - **Pituitary Tumor**: A growth that occurs in the pituitary gland, which can affect hormone production and regulation.
  - **No Tumor**: Indicates the absence of any detectable tumor in the MRI image.
- **UNET**: A type of convolutional neural network architecture commonly used for image segmentation tasks, characterized by its U-shaped design with encoder and decoder paths.

## 16. AI Disclosure

NOTE: This disclosure is also available in the [README](https://github.com/chempananickal/Tumor_Classifier/blob/main/README.md) file in the repository.

GitHub Copilot (GPT 5) has been used in the creation of this project. All information available to GitHub Copilot during Agent Mode is available under .github/copilot-instructions.md.

In addition to writing everything in tests/ and contributing significantly to the files in docs/ (specifically architecture.md and the three ADRs), GitHub Copilot was prompted to help with bug fixes and feature improvements. The following prompts were used:

- Prompt: "The data is currently in the format data/pituitary_tumor, data/no_tumor, etc... Write a script that turns it into a format that your scripts expect. The shuffling into train, val and test should be random but reproducible (seed)"

	- LLM Response: --created scripts/prepare_dataset.py following my instructions--

- Prompt:"Check through the entire training regimen. It seems to classify everything exactly wrong in the same way. As in, pituitary is predicted as negative, meningioma as glioma, glioma as pituitary, and negative as meningioma. My data is in the correct folders. It's some slight oversight in the training code"

	- LLM Response: "You were seeing a consistent “rotated” misclassification because the class ordering used at inference didn’t match the dataset’s true class index ordering during training. ImageFolder assigns class indices alphabetically based on folder names. Your earlier CLASSES constant (and possibly a modified version) didn’t match that ordering. So each logit index was being interpreted as the wrong semantic class, producing a stable permutation error (a cycle mapping each class to another). The pattern you described (pituitary -> negative, meningioma -> glioma, glioma -> pituitary, negative -> meningioma) is exactly what happens when label interpretation is shifted by a permutation."

- Prompt: "How do I teach the model to stop taking the parts outside the MRI into account? GRAD CAM shows the reason for false negatives being it looking at the corners of the image, not the bounds of the brain".

	- LLM Response: --added RandomCornerMask to app/preprocessing.py, fixed __call__ method in BrainCrop--

- Prompt: "Make the training a bit more verbose. Right now, I don't even know if it's running until an entire epoch is completed"

	- LLM Response: --added batch-level verbose logging to scripts/train.py--

- Prompt: "Look through my code, construct an architecture overview document for docs/architecture.md"

	- LLM Response: --created docs/architecture.md following my instructions--

Additionally, automatic code completion has been used during the code writing process (GPT 4.1). This model also assisted in writing the initial version of this README.md file and helps write commit messages.

Every substantial instance of AI assistance, whether by code completion or otherwise, has been disclosed in comments in the code itself with the NOTE tags `# NOTE: AI Generated` or `# NOTE: Fixed by AI`.

I hereby certify that I have disclosed the use of AI assistance in this project to the best of my ability.
