# Introducing DiffMM with Modality-Aware Dynamic Contrastive Learning
 In the existing DiffMM framework, current methods apply fixed contrastive weights to all modalities, ignoring their dynamic importance to user preferences. My improvement enhances the contrastive learning mechanism through adaptive weight allocation, dynamically adjusting contrastive weights for different modalities. This strengthens the contrastive signals of important modalities while suppressing the influence of noisy ones.
 
 The original DiffMM project can be found at https://github.com/HKUDS/DiffMM, readers can find instructions on how to run this code in the README provided by the DiffMM authors at the given link. The corresponding paper is available at arxiv.org/abs/2406.11781.
 
 In the code I uploaded, the datasets Datasets\baby\image_feat.npy and Datasets\sports\image_feat.npy could not be uploaded locally due to their size exceeding 100MB. Readers can find Datasets\baby\image_feat.npy in the original project and download Datasets\sports\image_feat.npy from Google Drive using the link provided in the dataset's README file.
