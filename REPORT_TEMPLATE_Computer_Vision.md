# REPORT — Module 3 · Assignment 4 · Computer Vision

**Name:**Ravit Bar-Lev  **ID:**029290400  **Date:** 08/08/26
**Chosen option:** (A · PlantVillage

> Keep this report in English. 99% on lab images proves nothing about a field photo.
> The reality check is the heart of this assignment.

---

## 1. Framing
Business question and metric: Early Identification of infected leaves / Plant Diseases for Farmers

Why transfer learning rather than training from scratch:
1. Save time and Resources
2. Consume less data
3. High Accuracy
   
---

## 2. Results
| Split | Accuracy | Notes |
|---|---|---|
| lab test set (overall) |88.21% |excellent performance on lab data |
| worst classes |significantly low | Rare diseases demonstrate low recall percentage|
| real-world / field images |0% | Total collapse|

---

## 3. Guiding questions (graded)
Answer each in 2-5 sentences.

1. **Why transfer learning.** Why not train from scratch? What does the pretrained backbone give you?
   Refer to clause #1 
2. **Per-class.** Which classes does it confuse, and why (visual similarity, class imbalance)?
BOTH!! In the confusion matrix, many other completely unrelated classes are getting mistargeted into these specific folders
3. **Lab-to-field.** Run it on real-world images. What happened, and why? (This is the core of the assignment.)
The steep drop in confidence and prediction accuracy is a textbook case of Domain Shift (Data Distribution Mismatch) between
the training environment and reality:
   -The "Lab Versus Wild" Effect: The PlantVillage dataset consists exclusively of clean, isolated single leaves shot against
   an artificially uniform, flat, and neutral background. In contrast, the real-world field images contain complex,
   uncontrolled noisy elements.
   -Spurious Correlations: During training, the convolutional layers likely latched onto shortcut features such as the sharp,
    clean edges of the isolated leaves or the specific contrast ratios against the gray studio background.
   -Illumination and Shadow Artifacts: Real-world conditions introduce harsh natural sunlight, dynamic range issues, and deep
   shadows. 
4. **Augmentation.** Which augmentations did you use, and did they help generalization? Show numbers.
   The pipeline utilized a specific combination of geometric and color-space transformations during
   training:transforms. Resize((224,224)): Standardizes input dimensions. transforms. RandomHorizontalFlip(): Mirror-images the
   leaves. transforms. RandomRotation(15): Tilts the leaf position slightly. transforms. ColorJitter(0.2, 0.2, 0.2): Shifts
   brightness, contrast, and saturation by up to 20%.
5. **Overfitting.** Learning curves — is the model memorizing the clean lab background?
   Yes, the model is absolutely memorizing the clean lab background. 
6. **The gap.** What specifically differs between lab and field images that breaks the model?
   The model breaks because of a phenomenon called Domain Shift (or Covariate Shift). A convolutional neural network (CNN)
   like ResNet18 learns by finding the easiest mathematical patterns (shortcuts) that help it minimize loss during training.
7. **Real world.** What data and steps would you need to make this work in a farmer's hand?
   To transition this model from a "lab" into a reliable tool that works in a farmer's hand, I must fundamentally change the
   model’s data and its training environment (the workflow). The goal is to force the neural network to develop background-
   invariance and illumination-robustness.
   I would also replace the PlantVillage data with a distribution that mirrors actual field
8. **Cost of error.** In agriculture, what is the cost of a false "healthy" vs a false "diseased"? How should the threshold reflect that?
In agriculture, the costs of these two types of statistical errors are highly asymmetric. A false "healthy" is almost always far more damaging than a false "diseased".
A false "healthy" occurs when the model tells a farmer a plant is fine, but it is actually infected.
A false "diseased" occurs when the model flags a completely healthy plant as sick.
---

## 4. Model Card (lab-to-field)
CV_MODEL_CARD = """
## 1. Overview
- Option / dataset / classes: PlantVillage dataset, focusing on the "color" folder subset containing 38 distinct classes of plant species and specific crop diseases.
- Backbone and what you froze vs fine-tuned: ResNet18 (pretrained on ImageNet). Initially, the entire feature extraction backbone was frozen (`requires_grad=False`), and a brand-new, untrained linear classifier (`model.fc`) was appended to match the 38 output classes. In the subsequent phase, the entire model was trained jointly with a low learning rate (`1e-4`).

## 2. Performance (lab test set)
- Overall accuracy: 88.21% (Achieved on the clean, stratified validation/test split at Epoch 3).
- Worst classes (confusion matrix) and why: Rare diseases and structurally similar species (such as different types of fungal spots or molds on the same host plant, e.g., Tomato molds). They suffer from high inter-class similarity in visual textures, causing the model to misclassify them across adjacent rows in the confusion matrix.

## 3. The reality check
- Accuracy on real field images: 0% correct classification with critically low confidence scores ranging between 15.45% and 22.12%.
- What specifically broke between lab and field? Extreme Domain Shift. The model suffered from a complete breakdown of feature generalization. In the lab, it relied on spurious correlations like the flat, gray background and pristine leaf silhouettes. In the field, it encountered complex "noise" including wild backgrounds (soil, weeds, equipment), overlapping foliage, harsh natural sunlight, dynamic shadows, and human hands holding the specimens.

## 4. Limitations & ethics
- Cost of a false 'healthy' vs a false 'diseased' in agriculture: 
  - A false 'healthy' (False Negative) is catastrophic: a farmer misses a disease outbreak, leading to crop death, rapid field-wide spreading, and devastating financial or food-supply loss.
  - A false 'diseased' (False Positive) leads to economic waste: unnecessary and expensive chemical spraying, environmental soil/water pollution, and forced destruction of perfectly viable crops.
- Who could be harmed if this were deployed as-is? Smallholder farmers and agricultural communities who rely entirely on their seasonal yield. Relying on an uncertain model with ~18% confidence could ruin their livelihood due to mismanaged crop treatments.

## 5. Real world
- What data and steps would make this work in a farmer's hand?
  1. **Data collection**: Source a diverse, "dirty" dataset captured in-the-wild, such as the PlantDoc dataset, featuring varying illumination, resolutions, and backgrounds.
  2. **Augmentation Pipeline**: Implement aggressive training augmentations like `RandomResizedCrop`, `ColorJitter`, and synthetic background blending to force background-invariance.
  3. **Operational Thresholding**: Hardcode a deployment guardrail that rejects any classification with a confidence score under 75%, instructing the farmer to take a clearer, closer photo instead of providing a random guess.
"""

```
(Model Card)
```

---

## 5. Reflection
What surprised you in the reality check? What would it take to make this trustworthy in the field?
