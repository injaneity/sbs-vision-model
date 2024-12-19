# sbs-vision-model
Using Roboflow's MoLa-VI in Car Detection API to detect stains, cuts and other damage to vehicles, applied to SBS buses as part of SBS Hackathon 2024.

### Disclaimer - why is it an API call?
The initial attempt was to finetune a YOLOv6 model on this dataset, and then test it against SBS-specific images. However, the key issue encountered was the unique pattern on the bus seats, which was causing the vision model to struggle with inference. Without a sufficiently large dataset of SBS bus seat images to train on, we were unable to yield any positive results on our limited test data.

This meant that finetuning on limited custom data was an unfeasible strategy, and for simplicty and ease of integration on a cloud hosted application, an API call to the pre-existing Roboflow model was effective enough -- given the time and resources, YOLOv6 could be explored for better overall efficacy.

**Therefore, this repository serves as a proof of concept for the vision portion of our vehicular defect detection system, working as one part of a larger SBS bus cleanliness system (other repositories will be linked when created.**
