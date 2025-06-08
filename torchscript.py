import torch
from torchvision import models

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("honey_detector_model.pt"))
model.eval()


scripted_model = torch.jit.script(model)
scripted_model.save("an_honey_detector_model.pt")