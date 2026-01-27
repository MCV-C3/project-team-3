import torchvision.models as models

model = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
model.AuxLogits = None

layers = ["Mixed_7a", "Mixed_7b", "Mixed_7c"]

print("hola")
total = 0
for name, module in model.named_children():
    if name in layers:
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: {params:,}")
        total += params

print(f"\nTotal (Mixed_7a–7c): {total:,}")