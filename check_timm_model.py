import timm
# num_classes is set to 289 to match the usage in main.py
model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=289)
print(model)