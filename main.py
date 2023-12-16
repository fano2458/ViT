from torchinfo import summary
from src.vit import ViT


vit = ViT(num_classes=10)

summary(model=vit,
        input_size=(32, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
