import torch


def get_model_details(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    if "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint

    layer_details = []
    total_params = 0
    total_size = 0

    for name, param in model_state.items():
        layer_shape = list(param.shape)
        param_count = param.numel()
        param_size = param.element_size() * param_count

        total_params += param_count
        total_size += param_size

        layer_details.append(
            f"Layer: {name}, Shape: {layer_shape}, Params: {param_count}, Size (bytes): {param_size}"
        )

    return layer_details, total_params, total_size


def write_model_info(
    checkpoint1_path, checkpoint2_path, output_file="model_details.txt"
):
    model1_details, model1_params, model1_size = get_model_details(checkpoint1_path)
    model2_details, model2_params, model2_size = get_model_details(checkpoint2_path)

    with open(output_file, "w") as f:
        f.write(f"Model 1: {checkpoint1_path}\n")
        f.write(
            f"Total Parameters: {model1_params}, Total Size: {model1_size / 1024:.2f} KB\n"
        )
        f.write("\n".join(model1_details))
        f.write("\n\n")

        f.write(f"Model 2: {checkpoint2_path}\n")
        f.write(
            f"Total Parameters: {model2_params}, Total Size: {model2_size / 1024:.2f} KB\n"
        )
        f.write("\n".join(model2_details))
        f.write("\n")


# Example usage:
checkpoint1 = "checkpoints/freevc.pth"
checkpoint2 = "checkpoints/best_model.pth"
write_model_info(checkpoint1, checkpoint2)
