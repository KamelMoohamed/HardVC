import torch

from models.synthesizer import SynthesizerTrn

# Constants from provided config
spec_channels = 80
segment_size = 8960
inter_channels = 512
hidden_channels = 512
resblock = "1"
resblock_kernel_sizes = (3, 7, 11)
resblock_dilation_sizes = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
upsample_rates = (10, 8, 2, 2)
upsample_initial_channel = 512
upsample_kernel_sizes = (16, 16, 4, 4)
gin_channels = 512
ssl_dim = 1024

# Initialize the model
model = SynthesizerTrn(
    spec_channels=spec_channels,
    segment_size=segment_size,
    inter_channels=inter_channels,
    hidden_channels=hidden_channels,
    resblock=resblock,
    resblock_kernel_sizes=resblock_kernel_sizes,
    resblock_dilation_sizes=resblock_dilation_sizes,
    upsample_rates=upsample_rates,
    upsample_initial_channel=upsample_initial_channel,
    upsample_kernel_sizes=upsample_kernel_sizes,
    gin_channels=gin_channels,
    ssl_dim=ssl_dim,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = model.to(device)

# Set model to training mode
model.train()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define loss function
criterion = torch.nn.MSELoss()

# Training loop parameters
epochs = 5
batch_size = 4  # Arbitrary batch size

for epoch in range(epochs):
    optimizer.zero_grad()

    # Generate random input tensors
    c = torch.randn(batch_size, ssl_dim, 100, device=device)
    spec = torch.randn(batch_size, spec_channels, 100, device=device)
    c_lengths = torch.randint(50, 100, (batch_size,), device=device)
    spec_lengths = torch.randint(50, 100, (batch_size,), device=device)

    # Perform a forward pass
    output, ids_slice, spec_mask, latent_vars = model(
        c,
        spec,
        filenames=[
            "data/U2U_source.wav",
            "data/U2U_target.wav",
            "data/U2U_source.wav",
            "data/U2U_target.wav",
        ],
        c_lengths=c_lengths,
        spec_lengths=spec_lengths,
    )

    # Create a dummy target tensor with the same shape as output
    target = torch.randn_like(output)

    # Compute loss
    loss = criterion(output, target)

    # Perform backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
