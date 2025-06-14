---
menu:
  tutorials:
    identifier: pytorch
    parent: integration-tutorials
title: PyTorch
weight: 1
---
{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

Integrate Weights & Biases (W&B) with a PyTorch convolutional neural network (CNN) that trains on MNIST.

By completing the steps in this tutorial, you:

- Write a script that records hyperparameters and logs training metrics to W&B in real-time.
- Export the trained model to ONNX and track it as a versioned artifact in W&B.

## Prerequisites

- Python 3.8+
- A free [wandb.ai](https://wandb.ai) account

## Step 1: Install packages

Create a virtual environment and install the [`wandb` package](/ref/python/) and other dependencies using your favorite package manager.

{{< tabpane text=true >}}
{{% tab header="pip" %}}
```sh
python -m venv .venv
source .venv/bin/activate
pip install wandb torch torchvision tqdm onnx
```
{{% /tab %}}
{{% tab header="uv" %}}
```sh
uv init
uv add wandb torch torchvision tqdm onnx
```
{{% /tab %}}
{{< /tabpane  >}}

## Step 2: Create the training script

1. Create a file named `pytorch_tutorial.py` and open it in your editor.

2. At the top of the file, import the dependencies and add a call to [`wandb.login()`](/ref/python/login/).

   ```python
   import torch
   import torch.nn as nn
   import torchvision.datasets as dsets
   import torchvision.transforms as transforms
   import wandb
   from tqdm.auto import tqdm

   # Log in to W&B if you haven't already
   wandb.login()
   ```

   `wandb.login()` prompts for your API key once per machine and writes it to `~/.netrc` (on Windows, `%USERPROFILE%\.netrc`).

    Once that file exists, every script on the machine that calls `wandb.login()` authenticates automatically.

3. Create an object to provide experiment configuration during the run, and then initialize W&B.

   ```python
   # Experiment configuration
   config = dict(
       epochs=5,
       batch_size=128,
       learning_rate=0.005,
       kernels=[16, 32],
       dataset="MNIST",
       architecture="CNN",
       classes=10,
   )

   run = wandb.init(project="pytorch-tutorial", config=config)
   cfg = wandb.config
   ```

   [`wandb.init()`](/ref/python/init/) starts a new run and immediately logs every field in `config`, giving you a searchable record of the exact hyperparameters used.

## Step 3: Prepare the data, model, and optimizer

In this step, add the following code to set up the PyTorch components that W&B will observe: a dataset loader, a CNN, a loss function, and an optimizer.

W&B isn't involved here. This section of code is plain PyTorch, so you could reuse the same helpers in another project.

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data(train: bool = True, stride: int = 5):
    ds = dsets.MNIST(
        root=".", train=train, download=True, transform=transforms.ToTensor()
    )
    subset = torch.utils.data.Subset(ds, range(0, len(ds), stride))
    return subset

def make_loader(ds, *, batch_size: int):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7 * 7 * kernels[1], classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.fc(x.view(x.size(0), -1))
```

Add the code to build the data loaders, move the model to the GPU if one is available, and create the loss function and optimizer.

```python
train_set, test_set = get_data(True), get_data(False)
train_loader = make_loader(train_set, batch_size=cfg.batch_size)
test_loader = make_loader(test_set, batch_size=cfg.batch_size)

model = ConvNet(cfg.kernels, cfg.classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
```

## Step 4: Log training metrics

Use [`wandb.watch()`](/ref/python/watch/) to record gradients and `wandb.log()` to send metrics.

```python
wandb.watch(model, criterion, log="all", log_freq=10)
```

`wandb.watch()` instruments the model to log parameter and gradient histograms every `log_freq` steps, which makes it easier for you to spot issues like vanishing gradients or exploding weights.

```python
for epoch in range(cfg.epochs):
    for step, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        if step % 25 == 0:
            wandb.log({"epoch": epoch, "loss": loss.item()},
                      step=epoch * len(train_loader) + step)
```

[`wandb.log()`](/ref/python/log/) streams arbitrary key-value metrics for the current run to W&B. Logging the loss every 25 minibatches lets you follow convergence live in the W&B dashboard instead of scanning console output.

## Step 5: Evaluate and save the model

Log accuracy and persist the model as an artifact.

```python
def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).argmax(1)
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
    return correct / total

accuracy = evaluate(test_loader)
wandb.log({"test_accuracy": accuracy})
print(f"Test accuracy: {accuracy:.2%}")
```

`wandb.log()` records the final test accuracy so you can quickly compare runs in the project table view.

```python
torch.onnx.export(model, next(iter(test_loader))[0].to(device), "model.onnx")
wandb.save("model.onnx")
```

[`wandb.save()`](/ref/python/save/) uploads `model.onnx` and versions it as an artifact, giving you and your team a permanent link to download or visualize the network.

```python
# Clean up
run.finish()
```

[`run.finish()`](/ref/python/finish/) flushes any remaining logs and marks the run as complete in the dashboard.

## Step 6: Run the script

Execute the script to run the experiment and publish the results to W&B.

To view the run during the experiment and view its results, navigate to the W&B dashboard URL that should appear in the output, similar to the following:

```sh
$ python pytorch_tutorial.py
wandb: Currently logged in as: wandbcustomer (wandbcustomer-apps) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.20.1
wandb: Run data is saved locally in /Users/wandbcustomer/repos/wandb-tut-pytorch/wandb/run-20250613_135842-00000000
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run wandering-serenity-6
wandb: ⭐️ View project at https://wandb.ai/wandbcustomer-apps/pytorch-tutorial
wandb: 🚀 View run at https://wandb.ai/wandbcustomer-apps/pytorch-tutorial/runs/00000000
...
```

For example, the **gradients** section of a run might look similar to the following:

{{< img src="images/tutorials/pytorch-01-workspace-gradients.png" alt="" >}}

## Recap

In this tutorial, you:

- **Installed and authenticated to W&B** so your scripts can sync data securely.
- **Created a PyTorch training script** that loads MNIST, builds a small CNN, and trains on GPU or CPU.
- **Instrumented the script with three W&B calls**, `wandb.init()`, `wandb.watch()`, and `wandb.log()` to capture hyperparameters, gradients, and metrics in real-time.
- **Exported the model to ONNX and versioned it as an artifact** for easy sharing or deployment.
- **Viewed live results in the W&B dashboard**, where you can compare runs and inspect gradients, system metrics, and logged files.

## Next steps

You've just implemented the minimal pattern for adding experiment tracking to any PyTorch project with W&B. Continue with the [PyTorch Lightning tutorial]({{< relref "lightning.md" >}}), where you build and track a CIFAR-10 image-classification pipeline using Lightning’s `DataModule`, `LightningModule`, and W&B logging.
