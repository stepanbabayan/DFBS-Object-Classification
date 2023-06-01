import gradio as gr
import pandas as pd
import torch

import Classification_PyTorch.load_data as load_data
import Classification_PyTorch.models as models
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split


def train_model(dataset_file, output):
    pass
    # # Load dataset from file
    # df = pd.read_csv(dataset_file)
    #
    # # Split data into training and validation sets
    # train_df, val_df = train_test_split(df, test_size=0.2)
    #
    # # Create PyTorch DataLoader objects for training and validation sets
    # train_loader = DataLoader(train_df, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_df, batch_size=32)
    #
    # # Create PyTorch model and optimizer
    # model = Net()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    #
    # # Train model for 10 epochs
    # for epoch in range(10):
    #     # Set model to training mode
    #     model.train()
    #
    #     # Train model on batches of training data
    #     for x, y in train_loader:
    #         optimizer.zero_grad()
    #         output = model(x)
    #         loss = nn.CrossEntropyLoss()(output, y)
    #         loss.backward()
    #         optimizer.step()
    #
    #     # Set model to evaluation mode
    #     model.eval()
    #
    #     # Compute validation loss and accuracy
    #     val_loss = 0
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for x, y in val_loader:
    #             output = model(x)
    #             val_loss += nn.CrossEntropyLoss(reduction='sum')(output, y).item()
    #             pred = output.argmax(dim=1, keepdim=True)
    #             correct += pred.eq(y.view_as(pred)).sum().item()
    #             total += x.size(0)
    #     val_loss /= total
    #     val_acc = 100 * correct / total
    #
    #     # Display information about training process
    #     output.update(f"Epoch {epoch + 1}: Validation Loss={val_loss:.4f}, Validation Accuracy={val_acc:.2f}%")
    #
    # # Return trained model
    # return model


def test_model(
        model, dataset_file, output,
        data_root='data/Inference/Subtypes/19_2/images',
        checkpoint_path='Classification_PyTorch/Checkpoint/Dense_6_High_Focal_25_3_Final/136.pth',
        num_classes=6,
        batch_size=128,
        input_shape=(160, 50)
):

    # Load dataset from file
    df = pd.read_csv(dataset_file)

    # Create PyTorch DataLoader object for test set
    infer_data = load_data.load_images(path=data_root, batch_size=batch_size, domain='inference', _drop_last=False)

    # Create list to store predicted classes
    predicted_classes = []

    # Set model to evaluation mode
    net = models.Model(num_classes=num_classes, input_shape=input_shape).to(device)
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()

    # Make predictions on test set
    with torch.no_grad():
        for x, y in infer_data:
            output = model(x)
            pred = output.argmax(dim=1, keepdim=True)
            predicted_classes += pred.squeeze().tolist()

    # Create histogram of predicted classes
    counts, bins = np.histogram(predicted_classes, bins=np.arange(11) - 0.5)

    # Display information about testing process
    output.update("Testing complete.")
    output.append("Histogram of Predicted Classes:")
    output.append(counts)

    # Return file or output with predictions and histogram
    return {"predicted_classes.csv": pd.DataFrame({"predicted_class": predicted_classes})}


# Define the input and output interface with Gradio
inputs = [gr.inputs.File(label="Dataset CSV"),
          gr.inputs.Radio(["train", "test"], label="Mode")]

output_text = gr.outputs.Textbox()

# Create Gradio interface
iface = gr.Interface(fn=train_model if mode == "train" else test_model,
                     inputs=inputs,
                     outputs=output_text,
                     title="PyTorch Image Classification",
                     description="Train or test an image classification model using PyTorch.")

# Launch Gradio interface
iface.launch()
