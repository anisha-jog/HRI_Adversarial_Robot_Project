# %% IMPORTS
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
# %% Create Dataset Class
class PartialImageDatasetPreLoad(torch.utils.data.Dataset):
    def __init__(self, image_df : pd.DataFrame):
        self.label_list = image_df['word'].unique().tolist()
        self.raw_data = image_df
        self.stroke_width = 1
        self.partial_img_data = []
        self.partial_img_label = []
        self.partial_img_next = []
        blank_img = np.full((256, 256),255, dtype = np.uint8)
        for idx, drawing in enumerate(image_df['drawing']):
            img = blank_img.copy()
            strokes = json.loads(drawing)
            for stroke_num in range(len(strokes)-1):
                img = self.add_stroke_to_image(img, strokes[stroke_num])
                self.partial_img_data.append(img.copy())
                self.partial_img_label.append(image_df['word'][idx])
                just_next_stroke = self.add_stroke_to_image(blank_img, strokes[stroke_num+1])
                self.partial_img_next.append(just_next_stroke)
        self.partial_img_data = np.array(self.partial_img_data)
        self.partial_img_label = np.array(self.partial_img_label)
        self.partial_img_next = np.array(self.partial_img_next)
        assert self.partial_img_data.shape[0] == self.partial_img_label.shape[0] == self.partial_img_next.shape[0]

    def add_stroke_to_image(self, image, stroke):
        for i in range(len(stroke[0])-1):
            cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]), (0,0,0), self.stroke_width)
        return image

    def __len__(self):
        return self.partial_img_data.shape[0]

    def __getitem__(self, idx):
        return self.partial_img_data[idx], self.partial_img_label[idx], self.partial_img_next[idx]

class PartialImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_df : pd.DataFrame):
        super().__init__()
        self.label_list = image_df['word'].unique().tolist()
        self.raw_data = image_df
        self.stroke_width = 1
        self.cum_sum_strokes = []
        for drawing in (image_df['drawing']):
            strokes = json.loads(drawing)
            self.cum_sum_strokes.append(len(strokes)-1)
        self.cum_sum_strokes = np.cumsum(self.cum_sum_strokes)

    def get_subidx(self, idx):
        sample_idx = np.searchsorted(self.cum_sum_strokes, idx, side='right')
        if sample_idx == 0:
            stroke_idx = idx
        else:
            stroke_idx = idx - self.cum_sum_strokes[sample_idx-1]
        return sample_idx, stroke_idx+1


    def add_stroke_to_image(self, image, stroke):
        for i in range(len(stroke[0])-1):
            cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]), (0,0,0), self.stroke_width)
        return image

    def __len__(self):
        return self.cum_sum_strokes[-1]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
        else:
            sample_idx, final_stroke_idx = self.get_subidx(idx)
            blank_img = np.full((256, 256),255, dtype = np.uint8)
            img = blank_img.copy()
            strokes = json.loads(self.raw_data['drawing'][sample_idx])
            for stroke_num in range(final_stroke_idx):
                img = self.add_stroke_to_image(img, strokes[stroke_num])
            partial_img_data = img.copy()
            partial_img_label = self.raw_data['word'][sample_idx]
            just_next_stroke = self.add_stroke_to_image(blank_img, strokes[final_stroke_idx])

            return partial_img_data, partial_img_label, just_next_stroke

# %% Define the Model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        # Only add dropout if the rate is non-zero
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
            # Note: Dropout2d is the correct choice for Conv layers

        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

# --- 2. The Conditional U-Net Model ---
class ArtistModel(nn.Module):
    def __init__(self, img_channels=1, base_c=64,depth = 4, dropout_rate=.2, bert_model_name = 'bert-base-uncased',device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device if hasattr(self, 'device') else 'cpu')
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.embed_dim = self.bert_model.config.hidden_size
        self.depth=depth
        self.device = device
        # Define channel sizes
        self.c_levels = [base_c*(2**i) for i in range(depth)]

        # Create Encoder
        self.encoders = nn.ModuleList([ConvBlock(img_channels if d==0 else self.c_levels[d-1], self.c_levels[d],dropout_rate=dropout_rate) for d in range(depth)])
        self.poll = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. BOTTLENECK (Feature Fusion Point)
        # The input to the bottleneck comes from the image (self.c_levels[-1]) plus the expanded language embedding (embed_dim).
        self.bottleneck = ConvBlock(self.c_levels[-1] + self.embed_dim, self.c_levels[-1],dropout_rate=dropout_rate*2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for d in range(depth-1, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(self.c_levels[d], self.c_levels[d-1], kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(self.c_levels[d-1]+self.c_levels[d], self.c_levels[d-1]))  # times 2 for skip connection

        self.upconvs.append(nn.ConvTranspose2d(self.c_levels[0], self.c_levels[0], kernel_size=2, stride=2))
        self.decoders.append(ConvBlock(self.c_levels[0]*2, self.c_levels[0]))

        self.out_conv = nn.Conv2d(self.c_levels[0] , img_channels, kernel_size=1)

        self.to(self.device)

    @staticmethod
    def to_bw_img(img:torch.Tensor):
        return (img > 0.5).float() * 255

    def to_float_img(self,img:torch.Tensor):
        return img.unsqueeze(1).float().to(self.device) / 255.0

    def forward(self, x, text_input):        # x: (B, 1, 256, 256), lang_embedding: (B, E_DIM)
        c_inner = []
        for enc in self.encoders:
            x = enc(x)
            c_inner.append(x)
            x = self.poll(x)
        encoded_img = x

        tokenized_input = self.tokenizer(
            text_input,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Ensure BERT tensors are on the same device as the image
        tokenized_input = {k: v.to(x.device) for k, v in tokenized_input.items()}

        # Get BERT output (no gradient tracking for initial BERT output)
        bert_output = self.bert_model(**tokenized_input)

        # Extract the contextual embedding for the whole sentence (the [CLS] token)
        # embedding_vector shape: (B, EMBEDDING_DIM=768)
        embedding_vector = bert_output.last_hidden_state[:, 0, :]

        # a. Prepare the Language Embedding for Concatenation
        # Reshape language vector to match feature map dimensions
        B, C, H_f, W_f = encoded_img.shape
        # Expand: (B, E_DIM) -> (B, E_DIM, 1, 1) -> (B, E_DIM, H_f, W_f)
        lang_feature_map = embedding_vector[:, :, None, None].expand(B, self.embed_dim, H_f, W_f)
        # b. Concatenate Image Features and Language Features
        fused_features = torch.cat([encoded_img, lang_feature_map], dim=1) # (B, 512 + E_DIM, 16, 16)
        # c. Bottleneck Convolution
        b = self.bottleneck(fused_features) # (B, 512, 16, 16)

        # --- 3. DECODER ---
        for i in range(self.depth):
            u = self.upconvs[i](b)
            skip_idx = self.depth - 1 - i
            c = c_inner[skip_idx]
            # CROP/PAD SAFETY CHECK (Still important)
            if u.shape[-2:] != c.shape[-2:]:
                diffY = c.size()[2] - u.size()[2]
                diffX = c.size()[3] - u.size()[3]
                c = c[:, :, diffY//2:c.size()[2] - (diffY - diffY//2), diffX//2:c.size()[3] - (diffX - diffX//2)]

            u = torch.cat([u, c], dim=1)
            b = self.decoders[i](u)

        # --- 4. OUTPUT ---
        output_image = torch.sigmoid(self.out_conv(b))

        return output_image

class TVLoss(nn.Module):
    """
    Calculates the Total Variation Loss to encourage spatial smoothness.
    Penalizes the difference between adjacent pixels.
    """
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        # x is the model output (e.g., [B, C, H, W] in the [0, 1] range)

        # 1. Horizontal variation (difference between pixel (i, j) and (i, j+1))
        # We slice the tensor to align pixels for subtraction.
        # This calculates the differences for all but the last column.
        horiz_diff = x[:, :, :, :-1] - x[:, :, :, 1:]

        # 2. Vertical variation (difference between pixel (i, j) and (i+1, j))
        # This calculates the differences for all but the last row.
        vert_diff = x[:, :, :-1, :] - x[:, :, 1:, :]

        # Calculate the L2-norm squared for horizontal and vertical differences
        # We use squaring (L2) instead of absolute value (L1) as it is differentiable
        # and often results in better optimization.
        tv_loss = torch.sum(horiz_diff.pow(2)) + torch.sum(vert_diff.pow(2))

        # Scale by the weight
        return self.weight * tv_loss

class DiceLoss(nn.Module):
    """
    Computes 1 - Dice Score, which is (1 - 2*Intersection / (A + B)).
    Assumes input 'pred' is the probability output (after Sigmoid).
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        # Flatten tensors to operate on pixels, not batches/channels
        prediction = prediction.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # Calculate Intersection (True Positives)
        intersection = (prediction * target).sum()

        # Calculate Union (Sum of all predicted and target pixels)
        dice_sum = prediction.sum() + target.sum()

        # Dice Score: 2 * Intersection / Union
        dice_score = (2. * intersection + self.smooth) / (dice_sum + self.smooth)

        # Dice Loss: 1 - Dice Score
        return 1. - dice_score


# %% Load Data
dataset_path = "data/master_doodle_dataframe.csv"
full_raw_data = pd.read_csv(dataset_path)
full_data = full_raw_data.drop(columns=["countrycode", "recognized", "key_id", "image_path"])
full_label_list = full_data['word'].unique().tolist()
# %% Subset Data
subset_labels = ['apple', 'banana', 'bicycle', 'car', 'cat']
subsample_dataset_ratio = 0.1
train_test_split_ratio = .8
split_seed = 42
batch_size = 4
data = full_data[full_data['word'].isin(subset_labels)].reset_index(drop=True)
data = data.sample(frac=subsample_dataset_ratio, random_state=split_seed).reset_index(drop=True)
train_df = data.sample(frac=train_test_split_ratio, random_state=split_seed)
test_df = data.drop(train_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)
train_set = PartialImageDataset(train_df)
test_set = PartialImageDataset(test_df)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
# %% Build ArtistModel
model_depth = 3
learning_rate = 1e-3
white_ratio = .95
pos_weight_value = (1-white_ratio) / white_ratio
num_epochs = 3
lambda_bin = 0.01
lambda_dice = 2  # Weight for Dice Loss (Good starting point: 0.05 to 0.2)
lambda_tv = 1   # Weight for TV Loss (Start small: 0.005 to 0.05)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = ArtistModel(depth=model_depth)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion_bse = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value))
criterion_dice = DiceLoss()
criterion_tv = TVLoss(weight=1.0)
# %% Train ArtistModel
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (partial_imgs, labels, next_strokes) in enumerate(train_dataloader):
        partial_imgs = model.to_float_img(partial_imgs)
        next_strokes = model.to_float_img(next_strokes)
        optimizer.zero_grad()
        outputs = model(partial_imgs, labels)
        loss_bce  = criterion_bse(outputs, next_strokes)
        loss_dice = criterion_dice(outputs, next_strokes)
        loss_tv = criterion_tv(outputs)
        loss_binary = 4 * torch.mean(outputs * (1 - outputs))
        total_loss = loss_bce + lambda_bin * loss_binary + (lambda_dice * loss_dice) + (lambda_tv * loss_tv)
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {running_loss/100:.4f}")
            running_loss = 0.0

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for partial_imgs, labels, next_strokes in test_dataloader:
            partial_imgs = model.to_float_img(partial_imgs)
            next_strokes = model.to_float_img(next_strokes)
            outputs = model(partial_imgs, labels)
            loss_bce  = criterion_bse(outputs, next_strokes)
            loss_dice = criterion_dice(outputs, next_strokes)
            loss_tv = criterion_tv(outputs)
            loss_binary = 4 * torch.mean(outputs * (1 - outputs))
            total_loss = loss_bce + lambda_bin * loss_binary + (lambda_dice * loss_dice) + (lambda_tv * loss_tv)
            val_loss += total_loss.item()
    val_loss /= len(test_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

# %% Save Model
import time
torch.save(model.state_dict(), f"../jack_cnn_test/saved_models/{time.strftime('%m%d_%H%M%S', time.localtime())}_artist_model.pth")

#%%
weights = torch.load("../jack_cnn_test/saved_models/1006_212659_artist_model.pth", weights_only=True,map_location=device)
model_test = ArtistModel(depth=model_depth).to(device)
model_test.load_state_dict(weights)
partial_imgs, labels, next_strokes =  next(iter(test_dataloader))
partial_imgs = partial_imgs.unsqueeze(1).float().to(device) / 255.0 * 2 - 1
next_strokes = next_strokes.unsqueeze(1).float().to(device) / 255.0 * 2 - 1
outputs = model_test(partial_imgs, labels)
partial_imgs, labels, next_strokes, s =  next(iter(test_dataloader))
outputs = model(model.to_float_img(partial_imgs),labels)
o_img = (outputs > 0.5).squeeze().cpu().detach().float().numpy() * 255