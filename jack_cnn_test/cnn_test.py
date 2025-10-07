# %% IMPORTS
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import cv2

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
    """Two 3x3 convolutions with ReLU and Batch Norm."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# --- 2. The Conditional U-Net Model ---
class ArtistModel(nn.Module):
    def __init__(self, img_channels=1, base_c=64,depth = 4,bert_model_name = 'bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device if hasattr(self, 'device') else 'cpu')
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.embed_dim = self.bert_model.config.hidden_size
        self.depth=depth
        # Define channel sizes
        self.c_levels = [base_c*(2**i) for i in range(depth)]

        # Create Encoder
        self.encoders = nn.ModuleList([ConvBlock(img_channels if d==0 else self.c_levels[d-1], self.c_levels[d]) for d in range(depth)])
        self.poll = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. BOTTLENECK (Feature Fusion Point)
        # The input to the bottleneck comes from the image (self.c_levels[-1]) plus the expanded language embedding (embed_dim).
        self.bottleneck = ConvBlock(self.c_levels[-1] + self.embed_dim, self.c_levels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for d in range(depth-1, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(self.c_levels[d], self.c_levels[d-1], kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(self.c_levels[d-1]+self.c_levels[d], self.c_levels[d-1]))  # times 2 for skip connection

        self.upconvs.append(nn.ConvTranspose2d(self.c_levels[0], self.c_levels[0], kernel_size=2, stride=2))
        self.decoders.append(ConvBlock(self.c_levels[0]*2, self.c_levels[0]))

        self.out_conv = nn.Conv2d(self.c_levels[0] , img_channels, kernel_size=1)

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
        output_image = torch.tanh(self.out_conv(b)) # Use tanh to constrain output to [-1, 1] range

        return output_image



# %% Helper Functions
def add_stroke_to_image(image, stroke, stroke_width=1):
    for i in range(len(stroke[0])-1):
        cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]), (0,0,0), stroke_width)
    return image

# %% Load Data
dataset_path = "data/master_doodle_dataframe.csv"
full_raw_data = pd.read_csv(dataset_path)
full_data = full_raw_data.drop(columns=["countrycode", "recognized", "key_id", "image_path"])
full_label_list = full_data['word'].unique().tolist()
# %% Subset Data
subset_labels = ['apple', 'banana', 'bicycle', 'car', 'cat']
subsample_dataset_ratio = 0.05
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
# %%
test_idx = 100
test_dd = dataset[test_idx:test_idx+2]
test_dd
# %%
cv2.destroyAllWindows()
# %%

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = ArtistModel(depth=3).to(device)

batch_size = 2
dummy_image = torch.randn(batch_size, 1, 256, 256).to(device)
dummy_text_input = ["dog",] * batch_size


output = model(dummy_image, dummy_text_input)
print(output.shape)

# %% Build Model
model_depth = 3
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = ArtistModel(depth=model_depth).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
num_epochs = 5
# %% Train Model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (partial_imgs, labels, next_strokes) in enumerate(train_dataloader):
        partial_imgs = partial_imgs.unsqueeze(1).float().to(device) / 255.0 * 2 - 1 # Normalize to [-1, 1]
        next_strokes = next_strokes.unsqueeze(1).float().to(device) / 255.0 * 2 - 1 # Normalize to [-1, 1]
        optimizer.zero_grad()
        outputs = model(partial_imgs, labels)
        loss = criterion(outputs, next_strokes)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {running_loss/100:.4f}")
            running_loss = 0.0

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for partial_imgs, labels, next_strokes in test_dataloader:
            partial_imgs = partial_imgs.unsqueeze(1).float().to(device) / 255.0 * 2 - 1
            next_strokes = next_strokes.unsqueeze(1).float().to(device) / 255.0 * 2 - 1
            outputs = model(partial_imgs, labels)
            loss = criterion(outputs, next_strokes)
            val_loss += loss.item()
    val_loss /= len(test_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

